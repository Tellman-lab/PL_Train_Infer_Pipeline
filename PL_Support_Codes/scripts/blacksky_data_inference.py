import os
import argparse
from glob import glob

import cv2
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
from tifffile import tifffile
from torch.utils.data import DataLoader

from st_water_seg.models import build_model
from st_water_seg.utils.utils_misc import load_config
from st_water_seg.utils.utils_image import ImageStitcher_v2
from st_water_seg.datasets.utils import generate_image_slice_object, CropParams, get_crop_slices


def custom_collate_fn(data):
    out_data = {}
    example = data[0]
    for key in example.keys():
        if key not in ['crop_params', 'image_name']:
            out = []
            for ex in data:
                out.append(ex[key])
            out = torch.stack(out, 0)
        else:
            out = []
            for ex in data:
                out.append(ex[key])
        out_data[key] = out
    return out_data


def compute_norm_params(image_paths, rnd_pct=0.001):

    values = []
    for image_path in tqdm(image_paths,
                           desc='Gathering pixel values',
                           colour='green'):
        image = tifffile.imread(image_path)[:, :, :3]  # [H, W, 3]
        pixels = rearrange(image, 'h w c -> (h w) c')
        indices = np.random.randint(0,
                                    pixels.shape[0],
                                    size=int(pixels.shape[0] * rnd_pct))
        sub_pixels = np.take(pixels, indices, axis=0)
        values.append(sub_pixels)

    all_values = np.concatenate(values, axis=0)
    mean = all_values.mean(axis=0) / 2**16
    std = all_values.std(axis=0) / 2**16
    return mean, std


class BlackSkyDataset():

    def __init__(self, crop_params, mean, std, resize_to_3m=False):
        self.mean, self.std = mean, std
        self.resize_to_3m = resize_to_3m

        # Get paths to images.
        self.crop_params = crop_params
        root_dir = '/media/mule/Projects/NASA/BlackSky/Data/chipped/'
        image_paths = sorted(glob(os.path.join(root_dir, 'imgs') + '/*.tif'))
        label_paths = sorted(glob(os.path.join(root_dir, 'labels') + '/*.tif'))

        # Generate examples.
        self.dataset = []
        for img_path, lbl_path in tqdm(zip(image_paths, label_paths),
                                       desc='Creating examples'):
            # Get image name.
            img_name = os.path.splitext(os.path.split(img_path)[1])[0]

            # Get size of image.
            # img_ds = rasterio.open(img_path)
            img = tifffile.imread(img_path)
            if resize_to_3m:
                height = int(img.shape[0] / 3)
                width = int(img.shape[1] / 3)
            else:
                height = img.shape[0]
                width = img.shape[1]

            # Create crop slices
            image_crops = get_crop_slices(
                height,
                width,
                self.crop_params.height,
                self.crop_params.width,
                self.crop_params.stride,
                mode="exact",
            )

            for image_crop in image_crops:
                example = {}
                example['image_path'] = img_path
                example['label_path'] = lbl_path
                example['crop_params'] = CropParams(*image_crop, height, width,
                                                    self.crop_params.height,
                                                    self.crop_params.width)
                example['image_name'] = img_name
                self.dataset.append(example)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        crop_params = example['crop_params']

        # Load image.
        image = self.load_crop_norm_image(example['image_path'],
                                          crop_params=crop_params,
                                          channels='RGB')

        # Load label.
        target = self.load_label_image(example['label_path'], crop_params)

        # Normalize image by mean and std.
        image = (image - mean[:, None, None]) / std[:, None, None]

        # Add buffer to image and label.
        image = self._add_buffer_to_image(image, crop_params.max_crop_height,
                                          crop_params.max_crop_width)
        target = self._add_buffer_to_image(target,
                                           crop_params.max_crop_height,
                                           crop_params.max_crop_width,
                                           constant_value=0)
        # Return output.
        output = {}
        output["image"] = torch.tensor(image.astype('float32'))
        output["target"] = torch.tensor(target.astype('int64'))
        output['image_name'] = example['image_name']
        output['crop_params'] = crop_params

        return output

    def _crop_image(self, image, crop_params):
        h0, w0 = crop_params.h0, crop_params.w0
        hE, wE = crop_params.hE, crop_params.wE

        # Get image dimensions.
        n_dims = len(image.shape)
        if n_dims == 2:
            crop = image[h0:hE, w0:wE]
        elif n_dims == 3:
            crop = image[:, h0:hE, w0:wE]
        else:
            raise NotImplementedError(
                f'Cannot crop image with "{n_dims}" dimensions.')

        return crop

    def _add_buffer_to_image(self,
                             image,
                             desired_height,
                             desired_width,
                             buffer_mode='constant',
                             constant_value=0):
        """Extend image to desired size if smaller in resolution.

        Do nothing if the image is larger than the desired size.

        Args:
            image (np.array): A numpy array of shape [height, width] or [channels, height, width].
            desired_height (int): Desired height the image should be.
            desired_width (int): Desired width the image should be.
            buffer_mode (str, optional): Method mode for determining how to fill buffered regions. Defaults to 'constant'.
            constant_value (int, optional): For constant method, what value to assign to default canvas value. Defaults to 0.

        Raises:
            NotImplementedError: No method to handle images with number of dimensions other than 2 or 3. 
            NotImplementedError: No method to handle images with number of dimensions other than 2 or 3. 

        Returns:
            np.array: A numpy array of shape [desired_height, desired_width].
        """
        # Get image dimensions.
        n_dims = len(image.shape)
        if n_dims == 2:
            image_height, image_width = image.shape
        elif n_dims == 3:
            n_channels, image_height, image_width = image.shape
        else:
            raise NotImplementedError(
                f'Cannot add buffer to image with "{n_dims}" dimensions.')

        # Check if image is smaller than desired resolution.
        if (image_height >= desired_height) and (image_width >= desired_width):
            return image
        else:
            if buffer_mode == 'constant':
                # Create buffer canvas.
                if n_dims == 2:
                    buffer_canvas = np.ones([desired_height, desired_width],
                                            dtype=image.dtype) * constant_value
                    buffer_canvas[:image_height, :image_width] = image
                elif n_dims == 3:
                    buffer_canvas = np.ones(
                        [n_channels, desired_height, desired_width],
                        dtype=image.dtype) * constant_value
                    buffer_canvas[:, :image_height, :image_width] = image
                image = buffer_canvas
            else:
                raise NotImplementedError(
                    f'No method to handle buffer mode of "{buffer_mode}"')

        return image

    def load_crop_norm_image(self,
                             image_path,
                             crop_params=None,
                             channels='RGB',
                             backend='tifffile'):
        # Load image.
        image = tifffile.imread(image_path)[:, :, :3]

        if self.resize_to_3m:
            rH, rW = int(image.shape[0] / 3), int(image.shape[1] / 3)
            image = cv2.resize(image, (rW, rH), interpolation=cv2.INTER_LINEAR)
        image = rearrange(image, 'h w c -> c h w')

        # Take a crop.
        if crop_params is not None:
            image = self._crop_image(image, crop_params)

        # Already normalized to [0,1].
        if image.dtype == 'uint16':
            image = image / 2**16

        return image

    def load_label_image(self, label_path, crop_params):
        # Load label.
        label = tifffile.imread(label_path)

        # Resize.
        if self.resize_to_3m:
            rH, rW = int(label.shape[0] / 3), int(label.shape[1] / 3)
            label = cv2.resize(label, (rW, rH),
                               interpolation=cv2.INTER_NEAREST)

        # Crop label image.
        label = self._crop_image(label, crop_params)

        # Binarize label values to not-flood-water (0) and flood-water (1).
        height, width = label.shape
        binary_label = np.zeros([height, width], dtype='uint8')
        # Value mapping:
        # 1: No water
        # 2: Low confidence water
        # 3: High confidence water

        # Get positive water label.
        # x, y = np.where((label == 2) | (label == 3))
        # binary_label[x, y] = 1

        x, y = np.where(label == 3)
        binary_label[x, y] = 1

        # # Get ignore label.
        # x, y = np.where((label == 0))
        # binary_label[x, y] = self.ignore_index

        return binary_label


def to_rgb(image):
    pass


def predict(loader, model, device, resample_to_3m=False):
    if resample_to_3m:
        res_str = '3m'
    else:
        res_str = 'trained3m_tested1m'

    base_dir = '/media/mule/Projects/NASA/BlackSky/Data/chipped/'
    save_dir = os.path.join(base_dir, res_str)
    os.makedirs(save_dir, exist_ok=True)

    pred_stitcher = ImageStitcher_v2(save_dir,
                                     '',
                                     save_backend='tifffile',
                                     save_ext='.tif')

    with torch.no_grad():
        for data in tqdm(loader, desc='Inference'):
            # Prepare data.
            data['image'] = data['image'].to(device)

            # Get predictions.
            pred = model(data)

            # Add predictions to image stitcher.
            pred = pred.detach().cpu().numpy()
            pred = rearrange(pred, 'b c h w -> b h w c')
            batch_size = data['image'].shape[0]

            for i in range(batch_size):
                pred_stitcher.add_image(pred[i], data['image_name'][i],
                                        data['crop_params'][i],
                                        data['crop_params'][i].og_height,
                                        data['crop_params'][i].og_width)

        combined_images = pred_stitcher.get_combined_images()
        for image_name, image in combined_images.items():
            pred = image[:, :, :2].argmax(axis=2)
            save_path = os.path.join(save_dir, image_name + '.tif')
            tifffile.imwrite(save_path, pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resample_to_3m', default=False, action='store_true')
    args = parser.parse_args()

    # Load model with checkpoint weights.
    # USA model
    # exp_dir = '/home/mpurri/research/spatial_temporal_water_seg/outputs/2023-01-15/13-11-23/'
    # model_name = 'model-epoch=13-valid_iou=0.790.ckpt'

    # Nepal model
    exp_dir = '/home/zhijiezhang/spatial_temporal_water_seg/outputs/2023-06-06/3m_label_Tulsa_low+no_3band/'
    model_name = 'model-epoch=07-val_MulticlassJaccardIndex=0.8513.ckpt'

    ## Get config file.
    cfg_path = os.path.join(exp_dir, 'hydra', 'config.yaml')
    cfg = load_config(cfg_path)

    ## Get checkpoint path.
    checkpoint_path = os.path.join(exp_dir, 'checkpoints', model_name)

    ## Initialize model.
    n_in_channels = {'ms_image': 3}  # RGB
    n_out_channels = 3  # Water/no-water
    model = build_model(cfg.model.name,
                        n_in_channels,
                        n_out_channels,
                        cfg.lr,
                        log_image_iter=cfg.log_image_iter,
                        to_rgb_fcn=None,
                        ignore_index=0,
                        **cfg.model.model_kwargs)

    model = model.load_from_checkpoint(checkpoint_path,
                                       in_channels=n_in_channels,
                                       n_classes=n_out_channels,
                                       lr=cfg.lr)
    model._set_model_to_eval()

    # Get device.
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = model.to(device)

    # Compute mean and std (once and save).
    # root_dir = '/media/mule/Projects/NASA/THP/blksky/'
    # image_paths = sorted(glob(os.path.join(root_dir, 'imgs') + '/*.tif'))
    # mean, std = compute_norm_params(image_paths)
    # mean = np.array([0.02841095, 0.02589743, 0.02507488])
    # std = np.array([0.01279131, 0.0105446 , 0.01000391])
    mean = np.array([0, 0, 0])
    std = np.array([1.0, 1.0, 1.0])

    # Create Dataset class.
    slice_params = generate_image_slice_object(cfg.crop_height, cfg.crop_width,
                                               cfg.crop_stride)
    dataset = BlackSkyDataset(slice_params,
                              mean,
                              std,
                              resize_to_3m=args.resample_to_3m)
    # inference_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.n_workers, collate_fn=custom_collate_fn)
    inference_loader = DataLoader(dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  collate_fn=custom_collate_fn)

    # Infer on blacksky data.
    predict(inference_loader, model, device, args.resample_to_3m)
