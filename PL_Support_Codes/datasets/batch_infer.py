import os
from glob import glob
from datetime import datetime

import torch
import rasterio
import tifffile
import numpy as np
from einops import rearrange

from PL_Support_Codes.utils.utils_image import resize_image
from PL_Support_Codes.datasets.base_dataset import BaseDataset
from PL_Support_Codes.datasets.utils import CropParams, get_crop_slices


class Batch_Infer_Dataset(BaseDataset):

    def __init__(self,
                 root_dir,
                 split,
                 slice_params,
                 eval_region=None,
                 transforms=None,
                 sensor='PS',
                 channels=None,
                 dset_name="batch_infer",
                 seed_num=0,
                 n_classes=3,
                 output_metadata=False,
                 norm_mode=None,
                 dem=False,
                 slope=False,
                 preflood=False,
                 pre_post_difference=False,
                 chirps=False,
                 hand=False,
                 ignore_index=-1,
                 train_split_pct=0.8):
        self.n_classes = n_classes

        super(Batch_Infer_Dataset,
              self).__init__(dset_name,
                             root_dir,
                             split,
                             slice_params,
                             eval_region=eval_region,
                             transforms=transforms,
                             sensor=sensor,
                             channels=channels,
                             seed_num=seed_num,
                             n_classes=n_classes,
                             norm_mode=norm_mode,
                             ignore_index=ignore_index,
                             train_split_pct=train_split_pct)

        self.output_metadata = output_metadata

        # Prepare data depending on sensor.
        self._prepare_data(sensor)

        # Get number of channels.
        self.n_channels = self._get_n_channels()

    def _prepare_data(self, sensor_name):
        
        image_dir = self.root_dir + '\\'


        self.dataset = []
        n_images = 0
        region_name = image_dir.split('\\')[-2]
        print(region_name)
        image_paths = sorted(glob(image_dir + '*SR.tif') + glob(image_dir + '*SR_clip.tif'))
        # image_paths = sorted(glob(image_dir + '*.tif'))
        year = 2019
        month = 10
        day = 1
        # breakpoint()
        # pass

        for image_path in image_paths:
            # Get image size.
            image_info = rasterio.open(image_path)
            height, width = image_info.height, image_info.width

            # Get image name.
            image_name = image_path.split('\\')[-1]
            
            dt = datetime(year, month, day)

            image_crops = get_crop_slices(
                height,
                width,
                self.slice_params.height,
                self.slice_params.width,
                self.slice_params.stride,
                mode="exact",
            )

            for crop in image_crops:
                example = {}
                example["image_path"] = image_path
                example["image_name"] = image_name
                example["region_name"] = region_name
                example["datetime"] = dt
                example["crop_params"] = CropParams(
                    *crop, height, width, self.slice_params.height,
                    self.slice_params.width)

                self.dataset.append(example)
            n_images += 1
        print(f'Number of images in {self.split} dataset: {n_images}')

    def _get_n_channels(self):
        n_channels = {}

        # Get number of channels for multispectral image.
        if self.sensor == 'S2':
            if self.channels == 'RGB':
                n_channels['ms_image'] = 3
            elif self.channels == 'RGB_NIR':
                n_channels['ms_image'] = 4
            elif self.channels == 'ALL':
                n_channels['ms_image'] = 10
            else:
                raise NotImplementedError(
                    f'Cannot get number of S2 channels for channel query "{self.channels}"'
                )
        elif self.sensor == 'PS':
            if self.channels == 'RGB':
                n_channels['ms_image'] = 3
            elif self.channels == 'RGB_NIR':
                n_channels['ms_image'] = 4
            elif self.channels == 'ALL':
                n_channels['ms_image'] = 4
            else:
                raise NotImplementedError(
                    f'Cannot get number of PS channels for channel query "{self.channels}"'
                )
        elif self.sensor == 'S1':
            if self.channels == 'ALL':
                n_channels['ms_image'] = 2
            else:
                raise NotImplementedError(
                    f'Cannot get number of S1 channels for channel query "{self.channels}"'
                )
        elif self.sensor == 'L8':
            if self.channels == 'ALL':
                n_channels['ms_image'] = 7
            else:
                raise NotImplementedError(
                    f'Cannot get number of L8 channels for channel query "{self.channels}"'
                )
        else:
            raise NotImplementedError(
                f'No method for getting number of channels for sensor "{self.sensor}"'
            )

        return n_channels

    def _load_crop_norm_image(self,
                              image_path,
                              crop_params=None,
                              channels='ALL',
                              resize_dims=[None, None],
                              backend='tifffile'):

        if self.sensor == 'S1':
            image = self._load_crop_norm_S1_image(image_path, crop_params,
                                                  channels, resize_dims,
                                                  backend)
        elif self.sensor == 'S2':
            image = self._load_crop_norm_S2_image(image_path, crop_params,
                                                  channels, resize_dims,
                                                  backend)
        elif self.sensor == 'PS':
            image = self._load_crop_norm_PS_image(image_path, crop_params,
                                                  channels, resize_dims,
                                                  backend)
        elif self.sensor == 'L8':
            image = self._load_crop_norm_L8_image(image_path, crop_params,
                                                  channels, resize_dims,
                                                  backend)
        else:
            raise NotImplementedError

        return image

    def _load_crop_norm_S1_image(self,
                                 image_path,
                                 crop_params=None,
                                 channels='ALL',
                                 resize_dims=[None, None],
                                 backend='tifffile'):
        """Load, crop, and normalize S1 image.

        Assumes that the range of S1 images are originally between [-50, 50].

        Args:
            image_path (str): Path to where image is saved on disk.
            crop_params (CropParam): An object containing crop parameters (see CropParam definition).
            channels (str): A string codeword describing which channels should be collected from image.
            resize_dims (list, optional): Dimensions describing what size to resize the image to. Defaults to [None, None].
            backend (str, optional): Which library to use for loading image. Defaults to 'rasterio'.

        Returns:
            np.array: A numpy array of size [channels, height, width].
        """
        # Load image.
        if backend == 'rasterio':
            image = rasterio.open(
                image_path).read()  # [channels, height, width]
            # image = rearrange(image, 'h w c -> c h w')
        elif backend == 'tifffile':
            image = tifffile.imread(image_path)  # [channels, height, width]

        else:
            raise NotImplementedError(
                f'No method for loading image with backend "{backend}"')

        # TODO: hacky, should fix on dataset side
        c, h, w = image.shape
        if (c > h) or (c > w):
            # Image actually as dimensions: [height, width, channels]
            # Expect images to be: [channels, height, width]
            image = rearrange(image, 'h w c -> c h w')

        # TODO: Hotfix
        c = image.shape[0]
        if c > 2:
            image = image[:2]

        # Subselect channels.
        if channels == 'ALL':
            pass
        else:
            raise NotImplementedError(
                f'No method to subselect S1 images with "{channels}" channel query.'
            )

        # Resize image to resize dimensions.
        if (resize_dims[0] is not None) and (resize_dims[1] is not None):
            image = resize_image(image, resize_dims[0], resize_dims[1])

        # Crop image.
        if crop_params is not None:
            image = self._crop_image(image, crop_params)

        # Normalize to [0,1], original range is [-50, 50].
        image = np.clip((image + 50) / 100, 0, 1)
        image = np.nan_to_num(image)

        return image

    def _load_crop_norm_S2_image(self,
                                 image_path,
                                 crop_params=None,
                                 channels='ALL',
                                 resize_dims=[None, None],
                                 backend='tifffile'):
        """Load, crop, and normalize S2 image.

        Expecting image to be saved with label stacked as last band.

        Args:
            image_path (str): Path to where image is saved on disk.
            crop_params (CropParam): An object containing crop parameters (see CropParam definition).
            channels (str): A string codeword describing which channels should be collected from image.
            resize_dims (list, optional): Dimensions describing what size to resize the image to. Defaults to [None, None].
            backend (str, optional): Which library to use for loading image. Defaults to 'rasterio'.

        Returns:
            np.array: A numpy array of size [channels, height, width].
        """
        # Load image.
        if backend == 'rasterio':
            image = rasterio.open(
                image_path).read()  # [channels, height, width]
        elif backend == 'tifffile':
            image = tifffile.imread(image_path)  # [channels, height, width]
        else:
            raise NotImplementedError(
                f'No method for loading image with backend "{backend}"')

        # Subselect channels.
        if channels == 'RGB':
            r_band, g_band, b_band = image[3], image[2], image[1]
            image = np.stack([r_band, g_band, b_band], axis=0)
        elif channels == 'RGB_NIR':
            r_band, g_band, b_band, nir_band = image[3], image[2], image[
                1], image[7]
            image = np.stack([r_band, g_band, b_band, nir_band], axis=0)
        elif channels == 'ALL':
            pass
        else:
            raise NotImplementedError(
                f'No method to subselect S1 images with "{channels}" channel query.'
            )

        # Resize image to resize dimensions.
        if (resize_dims[0] is not None) and (resize_dims[1] is not None):
            image = resize_image(image, resize_dims[0], resize_dims[1])

        # Crop image.
        if crop_params is not None:
            image = self._crop_image(image, crop_params)

        # Normalize to [0,1], original range is [0, 2^16].
        image = np.clip(image / 2**12, 0, 1)

        return image

    def _load_crop_norm_PS_image(self,
                                 image_path,
                                 crop_params=None,
                                 channels='ALL',
                                 resize_dims=[None, None],
                                 backend='tifffile'):
        """Load, crop, and normalize PS image.

        Args:
            image_path (str): Path to where image is saved on disk.
            crop_params (CropParam): An object containing crop parameters (see CropParam definition).
            channels (str): A string codeword describing which channels should be collected from image.
            resize_dims (list, optional): Dimensions describing what size to resize the image to. Defaults to [None, None].
            backend (str, optional): Which library to use for loading image. Defaults to 'rasterio'.

        Returns:
            np.array: A numpy array of size [channels, height, width].
        """
        # Load image.
        if backend == 'rasterio':
            image = rasterio.open(
                image_path).read()  # [height, width, channels]
            image = image.transpose(2, 0, 1)  # [channels, height, width]
        elif backend == 'tifffile':
            image = tifffile.imread(image_path)  # [height, width, channels]
            image = image.transpose(2, 0, 1)  # [channels, height, width]
        else:
            raise NotImplementedError(
                f'No method for loading image with backend "{backend}"')

        # TODO: Hack to handle inconsistent number of channels in PS images.
        if image.shape[0] > 4:
            image = image[:4]

        # Subselect channels.
        if channels == 'RGB':
            r_band, g_band, b_band = image[2], image[1], image[0]
            image = np.stack([r_band, g_band, b_band], axis=0)
        elif channels == 'RGB_NIR':
            r_band, g_band, b_band, nir_band = image[2], image[1], image[
                0], image[3]
            image = np.stack([r_band, g_band, b_band, nir_band], axis=0)
        elif channels == 'ALL':
            pass
        else:
            raise NotImplementedError(
                f'No method to subselect S1 images with "{channels}" channel query.'
            )

        # Resize image to resize dimensions.
        if (resize_dims[0] is not None) and (resize_dims[1] is not None):
            image = resize_image(image, resize_dims[0], resize_dims[1])

        # Crop image.
        if crop_params is not None:
            image = self._crop_image(image, crop_params)

        # Already normalized to [0,1].
        if image.dtype == 'uint16':
            image = image / 2**16

        return image

    def _load_crop_norm_L8_image(self,
                                 image_path,
                                 crop_params=None,
                                 channels='ALL',
                                 resize_dims=[None, None],
                                 backend='tifffile'):
        """Load, crop, and normalize L8 image.

        Args:
            image_path (str): Path to where image is saved on disk.
            crop_params (CropParam): An object containing crop parameters (see CropParam definition).
            channels (str): A string codeword describing which channels should be collected from image.
            resize_dims (list, optional): Dimensions describing what size to resize the image to. Defaults to [None, None].
            backend (str, optional): Which library to use for loading image. Defaults to 'rasterio'.

        Returns:
            np.array: A numpy array of size [channels, height, width].
        """
        # Load image.
        if backend == 'rasterio':
            image = rasterio.open(
                image_path).read()  # [channels, height, width]
        elif backend == 'tifffile':
            image = tifffile.imread(image_path)  # [channels, height, width]
        else:
            raise NotImplementedError(
                f'No method for loading image with backend "{backend}"')

        # Subselect channels.
        if channels == 'RGB':
            breakpoint()
            r_band, g_band, b_band = image[2], image[1], image[0]
            image = np.stack([r_band, g_band, b_band], axis=0)
        elif channels == 'RGB_NIR':
            breakpoint()
            r_band, g_band, b_band, nir_band = image[2], image[1], image[
                0], image[3]
            image = np.stack([r_band, g_band, b_band, nir_band], axis=0)
        elif channels == 'ALL':
            pass
        else:
            raise NotImplementedError(
                f'No method to subselect S1 images with "{channels}" channel query.'
            )

        # Resize image to resize dimensions.
        if (resize_dims[0] is not None) and (resize_dims[1] is not None):
            image = resize_image(image, resize_dims[0], resize_dims[1])

        # Crop image.
        if crop_params is not None:
            image = self._crop_image(image, crop_params)

        # TODO: Already normalized to [0,1].
        image = np.clip(image, 0, 18607.72) / 18607.72

        return image

    def __getitem__(self, index, output_metadata=False):
        example = self.dataset[index]
        crop_params = example['crop_params']

        image = self._load_crop_norm_image(
            example['image_path'],
            crop_params,
            self.channels,
            resize_dims=[crop_params.og_height, crop_params.og_width])

        # Normalize by norm_params.
        image, mean, std = self.normalize(image, self.sensor)

        # Add buffer to image and label.
        image = self._add_buffer_to_image(image, crop_params.max_crop_height,
                                          crop_params.max_crop_width)

        output = {}
        output["image"] = torch.tensor(image).float()
        output['mean'] = mean
        output['std'] = std

        if output_metadata or self.output_metadata:
            output["metadata"] = {
                "image_path": example["image_path"],
                "crop_params": example["crop_params"],
                "datetime": example["datetime"]
            }
            if 'region_name' in example.keys():
                output['metadata']["region_name"] = example["region_name"]

        return output

    def _to_RGB_PS(self, image, gamma=0.6):
        image *= 2*4

        # Get RGB bands.
        if self.channels == 'RGB':
            red_band = image[0]
            green_band = image[1]
            blue_band = image[2]
        elif self.channels == 'RGB_NIR':
            red_band = image[0]
            green_band = image[1]
            blue_band = image[2]
        elif self.channels == 'ALL':
            red_band = image[2]
            green_band = image[1]
            blue_band = image[0]
        else:
            raise NotImplementedError

        # Adjust gamma values.
        red_band = red_band**(gamma)
        green_band = green_band**(gamma)
        blue_band = blue_band**(gamma)

        # Combine RGB bands.
        rgb_image = np.stack([red_band, green_band, blue_band], axis=2)

        return rgb_image


if __name__ == '__main__':
    import argparse
    from copy import deepcopy

    from tqdm import tqdm
    from PIL import Image

    from utils import get_dset_path, generate_image_slice_object

    parser = argparse.ArgumentParser()
    parser.add_argument('--ex_indices', type=int, default=[0], nargs='+')
    parser.add_argument('--split', type=str, default='all')
    parser.add_argument('--crop_size', type=int, default=1024)
    parser.add_argument('--sensor', type=str, default='PS')
    parser.add_argument('--eval_region',
                        type=str,
                        default=None,
                        help='Bolivia')
    parser.add_argument('--channels', type=str, default='ALL')
    args = parser.parse_args()

    dset_name = "batch_infer"
    root_dir = get_dset_path(dset_name)
    slice_params = generate_image_slice_object(args.crop_size,
                                               args.crop_size,
                                               scale=1,
                                               stride=args.crop_size//2)
    # Load dataset object.
    dataset = Batch_Infer_Dataset(root_dir,
                                     args.split,
                                     slice_params,
                                     sensor=args.sensor,
                                     channels=args.channels,
                                     eval_region=args.eval_region,
                                     dem=False,
                                     output_metadata=True,
                                     slope=False)
    # Get an example.
    # for index in tqdm(args.ex_indices):
    #     example = dataset.__getitem__(index)

    #     # Create an RGB version of image.
    #     example["image"] *= 2**2
    #     rgb_image = dataset.to_RGB(example["image"], gamma=0.5)

    #     # Save image
    #     Image.fromarray((rgb_image * 255).astype('uint8')).save(f"./{dset_name}_{example['metadata']['region_name']}_{dataset.sensor}_{index}.png")

    # Test stitching images together.
    from PL_Support_Codes.utils.utils_image import ImageStitcher_v2 as ImageStitcher
    n_examples = dataset.__len__()
    float_stitchers = {}
    for i in tqdm(range(n_examples)):
        example = dataset.__getitem__(i)
        if True:
            breakpoint()
            pass

        image = example['image']
        metadata = example['metadata']
        region_name = metadata['region_name']

        image_name = os.path.splitext(os.path.split(metadata['image_path'])[1])[0]
        if region_name not in float_stitchers.keys():
            float_stitchers[region_name] = ImageStitcher('.')
        
        image = np.asarray(image).transpose(1,2,0)
        float_stitchers[region_name].add_image(image, image_name, metadata['crop_params'], metadata['crop_params'].og_height, metadata['crop_params'].og_width)


    # Turn it into an RGB image.
    for reigon_name, float_stitcher in float_stitchers.items():
        combined_image = float_stitcher.get_combined_images()[image_name]

        combined_image *= 2**2
        rgb_image = dataset.to_RGB(combined_image.transpose(2,0,1), gamma=0.5)

        save_path = f'./stitch_test_2_{region_name}.png'
        Image.fromarray((rgb_image*255).astype('uint8')).save(save_path)

