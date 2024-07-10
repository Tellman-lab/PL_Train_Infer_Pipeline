import os
import random
from glob import glob
from copy import deepcopy

import cv2
import torch
import rasterio
import omegaconf
import numpy as np
from einops import rearrange
from tifffile import tifffile

from PL_Support_Codes.utils.utils_image import resize_image
from PL_Support_Codes.datasets.base_dataset import BaseDataset
from PL_Support_Codes.datasets.utils import CropParams, get_crop_slices


class S1F11(BaseDataset):

    def __init__(self,
                 root_dir,
                 split,
                 slice_params,
                 eval_region=None,
                 transforms=None,
                 sensor='S1',
                 channels=None,
                 dset_name="s1f11",
                 seed_num=0,
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

        self.dem = dem
        self.hand = hand
        self.slope = slope
        self.chirps = chirps
        self.preflood = preflood
        self.pre_post_difference = pre_post_difference

        super(S1F11, self).__init__(dset_name,
                                               root_dir,
                                               split,
                                               slice_params,
                                               eval_region=eval_region,
                                               transforms=transforms,
                                               sensor=sensor,
                                               channels=channels,
                                               seed_num=seed_num,
                                               norm_mode=norm_mode,
                                               ignore_index=ignore_index,
                                               train_split_pct=train_split_pct)

        self.n_classes = 3
        self.output_metadata = output_metadata

        # Prepare data depending on sensor.
        self._prepare_data(sensor)

        # Get number of channels.
        self.n_channels = self._get_n_channels()

    def _prepare_data(self, sensor_name):
        # thp_region_dirs = sorted(
        #     glob(os.path.join(self.root_dir, 'THP') + "/*/"))
        # thp_region_names = [p.split('/')[-2] for p in thp_region_dirs]
        # breakpoint()
        csdap_region_dirs = sorted(
            glob(self.root_dir + "/*/"))
        print()
        csdap_region_names = [p.split('/')[-2] for p in csdap_region_dirs]

        # region_names = thp_region_names + csdap_region_names
        # region_dirs = thp_region_dirs + csdap_region_dirs
        region_names = csdap_region_names
        region_dirs = csdap_region_dirs

        region_dirs_dict = {}
        for region_name, region_dir in zip(region_names, region_dirs):
            region_dirs_dict[region_name] = region_dir

        image_paths = self._split_data(region_dirs_dict, sensor_name)

        n_images = 0
        self.dataset = []
        for image_path, region_name in image_paths:
            image_name = os.path.splitext(os.path.split(image_path)[1])[0]

            # # Get label path.
            # label_path = os.path.join('/'.join(image_path.split('/')[:-3]),
            #                           'labels', image_name + '.tif')
            
            # Get label path for Blacksky
            label_path = os.path.join('/'.join(image_path.split('/')[:-3]),
                                      'labels', image_name + '.tif')
                                      
            if os.path.exists(label_path) is False:
                breakpoint()
                pass

            # TODO: Get label width and height.
            # label_info = rasterio.open(label_path)
            # TODO: hack way to not resize the input when predicting
            # TODO: 
            # # this makes sure that we do not alter the resolution of the input
            label_info = rasterio.open(image_path)
            label_height, label_width = label_info.height, label_info.width
            # TODO: 
            # TODO: 

            if self.dem:
                raise NotImplementedError(
                    f'DEM finding not implemented for "{self.dset_name}" dataset.'
                )

            if self.slope:
                raise NotImplementedError(
                    f'SLOPE finding not implemented for "{self.dset_name}" dataset.'
                )

            image_crops = get_crop_slices(
                label_height,
                label_width,
                self.slice_params.height,
                self.slice_params.width,
                self.slice_params.stride,
                mode="exact",
            )

            for image_crop in image_crops:
                # Create example object.
                example = {}
                example["image_path"] = image_path
                example["label_path"] = label_path
                example["region_name"] = region_name
                example["crop_params"] = CropParams(*image_crop, label_height,
                                                    label_width,
                                                    self.slice_params.height,
                                                    self.slice_params.width)

                self.dataset.append(example)
            n_images += 1
        print(f'Number of images in {self.split} dataset: {n_images}')

        self.image_paths = image_paths

    def _split_data(self, region_dirs, sensor_name):
        if len(region_dirs.keys()) == 0:
            raise ValueError(
                f'No regions found for dataset "{self.dset_name}" and sensor "{self.sensor}"'
            )

        image_paths = []
        if self.eval_region is None:
            # Get image paths from all regions.
            pass
        else:
            # Check type of eval_region input.
            if isinstance(self.eval_region, str):
                self.eval_region = [self.eval_region]

            if isinstance(self.eval_region,
                          (list, omegaconf.listconfig.ListConfig)) is False:
                raise ValueError(
                    f'Eval regions variable is not a list but a {type(self.eval_region)}'
                )

            # Split data by regions instead of by images.
            if self.split == 'train':
                # Get all region names.
                region_names = list(region_dirs.keys())

                # Check that validation region name is included in the found regions.
                for eval_region in self.eval_region:
                    if eval_region not in region_names:
                        raise ValueError(
                            f'Eval region {eval_region} not found in avilable regions {region_names}'
                        )

                # Remove eval region from regions names.
                for eval_region in self.eval_region:
                    del region_dirs[eval_region]

            elif self.split in ['valid', 'test']:
                # Get all region names.
                region_names = list(region_dirs.keys())

                # Check that validation region name is included in available regions.
                for eval_region in self.eval_region:
                    if eval_region not in region_names:
                        raise ValueError(
                            f'Eval region {eval_region} not found in avilable regions {region_names}'
                        )

                # Get region directories of eval regions.
                sub_region_dirs = {}
                for eval_region in self.eval_region:
                    sub_region_dirs[eval_region] = region_dirs[eval_region]
                region_dirs = sub_region_dirs
            elif self.split == 'all':
                pass
            else:
                raise ValueError(
                    f'Cannot handle split "{self.split}" for splitting data by region.'
                )

        # Get image dir and pair with region name.
        for region_name, region_dir in region_dirs.items():
            region_image_paths = glob(region_dir + f'/{sensor_name}/*.tif')
            if len(region_image_paths) == 0:
                continue
            for image_dir in region_image_paths:
                image_paths.append([image_dir, region_name])

        if self.eval_region is None:
            # Split data randomly.
            random.shuffle(image_paths)

            n_image_paths = len(image_paths)

            n_train_image_paths = int(n_image_paths * self.train_split_pct)

            if self.split == 'train':
                image_paths = image_paths[:n_train_image_paths]
            else:
                image_paths = image_paths[n_train_image_paths:]

        if len(image_paths) == 0:
            raise ValueError(
                f'No images found for eval regions "{self.eval_region}" and sensor "{self.sensor}"'
            )

        print(
            f'{self.split.capitalize()} split: {len(image_paths)} images found'
        )

        return image_paths

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
        
        elif self.sensor == 'S1':
            if self.channels == 'ALL':
                n_channels['ms_image'] = 2
            else:
                raise NotImplementedError(
                    f'Cannot get number of S1 channels for channel query "{self.channels}"'
                )
        
        else:
            raise NotImplementedError(
                f'No method for getting number of channels for sensor "{self.sensor}"'
            )

        # Get number of channels for auxiliary data.
        if self.dem:
            n_channels['dem'] = 1

        if self.slope:
            n_channels['slope'] = 1

        return n_channels

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
# TODO: this place resize the input to the size of label
        # # Resize image to resize dimensions.
        # if (resize_dims[0] is not None) and (resize_dims[1] is not None):
        #     image = resize_image(image, resize_dims[0], resize_dims[1])
# TODO: 

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

        # # TODO: Hotfix
        # c = image.shape[0]
        # if c > 2:
        #     image = image[:2]

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

        # # TODO: Resize image to resize dimensions.
        # if (resize_dims[0] is not None) and (resize_dims[1] is not None):
        #     image = resize_image(image, resize_dims[0], resize_dims[1])

        # Crop image.
        if crop_params is not None:
            image = self._crop_image(image, crop_params)

        # Normalize to [0,1], original range is [0, 2^16].
        image = np.clip(image / 2**12, 0, 1)

        return image

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
        
        else:
            raise NotImplementedError

        return image

    def _load_label_image(self,
                          label_path,
                          desired_height,
                          desired_width,
                          crop_params,
                          backend='tifffile'):
        # Load label.
        if backend == 'rasterio':
            label_dataset = rasterio.open(label_path)
            label = label_dataset.read()  # [height, width]
        elif backend == 'tifffile':
            label = tifffile.imread(label_path)  # [height, width]
        else:
            raise NotImplementedError(
                f'No method for loading image with backend "{backend}"')

        # Resize image if not desired resolution.
        height, width = label.shape
        if (height != desired_height) or (width != desired_width):
            label = resize_image(label,
                                 desired_height,
                                 desired_width,
                                 resize_mode=cv2.INTER_NEAREST)

        # Crop label image.
        label = self._crop_image(label, crop_params)

        # Binarize label values to not-flood-water (0) and flood-water (1).
        height, width = label.shape
        binary_label = np.zeros([height, width], dtype='uint8')
        
        # Value mapping for s1f11:
        # -1: No data (ignore) -> Ignore index
        # 0: No flood -> 0
        # 1: Water -> 1

        # Value mapping for impact:
        # 15: No data (ignore) -> Ignore index
        # 0: No flood -> 0
        # 255: Water -> 1
        # Get positive water label.
        x, y = np.where(label == 255)
        binary_label[x, y] = 1
        # Get ignore label.
        
        x, y = np.where((label == 15))

        binary_label[x, y] = self.ignore_index

        # if binary_label.max() > 1:
        #     breakpoint()
        # # Value mapping for IMPACT:
        # # 15: No data (ignore) -> Ignore index
        # # 0: No flood -> 0
        # # 255: Water -> 1
        # # Get positive water label.
        # x, y = np.where(label == 255)
        # binary_label[x, y] = 1
        # # Get ignore label.
        # x, y = np.where((label == 15))
        # binary_label[x, y] = self.ignore_index

        return binary_label

    def __getitem__(self, index, output_metadata=False):
        example = self.dataset[index]
        crop_params = example['crop_params']

        # Load image
        image = self._load_crop_norm_image(
            example['image_path'],
            crop_params,
            self.channels,
            resize_dims=[crop_params.og_height, crop_params.og_width])

        # Load label.
        target = self._load_label_image(example['label_path'],
                                        crop_params.og_height,
                                        crop_params.og_width, crop_params)

        # Normalize by norm_params.
        image, mean, std = self.normalize(image, self.sensor)

        # Add buffer to image and label.
        image = self._add_buffer_to_image(image, crop_params.max_crop_height,
                                          crop_params.max_crop_width)
        # breakpoint()
        target = self._add_buffer_to_image(target,
                                           crop_params.max_crop_height,
                                           crop_params.max_crop_width,
                                           constant_value=self.ignore_index)
        
        # Apply transforms.
        if self.transforms is not None:
            active_transforms = self.sample_transforms()
            image = self.apply_transforms(image, active_transforms, is_anno=False)
            target = self.apply_transforms(target, active_transforms, is_anno=True)
        else:
            image = torch.tensor(image)
            target = torch.tensor(target)

        # Get correct type.
        image = image.float()
        target = target.long()

        output = {}
        output["image"] = image
        output["target"] = target
        output['mean'] = mean
        output['std'] = std

        if output_metadata:
            output["metadata"] = {
                "image_path": example["image_path"],
                "crop_params": example["crop_params"]
            }
            if 'region_name' in example.keys():
                output['metadata']["region_name"] = example["region_name"]

        return output


def test_image_transforms(root_dir, slice_params, args):
    from omegaconf import OmegaConf
    no_t_dataset = Combined_Dataset(root_dir,
                               args.split,
                               slice_params,
                               sensor=args.sensor,
                               channels=args.channels,
                               eval_region=args.eval_region,
                               dem=False,
                               slope=False)
    
    conf = OmegaConf.create({'hflip': {'active': True, 'likelihood': 0}, 
                            'vflip': {'active': True, 'likelihood': 0}, 
                            'rotate': {'active': True,
                                       'likelihood': 1,
                                        'min_rot_angle': 0, 
                                        'max_rot_angle': 360}})

    t_dataset = Combined_Dataset(root_dir,
                               args.split,
                               slice_params,
                               sensor=args.sensor,
                               channels=args.channels,
                               eval_region=args.eval_region,
                               transforms=conf,
                               dem=False,
                               slope=False)

    index = 0
    example = no_t_dataset.__getitem__(index)

    # Create an RGB version of image.
    rgb_image = no_t_dataset.to_RGB(example["image"])
    mask = example["target"]

    # Visualize RGB image w/ and w/o overlay (gif).
    rgb_overlay = deepcopy(rgb_image)
    x, y = np.where(mask == 1)
    rgb_overlay[x, y, :] = np.asarray([0, 1, 1])
    x, y = np.where(mask == 2)
    rgb_overlay[x, y, :] = np.asarray([1, 0, 0])

    rgb_image = (rgb_image * 255).astype("uint8")
    rgb_overlay = (rgb_overlay * 255).astype("uint8")

    create_gif([rgb_image, rgb_overlay],
                f"./{dset_name}_test_trans_{no_t_dataset.sensor}_{index}_no_T.gif")

    example = t_dataset.__getitem__(index)

    # Create an RGB version of image.
    rgb_image = t_dataset.to_RGB(example["image"])
    mask = example["target"]

    # Visualize RGB image w/ and w/o overlay (gif).
    rgb_overlay = deepcopy(rgb_image)
    x, y = np.where(mask == 1)
    rgb_overlay[x, y, :] = np.asarray([0, 1, 1])
    x, y = np.where(mask == 2)
    rgb_overlay[x, y, :] = np.asarray([1, 0, 0])

    rgb_image = (rgb_image * 255).astype("uint8")
    rgb_overlay = (rgb_overlay * 255).astype("uint8")

    create_gif([rgb_image, rgb_overlay],
                f"./{dset_name}_test_trans_{t_dataset.sensor}_{index}_T.gif")

if __name__ == "__main__":
    import argparse
    from copy import deepcopy

    from tqdm import tqdm

    from st_water_seg.tools import create_gif
    from utils import get_dset_path, generate_image_slice_object

    parser = argparse.ArgumentParser()
    parser.add_argument('--ex_indices', type=int, default=[0], nargs='+')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--crop_size', type=int, default=1024)
    parser.add_argument('--sensor', type=str, default='S2')
    parser.add_argument('--eval_region',
                        type=str,
                        default=None,
                        help='Bolivia')
    parser.add_argument('--channels', type=str, default='ALL')
    args = parser.parse_args()

    dset_name = "combined"
    root_dir = get_dset_path(dset_name)
    slice_params = generate_image_slice_object(args.crop_size,
                                               args.crop_size,
                                               scale=1,
                                               stride=None)
    
    test_image_transforms(root_dir, slice_params, args)
