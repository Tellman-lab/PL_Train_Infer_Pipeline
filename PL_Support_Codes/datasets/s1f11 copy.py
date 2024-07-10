import os
import random
from glob import glob

import cv2
import torch
import rasterio
import tifffile
import omegaconf
import numpy as np

from st_water_seg.utils.utils_image import resize_image
from st_water_seg.datasets.base_dataset import BaseDataset
from st_water_seg.datasets.utils import CropParams, get_crop_slices


class S1Floods(BaseDataset):

    def __init__(
        self,
        root_dir,
        split,
        slice_params,
        eval_region=None,
        transforms=None,
        sensor='S2',
        channels=None,
        dset_name='s1floods',
        seed_num=0,
        output_metadata=False,
        norm_mode=None,
        ignore_index=-1,
        train_split_pct=0.8,
        dem=False,
        slope=False,
        preflood=False,
        pre_post_difference=False,
        chirps=False,
        hand=False,
    ):
        super(S1Floods, self).__init__(dset_name,
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

        self.n_classes = 3  # no-data, no-water, water
        self.output_metadata = output_metadata

        # Prepare data depending on sensor.
        self._prepare_data()

        # Get number of channels.
        self.n_channels = self._get_n_channels()

    def _get_n_channels(self):
        n_channels = {}

        # Get number of channels for multispectral image.
        if self.sensor == 'S2':
            if self.channels == 'RGB':
                n_channels['ms_image'] = 3
            elif self.channels == 'RGB_NIR':
                n_channels['ms_image'] = 4
            elif self.channels == 'ALL':
                n_channels['ms_image'] = 13
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
        # NOTE: No auxiliary data for this dataset.

        return n_channels

    def _prepare_data(self):

        sensor_to_prepare_data_func = {
            'S2': self._prepare_S2_data,
            'S1': self._prepare_S1_data
        }

        try:
            sensor_to_prepare_data_func[self.sensor](self.root_dir)
        except KeyError:
            raise NotImplementedError(
                f'Data preperation method not created for sensor "{self.sensor}"'
            )

    def _split_data(self, region_to_paths_dict):
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
                region_names = list(region_to_paths_dict.keys())

                # Check that validation region name is included in the found regions.
                for eval_region in self.eval_region:
                    if eval_region not in region_names:
                        raise ValueError(
                            f'Eval region {eval_region} not found in avilable regions {region_names}'
                        )
                # Remove eval region from regions names.
                for eval_region in self.eval_region:
                    del region_to_paths_dict[eval_region]

            elif self.split in ['valid', 'test']:
                # Get all region names.
                region_names = list(region_to_paths_dict.keys())

                # Check that validation region name is included in available regions.
                for eval_region in self.eval_region:
                    if eval_region not in region_names:
                        raise ValueError(
                            f'Eval region {eval_region} not found in avilable regions {region_names}'
                        )

                # Get region directories of eval regions.
                sub_region_dirs = {}
                for eval_region in self.eval_region:
                    sub_region_dirs[eval_region] = region_to_paths_dict[
                        eval_region]
                region_to_paths_dict = sub_region_dirs

                # # Get image dir and pair with region name.
                # image_paths = region_to_paths_dict[self.eval_region]
            elif self.split == 'all':
                pass
            else:
                raise ValueError(
                    f'Cannot handle split "{self.split}" for splitting data by region.'
                )

        # Get image dir and pair with region name.
        for region_img_paths in region_to_paths_dict.values():
            image_paths.extend(region_img_paths)

        if (self.eval_region is None) and (self.split != 'all'):
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

    def _prepare_S2_data(self, base_dir):
        # Get all region names.
        ## Get all image names which contain region names.
        image_paths = sorted(glob(os.path.join(base_dir, 'S2Hand') + '/*.tif'))
        region_names = list(
            set([os.path.split(p)[1].split('_')[0] for p in image_paths]))

        # Get all image images from these regions.
        region_to_paths_dict = {}
        for region_name in sorted(region_names):
            # Get all image paths corresponding to this region.
            region_img_paths = sorted(
                glob(
                    os.path.join(base_dir, 'S2Hand') + f'/{region_name}*.tif'))
            if len(region_img_paths) == 0:
                breakpoint()
                pass
            region_to_paths_dict[region_name] = region_img_paths

        # Split data.
        image_paths = self._split_data(region_to_paths_dict)
        if len(image_paths) == 0:
            raise ValueError(
                f'No image paths found for dataset "{self.dset_name}" and sensor "{self.sensor}".'
            )

        # Get all data corresponding and activated data paths
        n_images = 0
        self.dataset = []
        for image_path in image_paths:
            # Get region name.
            image_name = os.path.split(image_path)[1][:-4]
            region_name = image_name.split('_')[0]

            # Get corresponding label path for the image path.
            label_path = os.path.join(
                base_dir, 'LabelHand',
                image_name.replace('S2Hand', 'LabelHand') + '.tif')
            if os.path.exists(label_path) is False:
                breakpoint()
                pass
            label_info = rasterio.open(label_path)
            label_height, label_width = label_info.height, label_info.width

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

    def _prepare_S1_data(self, base_dir):
        # Get all region names.
        ## Get all image names which contain region names.
        image_paths = sorted(glob(os.path.join(base_dir, 'S1Hand') + '/*.tif'))
        region_names = list(
            set([os.path.split(p)[1].split('_')[0] for p in image_paths]))

        # Get all image images from these regions.
        region_to_paths_dict = {}
        for region_name in sorted(region_names):
            # Get all image paths corresponding to this region.
            region_img_paths = sorted(
                glob(
                    os.path.join(base_dir, 'S1Hand') + f'/{region_name}*.tif'))

            if len(region_img_paths) == 0:
                breakpoint()
                pass

            region_to_paths_dict[region_name] = region_img_paths

        # Split data.
        image_paths = self._split_data(region_to_paths_dict)

        # Get all data corresponding and activated data paths
        n_images = 0
        self.dataset = []
        for image_path in image_paths:
            # Get region name.
            image_name = os.path.split(image_path)[1][:-4]
            region_name = image_name.split('_')[0]

            # Get corresponding label path for the image path.
            label_path = os.path.join(
                base_dir, 'LabelHand',
                image_name.replace('S1Hand', 'LabelHand') + '.tif')
            if os.path.exists(label_path) is False:
                breakpoint()
                pass
            label_info = rasterio.open(label_path)
            label_height, label_width = label_info.height, label_info.width

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
        # Value mapping:
        # -1: No data (ignore) -> Ignore index
        # 0: No flood -> 0
        # 1: Water -> 1

        # Get positive water label.
        x, y = np.where(label == 1)
        binary_label[x, y] = 1

        # Get ignore label.
        x, y = np.where((label == -1))
        binary_label[x, y] = self.ignore_index

        return binary_label

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
        elif backend == 'tifffile':
            image = tifffile.imread(image_path)  # [channels, height, width]
        else:
            raise NotImplementedError(
                f'No method for loading image with backend "{backend}"')

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


if __name__ == '__main__':
    import argparse
    from copy import deepcopy

    from tqdm import tqdm

    from st_water_seg.tools import create_gif
    from utils import get_dset_path, generate_image_slice_object

    parser = argparse.ArgumentParser()
    parser.add_argument('--ex_indices', type=int, default=[0], nargs='+')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--sensor', type=str, default='S2')
    parser.add_argument('--eval_region',
                        type=str,
                        default=None,
                        help='Bolivia')
    parser.add_argument('--channels', type=str, default='ALL')
    args = parser.parse_args()

    dset_name = "s1floods"
    root_dir = get_dset_path(dset_name)
    slice_params = generate_image_slice_object(args.crop_size,
                                               args.crop_size,
                                               scale=1,
                                               stride=None)

    # Load dataset object.
    dataset = S1Floods(root_dir,
                       args.split,
                       slice_params,
                       sensor=args.sensor,
                       channels=args.channels,
                       eval_region=args.eval_region)
    # Get an example.
    for index in tqdm(args.ex_indices):
        example = dataset.__getitem__(index)

        # Create an RGB version of image.
        rgb_image = dataset.to_RGB(example["image"])
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
                   f"./{dset_name}_{dataset.sensor}_{index}.gif")
