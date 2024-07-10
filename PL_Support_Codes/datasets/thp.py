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


class THP_Dataset(BaseDataset):

    def __init__(self,
                 root_dir,
                 split,
                 slice_params,
                 eval_region=None,
                 transforms=None,
                 sensor='S2',
                 channels=None,
                 dset_name="thp",
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

        super(THP_Dataset, self).__init__(dset_name,
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

        # Check that certain input parameters are possible.
        if self.preflood and self.sensor != 'S1':
            raise NotImplementedError(
                'Only the S1 currently as preflood images.')

        if (self.preflood is False) and self.pre_post_difference:
            raise NotImplementedError(
                'Cannot generate pre_post_difference without preflood active.')

        self.n_classes = 3
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

        else:
            raise NotImplementedError(
                f'No method for getting number of channels for sensor "{self.sensor}"'
            )

        # Get number of channels for auxiliary data.
        if self.dem:
            n_channels['dem'] = 1

        if self.slope:
            n_channels['slope'] = 1

        if self.preflood:
            n_channels['preflood'] = n_channels['ms_image']

        if self.pre_post_difference:
            n_channels['pre_post_difference'] = n_channels['ms_image']

        if self.hand:
            n_channels['hand'] = 1

        if self.chirps:
            n_channels['hand'] = 6

        return n_channels

    def _split_data(self, region_dirs):
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
            else:
                raise ValueError(
                    f'Cannot handle split "{self.split}" for splitting data by region.'
                )

        # Get image dir and pair with region name.
        for region_name, region_dir in region_dirs.items():
            region_image_paths = glob(region_dir + f'/{self.sensor}/*.tif')
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

    def _prepare_data(self):
        if self.sensor == 'S2':
            raise FileExistsError(f'Bug: No S2 images in {self.root_dir}')
            # base_dir = os.path.join(,
            #                         "Data/TrainingDatasets/S2_Data/")
            self._prepare_S2_data(self.root_dir)
        elif self.sensor == 'S1':
            # base_dir = os.path.join(
            #     self.root_dir,
            #     "Data/TrainingDatasets/S1onlyDatasetTHPv1013/GroupedByEvent/")
            self._prepare_S1_data(self.root_dir)
        elif self.sensor == 'PS':
            # base_dir = os.path.join(
            #     self.root_dir,
            #     "Data/TrainingDatasets/PSonlyDatasetTHPv1013/GroupedByEvent/")
            self._prepare_PS_data(self.root_dir)
        else:
            raise NotImplementedError(
                f'Data preperation method not created for sensor "{self.sensor}"'
            )

    def _prepare_S1_data(self, base_dir):
        # Get all region directories.
        region_dirs = sorted(glob(base_dir + "/*/"))
        region_names = [p.split('/')[-2] for p in region_dirs]

        region_dirs_dict = {}
        for region_name, region_dir in zip(region_names, region_dirs):
            region_dirs_dict[region_name] = region_dir
        image_paths = self._split_data(region_dirs_dict)

        n_images = 0
        self.dataset = []
        self.dem_paths, self.slope_paths, self.preflood_paths = [], [], []
        for image_path, region_name in image_paths:
            image_name = os.path.splitext(os.path.split(image_path)[1])[0]

            # Get label path.
            label_path = os.path.join(base_dir, region_name, 'labels',
                                      image_name + '.tif')
            if os.path.exists(label_path) is False:
                breakpoint()
                pass

            # Get label width and height.
            label_info = rasterio.open(label_path)
            label_height, label_width = label_info.height, label_info.width

            if self.dem:
                dem_path = os.path.join(self.root_dir,
                                        'Data/TrainingDatasets/DEMclipped',
                                        image_name + '.tif')
                if os.path.exists(dem_path) is False:
                    # Skip this example because there is no associated DEM image.
                    continue

                # Get DEM image.
                self.dem_paths.append(dem_path)

            if self.slope:
                slope_path = os.path.join(
                    self.root_dir, 'Data/TrainingDatasets/SLOPEclipped',
                    image_name + '.tif')
                if os.path.exists(slope_path) is False:
                    # Skip this example because there is no associated SLOPE image.
                    continue

                # Get SLOPE image.
                self.slope_paths.append(slope_path)

            if self.preflood:
                preflood_path = os.path.join(
                    self.root_dir, 'Data/TrainingDatasets/S1PreFlood/',
                    image_name + '.tif')
                if os.path.exists(preflood_path) is False:
                    # Skip this example because there is no associated SLOPE image.
                    continue

                # Get Preflood image.
                self.preflood_paths.append(preflood_path)

            if self.hand:
                hand_dir = os.path.join(self.root_dir,
                                        'Data/TrainingDatasets/HANDclipped/',
                                        region_name)
                hand_paths = glob(hand_dir + '/*.tif')

                hand_path = None
                for p in hand_paths:
                    if image_name in p:
                        hand_path = p

                if hand_path is None:
                    continue

            if self.chirps:
                chirps_dir = os.path.join(
                    self.root_dir, 'Data/TrainingDatasets/CHIRPSclipped/',
                    region_name)
                chirps_paths = glob(chirps_dir + '/*.tif')

                chirps_path = None
                for p in chirps_paths:
                    if image_name in p:
                        chirps_path = p

                if chirps_path is None:
                    continue

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

                if self.dem:
                    example['dem_path'] = dem_path

                if self.slope:
                    example['slope_path'] = slope_path

                if self.preflood:
                    example['preflood_path'] = preflood_path

                if self.hand:
                    example['hand_path'] = hand_path

                if self.chirps:
                    example['chirps_path'] = chirps_path

                self.dataset.append(example)
            n_images += 1
        print(f'Number of images in {self.split} dataset: {n_images}')

        self.image_paths = image_paths

    def _prepare_S2_data(self, base_dir):
        # Get all region directories.
        region_dirs = sorted(glob(base_dir + "/*/"))
        region_names = [p.split('/')[-2] for p in region_dirs]

        region_dirs_dict = {}
        for region_name, region_dir in zip(region_names, region_dirs):
            region_dirs_dict[region_name] = region_dir
        image_paths = self._split_data(region_dirs_dict)

        n_images = 0
        self.dataset = []
        self.dem_paths, self.slope_paths = [], []
        for image_path, region_name in image_paths:
            image_name = os.path.splitext(os.path.split(image_path)[1])[0]
            try:
                image_id = image_name.split('_')[-3]
            except:
                image_id = image_name.split('-')[0]

            # Get label path.
            label_path = os.path.join(base_dir, region_name, 'labels',
                                      image_name.split('-')[0] + '.tif')
            if os.path.exists(label_path) is False:
                breakpoint()
                pass

            # Get label width and height.
            label_info = rasterio.open(label_path)
            label_height, label_width = label_info.height, label_info.width

            if self.dem:
                dem_path = os.path.join(self.root_dir,
                                        'Data/TrainingDatasets/DEMclipped',
                                        image_name + '.tif')
                if os.path.exists(dem_path) is False:
                    # Skip this example because there is no associated DEM image.
                    continue

                # Get DEM image.
                self.dem_paths.append(dem_path)

            if self.slope:
                slope_path = os.path.join(
                    self.root_dir, 'Data/TrainingDatasets/SLOPEclipped',
                    image_name + '.tif')
                if os.path.exists(slope_path) is False:
                    # Skip this example because there is no associated SLOPE image.
                    continue

                # Get SLOPE image.
                self.slope_paths.append(slope_path)

            if self.hand:
                hand_dir = os.path.join(self.root_dir,
                                        'Data/TrainingDatasets/HANDclipped/',
                                        region_name)
                hand_paths = glob(hand_dir + '/*.tif')

                hand_path = None
                for p in hand_paths:
                    if image_id in p:
                        hand_path = p

                if hand_path is None:
                    continue

            if self.chirps:
                chirps_dir = os.path.join(
                    self.root_dir, 'Data/TrainingDatasets/CHIRPSclipped/',
                    region_name)
                chirps_paths = glob(chirps_dir + '/*.tif')

                chirps_path = None
                for p in chirps_paths:
                    if image_id in p:
                        chirps_path = p

                if chirps_path is None:
                    continue

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

                if self.dem:
                    example['dem_path'] = dem_path

                if self.slope:
                    example['slope_path'] = slope_path

                if self.hand:
                    example['hand_path'] = hand_path

                if self.chirps:
                    example['chirps_path'] = chirps_path

                self.dataset.append(example)
            n_images += 1
        print(f'Number of images in {self.split} dataset: {n_images}')

        self.image_paths = image_paths

    def _prepare_PS_data(self, base_dir):
        # Get all region directories.
        region_dirs = sorted(glob(base_dir + "/*/"))
        region_names = [p.split('/')[-2] for p in region_dirs]

        region_dirs_dict = {}
        for region_name, region_dir in zip(region_names, region_dirs):
            region_dirs_dict[region_name] = region_dir
        image_paths = self._split_data(region_dirs_dict)

        n_images = 0
        self.dataset = []
        self.dem_paths, self.slope_paths = [], []
        for image_path, region_name in image_paths:
            image_name = os.path.splitext(os.path.split(image_path)[1])[0]

            # Get label path.
            label_path = os.path.join(base_dir, region_name, 'labels',
                                      image_name + '.tif')
            if os.path.exists(label_path) is False:
                breakpoint()
                pass

            # Get label width and height.
            label_info = rasterio.open(label_path)
            label_height, label_width = label_info.height, label_info.width

            if self.dem:
                dem_path = os.path.join(self.root_dir,
                                        'Data/TrainingDatasets/DEMclipped',
                                        image_name + '.tif')
                if os.path.exists(dem_path) is False:
                    # Skip this example because there is no associated DEM image.
                    continue

                # Get DEM image.
                self.dem_paths.append(dem_path)

            if self.slope:
                slope_path = os.path.join(
                    self.root_dir, 'Data/TrainingDatasets/SLOPEclipped',
                    image_name + '.tif')
                if os.path.exists(slope_path) is False:
                    # Skip this example because there is no associated SLOPE image.
                    continue

                # Get SLOPE image.
                self.slope_paths.append(slope_path)

            if self.hand:
                hand_dir = os.path.join(self.root_dir,
                                        'Data/TrainingDatasets/HANDclipped/',
                                        region_name)
                hand_paths = glob(hand_dir + '/*.tif')

                hand_path = None
                for p in hand_paths:
                    if image_name in p:
                        hand_path = p

                if hand_path is None:
                    continue

            if self.chirps:
                chirps_dir = os.path.join(
                    self.root_dir, 'Data/TrainingDatasets/CHIRPSclipped/',
                    region_name)
                chirps_paths = glob(chirps_dir + '/*.tif')

                chirps_path = None
                for p in chirps_paths:
                    if image_name in p:
                        chirps_path = p

                if chirps_path is None:
                    continue

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

                if self.dem:
                    example['dem_path'] = dem_path

                if self.slope:
                    example['slope_path'] = slope_path

                if self.hand:
                    example['hand_path'] = hand_path

                if self.chirps:
                    example['chirps_path'] = chirps_path

                self.dataset.append(example)
            n_images += 1
        print(f'Number of images in {self.split} dataset: {n_images}')

        self.image_paths = image_paths

    def _load_S2_label_image(self,
                             label_path,
                             desired_height,
                             desired_width,
                             crop_params,
                             backend='tifffile'):
        # Load label.
        if backend == 'rasterio':
            label = rasterio.open(
                label_path).read()  # [channels+1, height, width]
            label = label.transpose(2, 0, 1)[-1]  # [1, height, width]
        elif backend == 'tifffile':
            label = tifffile.imread(label_path)  # [channels+1, height, width]
            label = label.transpose(2, 0, 1)[-1]  # [1, height, width]
        else:
            raise NotImplementedError(
                f'No method for loading label with backend "{backend}"')

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
        x, y = np.where((label == 44) | (label == 253) | (label == 254))
        binary_label[x, y] = 1

        return binary_label

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
        # 0: No data (ignore)
        # 1: No flood
        # 2: Low confidence flood
        # 3: High confidence flood

        # Get positive water label.
        x, y = np.where((label == 2) | (label == 3))
        binary_label[x, y] = 1

        # Get ignore label.
        x, y = np.where((label == 0))
        binary_label[x, y] = self.ignore_index

        return binary_label

    def _load_crop_norm_dem_image(self,
                                  dem_path,
                                  desired_height,
                                  desired_width,
                                  crop_params,
                                  backend='tifffile'):
        if backend == 'rasterio':
            dem = rasterio.open(dem_path).read()  # [height, width]
        elif backend == 'tifffile':
            dem = tifffile.imread(dem_path)  # [height, width]
        else:
            raise NotImplementedError(
                f'No method for loading DEM with backend "{backend}"')

        # Resize image if not desired resolution.
        height, width = dem.shape
        if (height != desired_height) or (width != desired_width):
            dem = resize_image(dem, desired_height, desired_width)

        # Crop DEM image.
        dem = self._crop_image(dem, crop_params)

        # Add buffer to image if necessary.
        dem = self._add_buffer_to_image(dem, crop_params.max_crop_height,
                                        crop_params.max_crop_width)

        # Normalize DEM.
        dem = np.clip(dem, 0, 400) / 400

        # Add channel dimension.
        dem = dem[None]

        return dem

    def _load_crop_norm_slope_image(self,
                                    slope_path,
                                    desired_height,
                                    desired_width,
                                    crop_params,
                                    backend='tifffile'):
        if backend == 'rasterio':
            slope = rasterio.open(slope_path).read()  # [height, width]
        elif backend == 'tifffile':
            slope = tifffile.imread(slope_path)  # [height, width]
        else:
            raise NotImplementedError(
                f'No method for loading SLOPE with backend "{backend}"')

        # Resize image if not desired resolution.
        height, width = slope.shape
        if (height != desired_height) or (width != desired_width):
            slope = resize_image(slope, desired_height, desired_width)

        # Crop SLOPE image.
        slope = self._crop_image(slope, crop_params)

        # Add buffer to image if necessary.
        slope = self._add_buffer_to_image(slope, crop_params.max_crop_height,
                                          crop_params.max_crop_width)

        # Normalize SLOPE.
        slope = np.clip(slope, 0, 90) / 90

        # Add channel dimension.
        slope = slope[None]

        return slope

    def _load_crop_norm_hand_image(self,
                                   hand_path,
                                   desired_height,
                                   desired_width,
                                   crop_params,
                                   backend='tifffile'):
        if backend == 'rasterio':
            hand = rasterio.open(hand_path).read()  # [height, width]
        elif backend == 'tifffile':
            hand = tifffile.imread(hand_path)  # [height, width]
        else:
            raise NotImplementedError(
                f'No method for loading HAND with backend "{backend}"')

        # Resize image if not desired resolution.
        height, width = hand.shape
        if (height != desired_height) or (width != desired_width):
            hand = resize_image(hand, desired_height, desired_width)

        # Crop HAND image.
        hand = self._crop_image(hand, crop_params)

        # Add buffer to image if necessary.
        hand = self._add_buffer_to_image(hand, crop_params.max_crop_height,
                                         crop_params.max_crop_width)

        # Check for NaNs.
        hand = np.nan_to_num(hand)

        # Normalize HAND (min/max: 0, 160.25)
        hand = np.clip(hand, 0, 160.25) / 160.25

        # Add channel dimension.
        hand = hand[None]

        return hand

    def _load_chirps_data(self, chirps_path, backend='tifffile'):
        if backend == 'rasterio':
            chirps = rasterio.open(chirps_path).read()  # [height, width]
        elif backend == 'tifffile':
            chirps = tifffile.imread(chirps_path)  # [height, width]
        else:
            raise NotImplementedError(
                f'No method for loading CHIRPS with backend "{backend}"')

        chirps = np.squeeze(chirps)

        # Normalize CHIRPS data (min/max: 0, 197.66)
        chirps = np.clip(chirps, 0, 200) / 200

        # Add channel dimension.
        chirps = chirps[None]

        return chirps

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

        # Load and add DEM to output.
        if self.dem:
            output['dem'] = self._load_crop_norm_dem_image(
                example['dem_path'], crop_params.og_height,
                crop_params.og_width, crop_params)

        # Load and add SLOPE to output.
        if self.slope:
            output['slope'] = self._load_crop_norm_slope_image(
                example['slope_path'], crop_params.og_height,
                crop_params.og_width, crop_params)

        if self.preflood:
            preflood = self._load_crop_norm_image(
                example['preflood_path'],
                crop_params,
                channels=self.channels,
                resize_dims=(crop_params.og_height,
                             crop_params.og_width)).astype('float32')
            preflood, _, _ = self.normalize(preflood, self.sensor)
            output['preflood'] = self._add_buffer_to_image(
                preflood, crop_params.max_crop_height,
                crop_params.max_crop_width)

        if self.pre_post_difference:
            output['pre_post_difference'] = (
                output['image'] - output['preflood']).astype('float32')

        if self.hand:
            output['hand'] = self._load_crop_norm_hand_image(
                example['hand_path'], crop_params.og_height,
                crop_params.og_width, crop_params)

        if self.chirps:
            output['chirps'] = self._load_chirps_data(example['chirps_path'])

        if output_metadata:
            output["metadata"] = {
                "image_path": example["image_path"],
                "crop_params": example["crop_params"]
            }
            if 'region_name' in example.keys():
                output['metadata']["region_name"] = example["region_name"]

        return output

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
                image_path).read()  # [height, width, channels]
            image = image.transpose(2, 0, 1)  # [channels, height, width]
        elif backend == 'tifffile':
            image = tifffile.imread(image_path)  # [height, width, channels]
            image = image.transpose(2, 0, 1)  # [channels, height, width]
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
            image = image.transpose(2, 0, 1)  # [channels, height, width]
        elif backend == 'tifffile':
            image = tifffile.imread(image_path)  # [channels, height, width]
            image = image.transpose(2, 0, 1)  # [channels, height, width]
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

        # Already normalized to [0,1] or ...
        # TODO: Hacky way to check if image is uint16, isinstance doesn't work here
        if image.dtype == np.uint16:
            image = image / 10000

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
        elif self.sensor == 'PS':
            image = self._load_crop_norm_PS_image(image_path, crop_params,
                                                  channels, resize_dims,
                                                  backend)

        return image


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    from st_water_seg.tools import create_gif
    from utils import get_dset_path, generate_image_slice_object

    parser = argparse.ArgumentParser()
    parser.add_argument('--ex_index', type=int, default=0)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--crop_size', type=int, default=1024)
    parser.add_argument('--sensor', type=str, default='PS')
    parser.add_argument('--eval_region', type=str, default=None, help='CMO')
    parser.add_argument('--dem', default=False, action='store_true')
    parser.add_argument('--slope', default=False, action='store_true')
    parser.add_argument('--channels', type=str, default='ALL')
    parser.add_argument('--preflood', default=False, action='store_true')
    parser.add_argument('--pre_post_difference',
                        default=False,
                        action='store_true')
    parser.add_argument('--hand', default=False, action='store_true')
    parser.add_argument('--chirps', default=False, action='store_true')
    args = parser.parse_args()

    dset_name = "thp"
    root_dir = get_dset_path(dset_name)
    slice_params = generate_image_slice_object(args.crop_size,
                                               args.crop_size,
                                               scale=1,
                                               stride=None)

    # Load dataset object.
    # dataset = THP_Dataset(root_dir, 'train', slice_params, sensor='S2')
    dataset = THP_Dataset(root_dir,
                          args.split,
                          slice_params,
                          sensor=args.sensor,
                          channels=args.channels,
                          eval_region=args.eval_region,
                          dem=args.dem,
                          slope=args.slope,
                          preflood=args.preflood,
                          pre_post_difference=args.pre_post_difference,
                          hand=args.hand,
                          chirps=args.chirps)
    # Get an example.
    index = args.ex_index
    example = dataset.__getitem__(index)

    # from time import time
    # getitem_durs = []
    # slow_samples = []
    # for index in tqdm(range(0, dataset.__len__())):
    #     start_time = time()
    #     example = dataset.__getitem__(index)

    #     time_dir = time()-start_time
    #     getitem_durs.append(time_dir)
    #     if time_dir > 2.0:
    #         print(index)
    #         slow_samples.append(index)

    # print(f'[{min(getitem_durs)}, {np.mean(getitem_durs)}, {max(getitem_durs)}]')
    # print(f'Slow indices: {slow_samples}')
    # exit()

    # # Slow indices
    # slow_samples = [115, 143, 186, 212, 214, 218, 221, 224, 231, 234, 247]
    # for index in slow_samples:
    #     example = dataset.__getitem__(index-1)
    # exit()

    # Create an RGB version of image.
    rgb_image = dataset.to_RGB(example["image"])
    mask = example["target"]

    # Visualize RGB image w/ and w/o overlay (gif).
    rgb_overlay = deepcopy(rgb_image)
    x, y = np.where(mask == 1)
    rgb_overlay[x, y, :] = np.asarray([0, 1, 1])

    rgb_image = (rgb_image * 255).astype("uint8")
    rgb_overlay = (rgb_overlay * 255).astype("uint8")

    create_gif([rgb_image, rgb_overlay],
               f"./{dset_name}_{dataset.sensor}_{index}.gif")
