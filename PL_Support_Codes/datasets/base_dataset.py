import os
import pickle


import cv2
import torch
import numpy as np
from tifffile import tifffile
import pytorch_lightning as pl
from torchvision.transforms import functional as F

from torch.utils.data import Dataset

from PL_Support_Codes.datasets.utils import load_global_dataset_norm_params


class BaseDataset(Dataset):

    def __init__(self,
                 dset_name,
                 root_dir,
                 split,
                 slice_params,
                 eval_region=None,
                 transforms=None,
                 sensor='S2',
                 channels=None,
                 n_classes=3,
                 seed_num=0,
                 norm_mode=None,
                 ignore_index=-1,
                 train_split_pct=0.8):
        if channels is None:
            self.channels = "ALL"
        else:
            self.channels = channels

        self.split = split
        self.sensor = sensor
        self.root_dir = root_dir
        self.seed_num = seed_num
        self.n_classes = n_classes
        self.dset_name = dset_name
        self.norm_mode = norm_mode
        self.transforms = transforms
        self.eval_region = eval_region
        self.ignore_index = ignore_index
        self.slice_params = slice_params

        if train_split_pct < 0 or train_split_pct > 1:
            raise ValueError(
                f'Train split pct must be between 0 and 1. Invalid value: {train_split_pct}'
            )
        else:
            self.train_split_pct = train_split_pct

        # Set random seed.
        if self.seed_num is not None:
            self._set_random_seed(self.seed_num)

        if self.norm_mode == 'global':
            self.global_norm_params = load_global_dataset_norm_params(
                self.dset_name == 'combined')

    def _set_random_seed(self, seed_num):
        if type(seed_num) is not int:
            raise TypeError(
                f"Input seed value is not an int but type {seed_num}")

        pl.seed_everything(self.seed_num)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        raise NotImplementedError(
            f'Dataset "{self.dset_name}" does not have __getitem__ method implemented yet.'
        )

    def normalize(self, image, input_type):
        """Normalize input image to distibution based on norm_mode.

        Args:
            image (np.array): A float array of shape [n_channels, height, width].

        Raises:
            NotImplementedError: A value for norm_mode other than ['global', 'local', None].

        Returns:
            np.array: A possibly normalized float array of shape [n_channels, height, width].
        """

        # Get mean and std based on normalization mode.
        if self.norm_mode == 'global':
            # Transform normalization parameters
            mean = self.global_norm_params[input_type]['mean'][:, None, None]
            std = self.global_norm_params[input_type]['std'][:, None, None]
        elif self.norm_mode == 'local':
            flat_image = image.reshape(image.shape[0],
                                       image.shape[1] * image.shape[2])

            # Make private variable so we can unnormalize it if needed.
            mean = flat_image.mean(axis=1)[:, None, None]
            std = flat_image.std(axis=1)[:, None, None]
        elif self.norm_mode is None: #substrac 0, and divide 1
            mean = np.zeros([image.shape[0], 1, 1], dtype=image.dtype)
            std = np.ones([image.shape[0], 1, 1], dtype=image.dtype)
        else:
            raise NotImplementedError(
                f'Normalization mode "{self.norm_mode}" not implemented.')

        image -= mean
        image /= std
        norm_image = image

        return norm_image, mean, std

    def _load_or_generate_normalization_parameters(self, pct_sample_dset=0.5):

        def compute_mean_and_std(image_paths, pct_sample):
            samples = []
            # Gather pixel values from images.
            for image_path in image_paths:
                if pct_sample < np.random.random():
                    image = tifffile.imread(image_path)  # [C, H, W]

                    if self.sensor == 'S2':
                        image = image / 2**12
                        image = np.clip(image, 0, 1)
                    elif self.sensor == 'PS':
                        image = np.clip(image, 0, 1).astype(float)
                    else:
                        raise NotImplementedError(
                            f'Sensor type "{self.sensor}" not handled.')

                    n_dims = len(image.shape)
                    if n_dims == 3:
                        image.resize(
                            [image.shape[2], image.shape[0] * image.shape[1]])
                    elif n_dims == 2:
                        image.resize([image.shape[-2] * image.shape[-1]])
                    else:
                        raise NotImplementedError

                    samples.append(image)

            # Compute mean and std.
            if n_dims == 3:
                stacked_samples = np.concatenate(samples, axis=1)
                mean = np.mean(stacked_samples, axis=1)
                std = np.std(stacked_samples, axis=1)
            elif n_dims == 2:
                stacked_samples = np.concatenate(samples, axis=0)
                mean = np.mean(stacked_samples, axis=0)
                std = np.std(stacked_samples, axis=0)
            else:
                raise NotImplementedError

            return mean, std

        breakpoint()
        pass

        # Check if normalization parameters have already been generated.
        save_path = os.path.join(self.root_dir,
                                 self.dset_name + '_norm_params.p')
        if os.path.exists(save_path) is False:
            # No norm parameters for this dataset have been created.
            print(
                f'No normalization parameters have been generated for dataset: {self.dset_name}'
            )
            norm_params = {self.seed_num: {self.split: {}}}
            pickle.dump(norm_params, open(save_path, 'wb'))

            image_norms = False

            if self.dem:
                dem_norms = False
            else:
                dem_norms = True

            if self.slope:
                slope_norms = False
            else:
                slope_norms = True
        else:
            norm_params = pickle.load(open(save_path, 'rb'))

            # Check that this train/val split (based on seed number)
            try:
                norm_params[self.seed_num][self.split][self.sensor]
                image_norms = True
            except KeyError:
                image_norms = False
                print(
                    f'No normalization parameters found for random seed split: {self.seed_num}'
                )
                norm_params[self.seed_num] = {self.split: {}}

            # Check if auxiliary features have had their normalization parameters computed.
            if self.dem:
                try:
                    norm_params[self.seed_num][self.split]['dem']
                    dem_norms = True
                except KeyError:
                    dem_norms = False
                    print(
                        f'No DEM normalization parameters found for random seed split: {self.seed_num}'
                    )
            else:
                dem_norms = True

            if self.slope:
                try:
                    norm_params[self.seed_num][self.split]['slope']
                    slope_norms = True
                except KeyError:
                    slope_norms = False
                    print(
                        f'No Slope normalization parameters found for random seed split: {self.seed_num}'
                    )
            else:
                slope_norms = True

        if image_norms is False:
            print(
                f'Generating image normalization parameters (split: {self.split}) ...'
            )
            # Compute norm parameters for images in this split.
            norm_params[self.seed_num][self.split][self.sensor] = {}
            img_mean, img_std = compute_mean_and_std(self.image_paths,
                                                     pct_sample_dset)
            norm_params[self.seed_num][self.split][
                self.sensor]['mean'] = img_mean
            norm_params[self.seed_num][self.split][
                self.sensor]['std'] = img_std

        if dem_norms is False:
            print(
                f'Generating DEM normalization parameters (split: {self.split}) ...'
            )
            dem_mean, dem_std = compute_mean_and_std(self.dem_paths,
                                                     pct_sample_dset)
            norm_params[self.seed_num][self.split]['dem'] = {}
            norm_params[self.seed_num][self.split]['dem']['mean'] = dem_mean
            norm_params[self.seed_num][self.split]['dem']['std'] = dem_std

        if slope_norms is False:
            print(
                f'Generating Slope normalization parameters (split: {self.split}) ...'
            )
            slope_mean, slope_std = compute_mean_and_std(
                self.slope_paths, pct_sample_dset)
            norm_params[self.seed_num][self.split]['slope'] = {}
            norm_params[self.seed_num][
                self.split]['slope']['mean'] = slope_mean
            norm_params[self.seed_num][self.split]['slope']['std'] = slope_std

        # Save updated normalization parameters.
        if (image_norms is False) or (dem_norms is False) or (slope_norms is
                                                              False):
            pickle.dump(norm_params, open(save_path, 'wb'))

        return norm_params

    def _get_size_ratios(self, from_image_hw, target_image_hw):
        f_h, f_w = from_image_hw
        t_h, t_w = target_image_hw

        h_ratio = t_h / f_h
        w_ratio = t_w / f_w
        return h_ratio, w_ratio

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
        if (image_height == desired_height) and (image_width == desired_width):
            return image
        elif (image_height > desired_height) and (image_width > desired_width):
            print('Image is larger than desired resolution. Error!!')
            breakpoint()
            pass
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
    

    def _crop_image(self, image, crop_params):
        h0, w0 = crop_params.h0, crop_params.w0
        hE, wE = crop_params.hE, crop_params.wE
        # breakpoint()
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

    def _resize_image(self,
                      image,
                      desired_height,
                      desired_width,
                      resize_mode=cv2.INTER_LINEAR):
        # Get image dimensions.
        n_dims = len(image.shape)

        if n_dims == 2:
            image = cv2.resize(image,
                               dsize=(desired_width, desired_height),
                               interpolation=resize_mode)
        elif n_dims == 3:
            image = image.transpose(1, 2, 0)
            image = cv2.resize(image,
                               dsize=(desired_width, desired_height),
                               interpolation=resize_mode)
            image = image.transpose(2, 0, 1)
        else:
            raise NotImplementedError(
                f'Cannot resize image with "{n_dims}" dimensions.')

        return image

    def _to_RGB_S2(self, image, gamma=0.8):
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
            red_band = image[3]
            green_band = image[2]
            blue_band = image[1]
        else:
            raise NotImplementedError

        # Adjust gamma values.
        red_band = red_band**(gamma)
        green_band = green_band**(gamma)
        blue_band = blue_band**(gamma)

        # Combine RGB bands.
        rgb_image = np.stack([red_band, green_band, blue_band], axis=2)

        return rgb_image

    def _to_RGB_L8(self, image, gamma=0.8):
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
            red_band = image[3]
            green_band = image[2]
            blue_band = image[1]
        else:
            raise NotImplementedError

        # Adjust gamma values.
        red_band = red_band**(gamma)
        green_band = green_band**(gamma)
        blue_band = blue_band**(gamma)

        # Combine RGB bands.
        rgb_image = np.stack([red_band, green_band, blue_band], axis=2)

        return rgb_image

    def _to_RGB_S1(self, image, gamma=1.0):
        # Get RGB bands.
        if self.channels == 'ALL':
            red_band = image[0]
            green_band = image[1]
            blue_band = image[1]
        else:
            raise NotImplementedError

        # Adjust gamma values.
        red_band = red_band**(gamma)
        green_band = green_band**(gamma)
        blue_band = blue_band**(gamma)

        # Combine RGB bands.
        rgb_image = np.stack([red_band, green_band, blue_band], axis=2)

        return rgb_image

    def _to_RGB_PS(self, image, gamma=0.6):
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

    def to_RGB(self, image, gamma=None):

        if gamma is None:
            if self.sensor == 'S2':
                rgb_image = self._to_RGB_S2(image)
            elif self.sensor == 'S1':
                rgb_image = self._to_RGB_S1(image)
            elif self.sensor == 'PS':
                rgb_image = self._to_RGB_PS(image)
            elif self.sensor == 'L8':
                rgb_image = self._to_RGB_L8(image)
            else:
                raise NotImplementedError
        else:  
            if self.sensor == 'S2':
                rgb_image = self._to_RGB_S2(image,gamma=gamma)
            elif self.sensor == 'S1':
                rgb_image = self._to_RGB_S1(image,gamma=gamma)
            elif self.sensor == 'PS':
                rgb_image = self._to_RGB_PS(image,gamma=gamma)
            elif self.sensor == 'L8':
                rgb_image = self._to_RGB_L8(image,gamma=gamma)
            else:
                raise NotImplementedError

        return rgb_image

    def sample_transforms(self):
        active_transforms = []

        # Horizontal flip.
        if self.transforms.hflip.active:
            coin = np.random.rand()
            if coin < self.transforms.hflip.likelihood:
                active_transforms.append({'transform': F.hflip, 'anno': True, 'kwargs': {}})

        # Vertical flip.
        if self.transforms.vflip.active:
            coin = np.random.rand()
            if coin < self.transforms.vflip.likelihood:
                active_transforms.append({'transform': F.vflip, 'anno': True, 'kwargs': {}})

        # Rotate image.
        if self.transforms.rotate.active:
            coin = np.random.rand()
            if coin < self.transforms.rotate.likelihood:
                rot_angle = np.random.uniform(self.transforms.rotate.min_rot_angle, self.transforms.rotate.max_rot_angle, size=1)[0]
                active_transforms.append({'transform': F.rotate, 'anno': True, 'kwargs': {'angle': rot_angle}})
        
        # Random crop.
        # TODO: needs some additional arguments such as height width
        # if self.transforms.random_crop.active:
        #     coin = np.random.rand()
        #     if coin > self.transforms.random_crop.liklihood:
        #         active_transforms.append({'transform': F.crop, 'anno': True, 'kwargs': {'size': self.transforms.random_crop.size}})


        return active_transforms

    def apply_transforms(self, image, active_transforms, is_anno):
        if isinstance(image, np.ndarray):
            image = torch.tensor(image)

        for transform in active_transforms:
            if is_anno is False:
                # Apply all transformations to image.
                image = transform['transform'](image, **transform['kwargs'])
            else:
                # Check if transform is applied to annotations.
                if transform['anno'] == is_anno:
                    try:
                        image = transform['transform'](image, **transform['kwargs'])
                    except RuntimeError:
                        # Add a channel dimension to the tensor.
                        image = transform['transform'](image[None], **transform['kwargs'])[0]
                else:
                    pass

        return image