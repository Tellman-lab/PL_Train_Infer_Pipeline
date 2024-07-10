import os

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from tifffile import tifffile
# import gdal


def resize_image(image,
                 desired_height,
                 desired_width,
                 resize_mode=cv2.INTER_LANCZOS4):#cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST, cv2.INTER_AREA
    """Resize the resolution of input image from original to that of labels.

    Args:
        image (np.array): A numpy array of shape [n_channels, height, width].
        desired_height (int): New height that the input image will be resized to.
        desired_width (int): New width that the input image will be resized to.
        resize_mode (int, optional): _description_. Defaults to cv2.INTER_LINEAR.

    Raises:
        NotImplementedError: Can only handle images with 2 or 3 dimensions.

    Returns:
        np.array: A numpy array of shape [n_channels, desired_height, desired_width]
    """
    # Get image dimensions.
    n_dims = len(image.shape)

    H, W = image.shape[-2], image.shape[-1]

    # Check if desired resolution matches the original resolution.
    if (desired_height == H) and (desired_width == W):
        return image

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


class ImageStitcher:

    def __init__(self,
                 save_dir,
                 image_type_name,
                 save_backend='PIL',
                 save_ext='.tif'):
        """Class to combine crops for a single type of image.

        Args:
            save_dir (str): TODO: _description_
        """
        self.save_dir = save_dir
        self.save_ext = save_ext
        self.save_backend = save_backend
        self.image_type_name = image_type_name

        self._images_combined = False

        # Create save directory if it does not already exist.
        os.makedirs(save_dir, exist_ok=True)

        # Initialize canvas
        self.image_canvas = {}
        self.weight_canvas = {}

    def add_images(self,
                   images,
                   image_names,
                   crop_info,
                   og_heights,
                   og_widths,
                   image_weights=None):
        # Populate image weights if it is None.
        if image_weights is None:
            image_weights = [None] * len(images)

        for img, name, crop, og_height, og_width, img_weight in zip(
                images, image_names, crop_info, og_heights, og_widths,
                image_weights):
            self.add_image(img,
                           name,
                           crop,
                           og_height,
                           og_width,
                           image_weight=img_weight)

    def add_image(self,
                  image,
                  image_name,
                  crop_info,
                  og_height,
                  og_width,
                  image_weight=None):
        # Get crop info.
        h0 = crop_info.h0
        w0 = crop_info.w0
        hE = crop_info.hE
        wE = crop_info.wE
        dh = hE - h0
        dw = wE - w0

        n_dims = len(image.shape)

        # Check if canvas image has already been generated.
        try:
            self.image_canvas[image_name]
            self.weight_canvas[image_name]
        except KeyError:
            # Create canvases for this image.

            ## Initialize canvas for this image.
            if n_dims == 2:
                self.image_canvas[image_name] = np.zeros([og_height, og_width],
                                                         dtype=type(image))
            elif n_dims == 3:
                self.image_canvas[image_name] = np.zeros(
                    [og_height, og_width, image.shape[-1]], dtype=image.dtype)
            else:
                raise NotImplementedError

            ## Initialize weight canvas for this image.
            self.weight_canvas[image_name] = np.zeros([og_height, og_width],
                                                      dtype='float')

        # Upadate canvases and weights.
        if n_dims == 2:
            self.image_canvas[image_name][h0:hE, w0:wE] += image[:dh, :dw]
        elif n_dims == 3:
            self.image_canvas[image_name][h0:hE,
                                          w0:wE, :] += image[:dh, :dw, :]
        self.weight_canvas[image_name][h0:hE, w0:wE] += np.ones([dh, dw],
                                                                dtype='float')

    def _combine_images(self):
        if self._images_combined:
            pass
        else:
            for image_name in tqdm(self.image_canvas.keys(),
                                   colour='green',
                                   desc='Combining images'):
                # Normalize canvases based on weights.
                if len(self.image_canvas[image_name].shape) == 2:
                    self.image_canvas[image_name] = self.image_canvas[
                        image_name] / (self.weight_canvas[image_name] + 1e-5)
                elif len(self.image_canvas[image_name].shape) == 3:
                    try:
                        self.image_canvas[
                            image_name] = self.image_canvas[image_name] / (
                                self.weight_canvas[image_name][:, :, None] +
                                1e-5)
                    except:
                        breakpoint()
                        pass
                else:
                    raise NotImplementedError(
                        f'Cannot save image in ImageStitcher with {len(self.image_canvas[image_name])} dimensions.'
                    )

                ## Make sure there are no NaN values kept during normalization stage.
                self.image_canvas[image_name] = np.nan_to_num(
                    self.image_canvas[image_name])
            self._images_combined = True

    def save_images(self):
        save_paths, image_names, image_sizes = [], [], []
        self._combine_images()
        for image_name in tqdm(self.image_canvas.keys(),
                               colour='green',
                               desc='Saving images'):
            # Save normalized image.
            img_save_dir = os.path.join(self.save_dir, image_name)
            os.makedirs(img_save_dir, exist_ok=True)
            save_path = os.path.join(img_save_dir,
                                     self.image_type_name + self.save_ext)

            ## Call save image method.
            self._save_image(self.image_canvas[image_name], save_path)

            ## Record image sizes, save path, and image name.
            image_sizes.append(self.image_canvas[image_name].shape)
            save_paths.append(save_path)
            image_names.append(image_name)

        return save_paths, image_names, image_sizes

    def _save_image(self, image, save_path):
        if self.save_backend == 'tifffile':
            tifffile.imwrite(save_path, image)
        elif self.save_backend == 'PIL':
            # Check the input type of image.
            if isinstance(image, int) is False:
                if image.max() < 1:
                    image = image * 255
                image = image.astype('uint8')

            Image.fromarray(image).save(save_path)
        else:
            raise NotImplementedError

    def get_combined_images(self):
        self._combine_images()
        return self.image_canvas


# class ImageStitcher:

#     def __init__(self, save_dir, save_backend='gdal', save_ext='.tif'):
#         """Class to combine crops for a single type of image.
#         Args:
#             save_dir (str): TODO: _description_
#         """
#         self.save_dir = save_dir
#         self.save_ext = save_ext
#         self.save_backend = save_backend

#         # Create save directory if it does not already exist.
#         os.makedirs(save_dir, exist_ok=True)

#         # Initialize canvas
#         self.image_canvas = {}
#         self.weight_canvas = {}

#     def add_images(self,
#                    images,
#                    image_names,
#                    crop_info,
#                    og_heights,
#                    og_widths,
#                    image_weights=None):
#         # Populate image weights if it is None.
#         if image_weights is None:
#             image_weights = [None] * len(images)

#         for img, name, crop, og_height, og_width, img_weight in zip(
#                 images, image_names, crop_info, og_heights, og_widths,
#                 image_weights):
#             self.add_image(img,
#                            name,
#                            crop,
#                            og_height,
#                            og_width,
#                            image_weight=img_weight)

#     def add_image(self,
#                   image,
#                   image_name,
#                   crop_info,
#                   og_height,
#                   og_width,
#                   image_weight=None):
#         # Get crop info.
#         h0, w0, dh, dw = crop_info
#         # h0, w0, dh, dw = h0.item(), w0.item(), dh.item(), dw.item()
#         hE, wE = h0 + dh, w0 + dw

#         # Check if canvas image has already been generated.
#         try:
#             self.image_canvas[image_name]
#             self.weight_canvas[image_name]
#         except KeyError:
#             # Create canvases for this image.

#             ## Initialize canvas for this image.
#             if len(image.shape) == 2:
#                 self.image_canvas[image_name] = np.zeros([og_height, og_width],
#                                                          dtype=type(image))
#             elif len(image.shape) == 3:
#                 self.image_canvas[image_name] = np.zeros(
#                     [image.shape[0], og_height, og_width], dtype=image.dtype)
#             else:
#                 raise NotImplementedError

#             ## Initialize weight canvas for this image.
#             self.weight_canvas[image_name] = np.zeros([og_height, og_width],
#                                                       dtype='float')

#         # Upadate canvases and weights.
#         self.image_canvas[image_name][:, h0:hE, w0:wE] += image[:, :dh, :dw]
#         self.weight_canvas[image_name][h0:hE, w0:wE] += np.ones([dh, dw],
#                                                                 dtype='float')

#     def save_images(self):
#         save_paths, image_names, image_sizes = [], [], []
#         for image_name in tqdm(self.image_canvas.keys(),
#                                colour='green',
#                                desc='Saving images'):
#             # Normalize canvases based on weights.
#             if len(self.image_canvas[image_name].shape) == 2:
#                 self.image_canvas[image_name] = self.image_canvas[
#                     image_name] / (self.weight_canvas[image_name] + 1e-5)
#             elif len(self.image_canvas[image_name].shape) == 3:
#                 self.image_canvas[image_name] = self.image_canvas[
#                     image_name] / (self.weight_canvas[image_name][None] + 1e-5)
#             else:
#                 raise NotImplementedError(
#                     f'Cannot save image in ImageStitcher with {len(self.image_canvas[image_name])} dimensions.'
#                 )

#             ## Make sure there are no NaN values kept during normalization stage.
#             self.image_canvas[image_name] = np.nan_to_num(
#                 self.image_canvas[image_name])

#             # Save normalized image.
#             save_path = os.path.join(self.save_dir, image_name + self.save_ext)

#             ## Call save image method.
#             self._save_image(self.image_canvas[image_name], save_path)

#             ## Record image sizes, save path, and image name.
#             image_sizes.append(self.image_canvas[image_name].shape)
#             save_paths.append(save_path)
#             image_names.append(image_name)

#         return save_paths, image_names, image_sizes

#     def _save_image(self, image, save_path):
#         if self.save_backend == 'gdal':
#             # kwimage.imwrite(save_path, image, backend='gdal')
#             driver = gdal.GetDriverByName("GTiff")
#             height, width = image.shape[-2], image.shape[-1]
#             if len(image.shape) == 2:
#                 outdata = driver.Create(save_path, width, height, 1,
#                                         gdal.GDT_Float32)
#                 outdata.GetRasterBand(1).WriteArray(image)
#                 outdata.FlushCache()
#                 del outdata
#             elif len(image.shape) == 3:
#                 n_channels = image.shape[0]
#                 outdata = driver.Create(save_path, width, height, n_channels,
#                                         gdal.GDT_Float32)
#                 for i in range(n_channels):
#                     outdata.GetRasterBand(i + 1).WriteArray(image[i])
#                     outdata.FlushCache()
#             else:
#                 raise NotImplementedError
#             del outdata
#             driver = None

#         elif self.save_backend == 'tifffile':
#             tifffile.imwrite(save_path, image)
#         elif self.save_backend == 'kwimage':
#             kwimage.imwrite(save_path, image)
#         else:
#             raise NotImplementedError


class ImageStitcher_v2:
# __init__ method is the constructor method for the class. It is called when an object of the class is created. 
# It is used to initialize the object's state. The self parameter is a reference to the current instance of the class, 
# and is used to access variables that belong to the class.
    def __init__(self,
                 save_dir,
                 image_type_name='',
                 save_backend='tifffile',
                 save_ext='.tif'):
        """Class to combine crops for a single type of image.
        Args:
            save_dir (str): TODO: _description_
        """
        self.save_dir = save_dir
        self.save_ext = save_ext
        self.save_backend = save_backend
        self.image_type_name = image_type_name

        self._images_combined = False

        # Create save directory if it does not already exist.
        os.makedirs(save_dir, exist_ok=True)

        # Initialize canvas
        self.image_canvas = {}
        self.weight_canvas = {}
# the following method is used to add multiple images to the image canvas, by calling the add_image method for each image
    def add_images(self,
                   images,
                   image_names,
                   crop_info,
                   og_heights,
                   og_widths,
                   image_weights=None):
        # Populate image weights if it is None.
        # image weight is very important for the image stitching process
        if image_weights is None:
            image_weights = [None] * len(images)

        for img, name, crop, og_height, og_width, img_weight in zip(
                images, image_names, crop_info, og_heights, og_widths,
                image_weights):
            self.add_image(img,
                           name,
                           crop,
                           og_height,
                           og_width,
                           image_weight=img_weight)

    def add_image(self,
                  image,
                  image_name,
                  crop_info,
                  og_height,
                  og_width,
                  image_weight=None):
        """
        Inputs:
            image (np.array): A np.array of shape [height, width] or [height, width, channels].
            image_name (str): TODO:
            crop_info (TODO:): TODO:
            og_height (int): TODO:
            og_width (int): TODO:
        """
        # Get crop info.
        h0 = crop_info.h0
        w0 = crop_info.w0
        hE = crop_info.hE
        wE = crop_info.wE
        dh = hE - h0
        dw = wE - w0

        n_dims = len(image.shape)

        # Check if canvas image has already been generated.
        try:
            self.image_canvas[image_name]
            self.weight_canvas[image_name]
        except KeyError:
            # Create canvases for this image.

            ## Initialize canvas for this image.
            if n_dims == 2:
                self.image_canvas[image_name] = np.zeros([og_height, og_width],
                                                         dtype=type(image))
            elif n_dims == 3:
                self.image_canvas[image_name] = np.zeros(
                    [og_height, og_width, image.shape[-1]], dtype=image.dtype)
            else:
                raise NotImplementedError

            ## Initialize weight canvas for this image.
            self.weight_canvas[image_name] = np.zeros([og_height, og_width],
                                                      dtype='float')

        # Upadate canvases and weights.
        if n_dims == 2:
            self.image_canvas[image_name][h0:hE, w0:wE] += image[:dh, :dw]
        elif n_dims == 3:
            self.image_canvas[image_name][h0:hE,
                                          w0:wE, :] += image[:dh, :dw, :]
        self.weight_canvas[image_name][h0:hE, w0:wE] += np.ones([dh, dw],
                                                                dtype='float')

    def _combine_images(self):
        if self._images_combined:
            pass
        else:
            for image_name in tqdm(self.image_canvas.keys(),
                                   colour='green',
                                   desc='Combining images'):
                # breakpoint()
                # pass
                # Normalize canvases based on weights.
                if len(self.image_canvas[image_name].shape) == 2:
                    self.image_canvas[image_name] = self.image_canvas[
                        image_name] / (self.weight_canvas[image_name] + 1e-5)
                elif len(self.image_canvas[image_name].shape) == 3:
                    self.image_canvas[
                        image_name] = self.image_canvas[image_name] / (
                            self.weight_canvas[image_name][:, :, None] + 1e-5)
                    
                else:
                    raise NotImplementedError(
                        f'Cannot save image in ImageStitcher with {len(self.image_canvas[image_name])} dimensions.'
                    )

                ## Make sure there are no NaN values kept during normalization stage.
                self.image_canvas[image_name] = np.nan_to_num(
                    self.image_canvas[image_name])
                
                # self.image_canvas[image_name][self.image_canvas[image_name]<0.5]=0
                # self.image_canvas[image_name][self.image_canvas[image_name]>=0.5]=1
            self._images_combined = True
        # breakpoint()
        # pass
    def save_images(self, save_class):
        save_paths, image_names, image_sizes = [], [], []
        self._combine_images()
        for image_name in tqdm(self.image_canvas.keys(),
                               colour='green',
                               desc='Saving images'):
            # Save normalized image.
            img_save_dir = os.path.join(self.save_dir, image_name)
            os.makedirs(img_save_dir, exist_ok=True)
            save_path = os.path.join(img_save_dir,
                                     self.image_type_name + self.save_ext)

            ## Call save image method.
            # breakpoint()
            self._save_image(self.image_canvas[image_name], save_path, save_class)

            ## Record image sizes, save path, and image name.
            image_sizes.append(self.image_canvas[image_name].shape)
            save_paths.append(save_path)
            image_names.append(image_name)

        return save_paths, image_names, image_sizes

    # def _save_image(self, image, save_path):
    def _save_image(self, image, save_path,save_class=False):
        if save_class:
            image[image>=0.5]=1
            image[image<0.5]=0

        if self.save_backend == 'tifffile':
            # breakpoint()
            # pass
            image = image.astype(np.float16) 
            # breakpoint()
            # pass
            tifffile.imwrite(save_path, image)
        elif self.save_backend == 'PIL':
            # Check the input type of image.
            if isinstance(image, int) is False:
                if image.max() < 1:
                    image = image * 255
                    # breakpoint()
                    # pass
                image = image.astype('uint8')

            Image.fromarray(image).save(save_path)
        elif self.save_backend == 'gdal':
            # kwimage.imwrite(save_path, image, backend='gdal')
            driver = gdal.GetDriverByName("GTiff")
            height, width = image.shape[-2], image.shape[-1]
            if len(image.shape) == 2:
                outdata = driver.Create(save_path, width, height, 1,
                                        gdal.GDT_Float32)
                outdata.GetRasterBand(1).WriteArray(image)
                outdata.FlushCache()
                del outdata
            elif len(image.shape) == 3:
                n_channels = image.shape[0]
                outdata = driver.Create(save_path, width, height, n_channels,
                                        gdal.GDT_Float32)
                for i in range(n_channels):
                    outdata.GetRasterBand(i + 1).WriteArray(image[i])
                    outdata.FlushCache()
            else:
                raise NotImplementedError
            del outdata
            driver = None

        else:
            raise NotImplementedError

    def get_combined_images(self):
        self._combine_images()
        return self.image_canvas

