import os
import json
import pickle
import collections

import numpy as np
from hydra.utils import get_original_cwd


def get_dset_path(dset_name):
    # Load file with dataset roots.
    try:
        base_dir = get_original_cwd()
        # print("get_original_cwd: ", base_dir)
    except ValueError:
        base_dir = os.getcwd()
        # print("get_cwd: ", base_dir)
    root_dirs_file_paths = os.path.join(base_dir, 'dataset_dirs.json')
    root_dirs = json.load(open(root_dirs_file_paths, "r"))
    dset_root_dir = root_dirs[dset_name]
    return dset_root_dir


class CropParams:

    def __init__(self, h0, w0, height, width, og_height, og_width,
                 max_crop_height, max_crop_width):
        """Constructor.

        Args:
            h0 (int): The first pixel along the vertical axis that this crop starts.
            w0 (int): The first pixel along the horizontal axis that this crop starts.
            height (int): The height of the crop.
            width (int): The width of the crop.
            og_height (int): Original height of image being cropped.
            og_width (int): Original width of image being cropped.
        """
        self.h0 = h0
        self.w0 = w0

        self.height = height
        self.width = width

        self.hE = h0 + height
        self.wE = w0 + width

        self.og_height = og_height
        self.og_width = og_width

        # TODO: Fix this hack.
        self.max_crop_height = max_crop_height
        self.max_crop_width = max_crop_width

    def __str__(self) -> str:
        return f"H0: {self.h0} | W0:{self.w0} \nHE: {self.hE} | WE: {self.wE}"


def generate_image_slice_object(height, width=None, stride=None, scale=1):
    """Create a more easily queried object for image dimension information.

    Args:
        height (int): Height of crop slice.
        width (int, optional): Width of crop slice. If None, then use equal to height. Defaults to None.
        stride (int, optional): Value to determine the amount to move a crop over an image vertically or
          horizontally. Defaults to None.
        scale (float, optional): Scale the height and width by this factor. Note: The scale is used to resize the
          height and width crop sizes. Defaults to 1.

    Returns:
        namedtuple: [description]
    """
    ImageSlice = collections.namedtuple("ImageSlice",
                                        ["height", "width", "scale", "stride"])
    ImageSlice.height = height

    if width is None:
        width = height

    if stride is None:
        stride = height

    ImageSlice.width = width
    ImageSlice.scale = scale
    ImageSlice.stride = stride

    return ImageSlice


def get_crop_slices(height,
                    width,
                    crop_height,
                    crop_width,
                    step=None,
                    mode="exact"):
    """Given an image size and desried crop, return all possible crop slices over space.

    Args:
        height (int): The height of the image to be cropped (y-axis).
        width (int): The width of the image to be cropped (x-axis).
        crop_height (int): The size of the crop height. Note: For certain modes,
            e.g. mode = 'under', crop height must be less than original image height.
        crop_width (int): The size of the crop width. Note: For certain modes,
            e.g. mode = 'under', crop width must be less than original image width.
        step (int): Amount of pixels to skip to begin new crop. Defaults to None which becomes equal to height.
        mode (str, optional): Method for how to handle edge cases. Defaults to 'exact'.
            - exact: Returns slices that do not go over original image size
            - over: Returns slices that have fixed crop size, covers full image
            - under: Returns slices that have fixed crop size, may not cover full image

    Raises:
        NotImplementedError: If invalid crop mode given.

    Returns:
        list: A list of crop slices. Each crop slice has the following form [h0, w0, h, w].
    """
    if step is not None:
        if type(step) is tuple:
            h_step, w_step = step[0], step[1]
        elif type(step) is int:
            h_step, w_step = step, step
        else:
            raise TypeError(f"Invalid step type: {type(step)}")

        if h_step <= 0:
            raise ValueError(f"Step of size {h_step} is too small.")
        if w_step <= 0:
            raise ValueError(f"Step of size {w_step} is too small.")

        if h_step > height:
            raise ValueError(
                f"Step of size {h_step} is too large for height {height}")
        if w_step > width:
            raise ValueError(
                f"Step of size {w_step} is too large for width {width}")
    else:
        # No step so use crop size for height.
        h_step, w_step = crop_height, crop_width

    crop_slices = []
    if mode == "over":
        num_h_crops = 0
        while True:
            if ((num_h_crops * h_step) + crop_height) > height:
                break
            num_h_crops += 1
        num_w_crops = 0
        while True:
            if ((num_w_crops * w_step) + crop_width) > width:
                break
            num_w_crops += 1
        num_h_crops += 1
        num_w_crops += 1

        for i in range(num_h_crops):
            for j in range(num_w_crops):
                crop_slices.append(
                    [i * h_step, j * w_step, crop_height, crop_width])
    elif mode == "under":
        num_h_crops = 0
        while True:
            if ((num_h_crops * h_step) + crop_height) > height:
                break
            num_h_crops += 1
        num_w_crops = 0
        while True:
            if ((num_w_crops * w_step) + crop_width) > width:
                break
            num_w_crops += 1

        for i in range(num_h_crops):
            for j in range(num_w_crops):
                crop_slices.append(
                    [i * h_step, j * w_step, crop_height, crop_width])
    #####################################################################
    elif mode == "exact":
        # Get number of crops fit in target image
        num_h_crops = 0
        while True:
            if ((num_h_crops * h_step) + crop_height) > height:
                break
            num_h_crops += 1
        num_w_crops = 0
        while True:
            if ((num_w_crops * w_step) + crop_width) > width:
                break
            num_w_crops += 1
# this generates the coordinates of each full crop, and add them to the crop_slices list
        for i in range(num_h_crops):
            for j in range(num_w_crops):
                crop_slices.append(
                    [i * h_step, j * w_step, crop_height, crop_width])

        # Get the remaining portion of the images
        rem_h = height - (num_h_crops * h_step)
        rem_w = width - (num_w_crops * w_step)

        # Get reminder crops along width axis
        if rem_w != 0:
            for i in range(num_h_crops):
                crop_slices.append(
                    [i * h_step, num_w_crops * w_step, crop_height, rem_w])

        # Get reminder crops along height axis
        if rem_h != 0:
            for j in range(num_w_crops):
                crop_slices.append(
                    [num_h_crops * h_step, j * w_step, rem_h, crop_height])

        # Get final crop corner
        if (rem_h != 0) and (rem_w != 0):
            crop_slices.append(
                [num_h_crops * h_step, num_w_crops * w_step, rem_h, rem_w])
#####################################################################
    elif mode == "exact_improved":
        # Get number of crops fit in target image
        num_h_crops = 0
        while True:
            if ((num_h_crops * h_step) + crop_height) > height:
                break
            num_h_crops += 1
        num_w_crops = 0
        while True:
            if ((num_w_crops * w_step) + crop_width) > width:
                break
            num_w_crops += 1
# this generates the coordinates of each crop, and add them to the crop_slices list
        for i in range(num_h_crops):
            for j in range(num_w_crops):
                crop_slices.append(
                    [i * h_step, j * w_step, crop_height, crop_width])

        # Get the remaining portion of the images
        rem_h = height - (num_h_crops * h_step)
        rem_w = width - (num_w_crops * w_step)

        # Get reminder crops along width axis
        if rem_w != 0:
            for i in range(num_h_crops):
                crop_slices.append(
                    [i * h_step, width - crop_width, crop_height, crop_width])

        # Get reminder crops along height axis, not done yet.
        if rem_h != 0:
            for j in range(num_w_crops):
                crop_slices.append(
                    [num_h_crops * h_step, j * w_step, crop_width, crop_height])

        # Get final crop corner
        if (rem_h != 0) and (rem_w != 0):
            crop_slices.append(
                [num_h_crops * h_step, num_w_crops * w_step, rem_h, rem_w])
###################################################################
    else:
        raise NotImplementedError(f"Invalid mode: {mode}")

    return crop_slices


def load_global_dataset_norm_params(dataset_name, norm_param_path=None):
    # ASSUMPTION: norm_params are saved in base directory of github repo.
    # Load norm_param file.
    utils_file_dir = os.path.dirname(os.path.realpath(__file__))
    norm_save_path = os.path.join('\\'.join(utils_file_dir.split('\\')[:-2]),
                                  'dataset_norm_params.p')
    all_norm_params = pickle.load(open(norm_save_path, 'rb'))

    try:
        norm_params = all_norm_params['combined']
    except KeyError:
        raise (
            f'Normalization parameters is not available for dataset name "{dataset_name}"'
        )

    return norm_params
