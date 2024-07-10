import os

import kwcoco
import rasterio
import ndsampler
from tifffile import tifffile

# from st_water_seg.datasets.utils import get_crop_slices, CropParams


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
    else:
        raise NotImplementedError(f"Invalid mode: {mode}")

    return crop_slices


class THP_KWCOCO_Ddataset:

    def __init__(self, dataset_path, crop_params, channels=''):
        self.coco_dset = kwcoco.CocoDataset(dataset_path)
        self.sampler = ndsampler.CocoSampler(self.coco_dset)

        # Create space sampler.
        gids = self.coco_dset.images().gids
        self.examples = []
        for gid in gids:
            image_data = self.coco_dset.index.imgs[gid]

            crop_slices = get_crop_slices(image_data['height'],
                                          image_data['width'],
                                          crop_params['height'],
                                          crop_params['width'],
                                          step=crop_params['stride'])
            for crop_slice in crop_slices:
                example = {}
                example['gid'] = gid
                example['crop_slice'] = crop_slice
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def _crop(self, image, crop_params):
        h0, w0, h, w = crop_params
        return image[h0:h0 + h, w0:w0 + w]

    def __getitem__(self, index):
        example_info = self.examples[index]
        image_info = self.coco_dset.index.imgs[example_info['gid']]

        slice_info = {
            'gid': example_info['gid'],
            'cy': example_info['crop_slice'][0],
            'cx': example_info['crop_slice'][1],
            'height': example_info['crop_slice'][2],
            'width': example_info['crop_slice'][3]
        }
        sample = self.sampler.load_sample(slice_info)

        # Load image.
        image_path = image_info['file_name']
        image = tifffile.imread(image_path)  # [H, W, C]

        ## Crop
        crop = self._crop(image, example_info['crop_slice'])

        ## Normalize (already between 0-1)
        pass

        # Load labels.
        label_info = image_info['auxiliary'][0]

        abc = self.coco_dset.load_image(example_info['gid'])
        target = self._crop(tifffile.imread(label_info['file_name']),
                            example_info['crop_slice'])

        # self.coco_dset.index.img[example_info['gid']]
        # abc = self.coco_dset.delayed_load(example_info['gid'])
        # img = self.coco_dset.load_image(example_info['gid'])


if __name__ == '__main__':
    import numpy as np

    kwcoco_path = '/media/mule/Projects/NASA/THP/Data/thp_nasa.kwcoco.json'
    crop_params = {'height': 200, 'width': 200, 'stride': 100}
    dataset = THP_KWCOCO_Ddataset(kwcoco_path, crop_params)
    n_examples = dataset.__len__()
    index = np.random.randint(0, n_examples)

    example = dataset.__getitem__(index)
