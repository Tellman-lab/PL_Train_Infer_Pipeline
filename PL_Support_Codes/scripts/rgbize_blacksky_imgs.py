import os
from glob import glob
from tqdm import tqdm
from PIL import Image
from tifffile import tifffile


def to_rgb(image_path, gamma=1.0):
    img = tifffile.imread(image_path)

    img = img / 2**12

    img = img**gamma

    return img


if __name__ == '__main__':
    img_dir = '/media/mule/Projects/NASA/THP/blksky/imgs/'
    img_paths = glob(img_dir + '/*.tif')

    base_save_dir = '/media/mule/Projects/NASA/THP/blksky/RGB/'

    for img_path in tqdm(img_paths):
        rgb_image = to_rgb(img_path)

        image_name = os.path.splitext(os.path.split(img_path)[1])[0]

        save_path = os.path.join(base_save_dir, image_name + '.png')

        Image.fromarray((rgb_image * 255).astype('uint8')).save(save_path)
