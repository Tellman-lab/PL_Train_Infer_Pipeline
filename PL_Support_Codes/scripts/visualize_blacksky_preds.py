import os
from glob import glob
from tqdm import tqdm
from PIL import Image
from tifffile import tifffile


def main():
    base_dir = '/media/mule/Projects/NASA/THP/blksky/predictions/3m/'

    image_paths = glob(base_dir + '/*.tif')

    for img_path in tqdm(image_paths):
        img = tifffile.imread(img_path)

        path, ext = os.path.splitext(img_path)
        save_path = path + '_rgb_pred.png'
        Image.fromarray((img * 255).astype('uint8')).save(save_path)


if __name__ == '__main__':
    main()
