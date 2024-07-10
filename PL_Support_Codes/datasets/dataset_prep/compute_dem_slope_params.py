import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
from tifffile import tifffile


def compute_stats(image_paths, image_type):

    pixels = []
    for image_path in tqdm(image_paths,
                           desc=f'Computing {image_type} stats',
                           colour='green'):
        image = tifffile.imread(image_path)
        flat_image = image.flatten()
        pixels.append(flat_image)
    pixels = np.concatenate(pixels, axis=0)

    print(image_type)
    print('-' * 25)
    print(f'Min: {pixels.min()}')
    print(f'Min  5%: {np.percentile(pixels, 5)}')
    print(f'Max: {pixels.max()}')
    print(f'Max 95%: {np.percentile(pixels, 95)}')
    print(f'Mean: {pixels.mean()}')
    print(f'STD: {pixels.std()}')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str)
    args = parser.parse_args()

    # Get DEM and SLOPE paths.
    root_dirs = {}
    if args.dataset_name == 'thp':
        root_dirs[
            'dem'] = '/media/mule/Projects/NASA/THP/Data/TrainingDatasets/DEMclipped/'
        root_dirs[
            'slope'] = '/media/mule/Projects/NASA/THP/Data/TrainingDatasets/SLOPEclipped/'
    else:
        raise NotImplementedError

    # Get all image paths.
    image_paths = {}
    image_paths['dem'] = glob(root_dirs['dem'] + '/*.tif')
    image_paths['slope'] = glob(root_dirs['slope'] + '/*.tif')

    assert len(
        image_paths['dem']) > 0, f'No DEM images found at {root_dirs["dem"]}'
    assert len(image_paths['slope']
               ) > 0, f'No SLOPE images found at {root_dirs["slope"]}'

    # Compute stats for DEM images.
    compute_stats(image_paths['dem'], 'DEM')
    compute_stats(image_paths['slope'], 'SLOPE')

    # # Load dataset.
    # slice_params = generate_image_slice_object(args.crop_size, args.crop_size, scale=1, stride=None)
    # dataset = build_dataset(args.dataset_name, 'all', sensor='PS', channels='RGB', dem=True, slope=True)

    # n_examples = dataset.__len__()
    # for index in tqdm(range(n_examples), desc='Computing DEM and SLOPE stats', colour='green'):
    #     example = dataset.__getitem__(index)

    #     # Get DEM and SLOPE.
    #     dem, slope = example['dem'], example['slope']

    #     # Get only pixels that are
    #     pass
