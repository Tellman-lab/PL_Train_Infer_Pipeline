import os
import pickle
import argparse
from tqdm import tqdm

import numpy as np

from PL_Support_Codes.datasets import build_dataset
from PL_Support_Codes.datasets.utils import generate_image_slice_object


def compute_dataset_normalization_parameters(dataset, subsample_pct=None):
    pbar = tqdm(range(0, dataset.__len__()))
    pixel_values, dem_values, slope_values = None, None, None
    for index in pbar:
        example = dataset.__getitem__(index)
        image = example['image']

        # Find pixels that are not padded.
        # ASSUMPTION: Padding is 0 and actual image wont contain this.
        x, y = np.where(image.mean(axis=0) != 0)

        # Gather pixel values.
        pixels = image[:, x, y]

        # Subselect pixels.
        if subsample_pct is None:
            pass
        else:
            n_total_pixels = pixels.shape[1]
            n_sub_pixels = int(n_total_pixels * subsample_pct)
            indices = np.random.choice(np.arange(n_total_pixels),
                                       size=n_sub_pixels,
                                       replace=False)
            pixels = np.take(pixels, indices, axis=1)

        if pixel_values is None:
            # Initial pixel_values.
            pixel_values = pixels
        else:
            pixel_values = np.concatenate((pixel_values, pixels), axis=1)

        if dataset.dem:
            dem_pixels = example['dem'][:, x, y]
            n_total_pixels = dem_pixels.shape[1]
            n_sub_pixels = int(n_total_pixels * subsample_pct)
            indices = np.random.choice(np.arange(n_total_pixels),
                                       size=n_sub_pixels,
                                       replace=False)
            dem_pixels = np.take(dem_pixels, indices, axis=1)

            if dem_values is None:
                # Initial dem_pixels.
                dem_values = dem_pixels
            else:
                dem_values = np.concatenate((dem_values, dem_pixels), axis=1)

        if dataset.slope:
            slope_pixels = example['slope'][:, x, y]
            n_total_pixels = slope_pixels.shape[1]
            n_sub_pixels = int(n_total_pixels * subsample_pct)
            indices = np.random.choice(np.arange(n_total_pixels),
                                       size=n_sub_pixels,
                                       replace=False)
            slope_pixels = np.take(slope_pixels, indices, axis=1)

            if slope_values is None:
                # Initial slope_pixels.
                slope_values = slope_pixels
            else:
                slope_values = np.concatenate((slope_values, slope_pixels), axis=1)

    norm_params = {
        dataset.sensor: {
            'mean': np.asarray(list(pixel_values.mean(axis=1).astype(float))),
            'std': np.asarray(list(pixel_values.std(axis=1).astype(float)))
        }
    }

    if dataset.dem:
        norm_params['dem'] = {
            'mean': dem_values.mean(axis=1),
            'std': dem_values.std(axis=1)
        }
    if dataset.slope:
        norm_params['slope'] = {
            'mean': slope_values.mean(axis=1),
            'std': slope_values.std(axis=1)
        }

    return norm_params


def main():
    # Get command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_name',
        type=str,
        help='Name of dataset to compute normalization parameters.')
    parser.add_argument('sensor_name',
                        type=str,
                        help='Type of satellite sensor.')
    parser.add_argument(
        '--norm_save_path',
        type=str,
        default=None,
        help='Overwrite the save path of the normalization parameters.')
    parser.add_argument('--crop_size',
                        type=int,
                        default=512,
                        help='The height and width of the loaded crops.')
    parser.add_argument('--subsample_pct', type=float, default=0.4,
                        help='Percentage of usuable pixels per image to collect for computing norm parameters. This decreases the ' \
                             'storage and time required to compute dataset norm params.')
    parser.add_argument('--include_dem',
                        default=False,
                        action='store_true',
                        help='Include DEM into normalization parameter.')
    parser.add_argument('--include_slope',
                        default=False,
                        action='store_true',
                        help='Include Slope into normalization parameter.')
    args = parser.parse_args()

    # Build dataset.
    ## Use any split parameters.
    slice_params = generate_image_slice_object(args.crop_size, args.crop_size,
                                               args.crop_size)
    dataset = build_dataset(args.dataset_name,
                            'all',
                            slice_params,
                            norm_mode=None,
                            sensor=args.sensor_name,
                            dem=args.include_dem,
                            slope=args.include_slope,
                            eval_region = None,
                            channels = 'ALL', n_classes=3)

    # Check save path for normalization parameters.
    if args.norm_save_path is None:
        script_file_dir = os.path.dirname(os.path.realpath(__file__))
        norm_save_path = os.path.join(
            '/'.join(script_file_dir.split('/')[:-2]), 'dataset_norm_params.p')
    else:
        norm_save_path = args.norm_save_path

    if os.path.exists(norm_save_path) is False:
        # Initialize file that will hold normalization parameters.
        all_norm_params = {}
        pickle.dump(all_norm_params, open(norm_save_path, 'wb'))
    else:
        all_norm_params = pickle.load(open(norm_save_path, 'rb'))

    # Compute normalization parameters.
    norm_params = compute_dataset_normalization_parameters(
        dataset, args.subsample_pct)

    # Save norm parameters to pickle file.
    try:
        all_norm_params[args.dataset_name] = norm_params
    except KeyError:
        all_norm_params[args.dataset_name] = {}
        all_norm_params[args.dataset_name] = norm_params
    pickle.dump(all_norm_params, open(norm_save_path, 'wb'))


if __name__ == '__main__':
    main()
