import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from PL_Support_Codes.datasets import build_dataset
from PL_Support_Codes.datasets.utils import generate_image_slice_object


def compute_feature_stats(dataset, feature_names):
    pbar = tqdm(range(0, dataset.__len__()))

    all_feature_stats = defaultdict(dict)
    for index in pbar:
        example = dataset.__getitem__(index)

        for feature_name in feature_names:
            # Resize / process feature
            feature = example[feature_name].flatten()

            feature_stats = all_feature_stats[feature_name]
            ex_min, ex_5, ex_95, ex_max = np.percentile(
                feature, [0, 5, 95, 100])

            if 'min' not in feature_stats.keys():
                feature_stats['min'] = 1e6
            if 'max' not in feature_stats.keys():
                feature_stats['max'] = -1e8
            if '5' not in feature_stats.keys():
                feature_stats['5'] = [ex_5]
            if '95' not in feature_stats.keys():
                feature_stats['95'] = [ex_95]

            all_feature_stats[feature_name]['min'] = min(
                ex_min, feature_stats['min'])
            all_feature_stats[feature_name]['max'] = max(
                ex_max, feature_stats['max'])
            all_feature_stats[feature_name]['5'].append(ex_5)
            all_feature_stats[feature_name]['95'].append(ex_95)

    for feature_name in all_feature_stats.keys():
        all_feature_stats[feature_name]['5'] = np.median(
            all_feature_stats[feature_name]['5'])
        all_feature_stats[feature_name]['95'] = np.median(
            all_feature_stats[feature_name]['95'])

    return all_feature_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('--feature_names', type=str, default=None)
    parser.add_argument('--crop_size', type=int, default=300)
    parser.add_argument('--sensor_name', type=str, default='S1')
    parser.add_argument('--eval_region', type=str, default='BEI')
    parser.add_argument('--include_ms_image',
                        default=False,
                        action='store_true')
    args = parser.parse_args()

    # Parse features to compute statistics for.
    if args.feature_names is not None:
        feature_names = args.feature_names.split('|')
    else:
        feature_names = []

    if 'dem' in feature_names:
        include_dem = True
    else:
        include_dem = False

    if 'slope' in feature_names:
        include_slope = True
    else:
        include_slope = False

    if 'preflood' in feature_names:
        include_preflood = True
    else:
        include_preflood = False

    if 'pre_post_difference' in feature_names:
        include_pre_post_difference = True
    else:
        include_pre_post_difference = False

    if 'hand' in feature_names:
        include_hand = True
    else:
        include_hand = False

    if 'chirps' in feature_names:
        include_chirps = True
    else:
        include_chirps = False

    # Build dataset
    ## Use any split parameters.
    slice_params = generate_image_slice_object(args.crop_size, args.crop_size,
                                               args.crop_size)
    dataset = build_dataset(args.dataset_name,
                            'train',
                            slice_params,
                            norm_mode=None,
                            eval_region=args.eval_region,
                            sensor=args.sensor_name,
                            dem=include_dem,
                            slope=include_slope,
                            preflood=include_preflood,
                            pre_post_difference=include_pre_post_difference,
                            hand=include_hand,
                            chirps=include_chirps)

    if args.include_ms_image:
        feature_names.append('image')

    if len(feature_names) == 0:
        print('FATAL: No features were requested to have the stats computed.')
        exit()

    feature_stats = compute_feature_stats(dataset, feature_names)

    # Print out stats.
    for feature_name, feat_stats in feature_stats.items():
        print(
            f"{feature_name} | Min: {feat_stats['min']} | 5%: {feat_stats['5']} | 95%: {feat_stats['95']} | Max: {feat_stats['max']}"
        )
