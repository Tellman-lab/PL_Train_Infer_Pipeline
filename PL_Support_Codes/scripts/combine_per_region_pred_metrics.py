import os
import json
import argparse

from glob import glob
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_dir', type=str)
    args = parser.parse_args()

    metric_paths = glob(args.pred_dir + '/**/metrics.json', recursive=True)

    all_metrics = {}
    for metric_path in tqdm(metric_paths):
        # Get region name.
        region_name = metric_path.split('/')[-3]

        # Load metrics.
        metrics = json.load(open(metric_path, 'r'))

        # Add to all metrics dict.
        all_metrics[region_name] = {}
        all_metrics[region_name]['f1_score'] = metrics['f1_score']
        all_metrics[region_name]['iou'] = metrics['iou']

    # Save metrics.
    save_path = os.path.join(args.pred_dir, 'all_metrics.json')
    json.dump(all_metrics, open(save_path, 'w'), indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
