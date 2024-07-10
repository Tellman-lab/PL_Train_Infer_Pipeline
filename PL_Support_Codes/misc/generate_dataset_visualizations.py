import os
import argparse
from copy import deepcopy
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from PL_Support_Codes.tools import create_gif
from PL_Support_Codes.datasets import build_dataset
from PL_Support_Codes.datasets.utils import generate_image_slice_object


def generate_dataset_visualizations(dataset_name, slice_params, base_save_dir):
    # Load dataset.
    dataset = build_dataset(dataset_name,
                            'all',
                            slice_params,
                            output_metadata=True)

    # Create visualization loop.
    region_count = defaultdict(lambda: 0)
    for i in tqdm(range(dataset.__len__())):
        # Load an example.
        example = dataset.__getitem__(i)
        region_name = example["metadata"]["region_name"]
        region_count[region_name] += 1
        rgb_image = dataset.to_RGB(example["image"])
        mask = example["water_mask"]

        # Create mask overlay.
        rgb_overlay = deepcopy(rgb_image)
        x, y = np.where(mask == 1)
        rgb_overlay[:, x, y] = np.asarray([[0, 1, 1]])

        # Format images for creating a GIF.
        rgb_image = (rgb_image * 255).astype("uint8")
        rgb_overlay = (rgb_overlay * 255).astype("uint8")

        # Get save path.
        save_dir = os.path.join(base_save_dir, dataset_name, region_name)
        os.makedirs(save_dir, exist_ok=True)
        save_name = f"{str(region_count[region_name]).zfill(4)}.gif"
        save_path = os.path.join(save_dir, save_name)

        create_gif([rgb_image, rgb_overlay], save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="Name of dataset")
    parser.add_argument("crop_size",
                        type=int,
                        help="Height and width of the dataset.")
    parser.add_argument(
        "base_save_dir",
        type=str,
        help="Base directory for where visualizations will be saved ",
    )
    args = parser.parse_args()

    slice_params = generate_image_slice_object(args.crop_size)

    generate_dataset_visualizations(args.dataset_name, slice_params,
                                    args.base_save_dir)
