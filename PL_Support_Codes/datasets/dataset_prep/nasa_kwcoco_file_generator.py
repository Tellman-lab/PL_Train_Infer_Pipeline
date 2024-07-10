import os
from glob import glob
from collections import defaultdict

import kwcoco
import kwimage
import rasterio
import numpy as np
from tqdm import tqdm


def get_dataset_paths():
    # Find all MS image paths.
    dataset = {}
    ps_image_dir = '/media/mule/Projects/NASA/THP/Data/TrainingDatasets/PSdatasetTHPv0822/images'
    image_paths = sorted(glob(ps_image_dir + '/*.tif'))

    ## Manually remove a couple of images from image_paths.
    image_paths.remove(os.path.join(ps_image_dir, 'nhselectedPS.tif'))
    image_paths.remove(os.path.join(ps_image_dir, 'selectedPSu2.tif'))

    # Add image paths to dataset
    image_names = [
        os.path.splitext(os.path.split(p)[1])[0] for p in image_paths
    ]
    for img_name, img_p in zip(image_names, image_paths):
        dataset[img_name] = {'image_path': img_p}

    # Get paths to various label and auxiliary data.
    ## Label data
    ps_label_dir = '/media/mule/Projects/NASA/THP/Data/TrainingDatasets/PSdatasetTHPv0822/labels'
    label_paths = sorted(glob(ps_label_dir + '/*.tif'))

    ## DEM data
    dem_dir = '/media/mule/Projects/NASA/THP/Data/TrainingDatasets/DEMclipped'
    dem_paths = sorted(glob(dem_dir + '/*.tif'))

    ## Flood data
    flood_dir = '/media/mule/Projects/NASA/THP/Data/TrainingDatasets/Flood'
    flood_paths = sorted(glob(flood_dir + '/*.tif'))

    ## HAND data
    hand_dir = '/media/mule/Projects/NASA/THP/Data/TrainingDatasets/HANDclipped'
    hand_subdirs = glob(hand_dir + '/*/')
    hand_paths, hand_names = [], []
    for hand_subdir in hand_subdirs:
        hand_subdir_paths = glob(hand_subdir + '/*.tif')
        hand_paths.extend(hand_subdir_paths)

        for p in hand_subdir_paths:
            hand_name = os.path.splitext(os.path.split(p)[1])[0][-4:]
            hand_names.append(hand_name)

    hand_path_info = {}
    for h_name, h_path in zip(hand_names, hand_paths):
        hand_path_info[h_name] = h_path

    ## CHIRPS data
    chirp_dir = '/media/mule/Projects/NASA/THP/Data/TrainingDatasets/CHIRPSclipped/SPS'
    chirp_subdirs = glob(chirp_dir + '/*/')
    chirp_paths, chirp_names = [], []
    for chirp_subdir in chirp_subdirs:
        chirp_subdir_paths = glob(chirp_subdir + '/*.tif')
        chirp_paths.extend(chirp_subdir_paths)

        for p in chirp_subdir_paths:
            chirp_name = os.path.splitext(os.path.split(p)[1])[0][-4:]
            chirp_names.append(chirp_name)

    chirp_path_info = {}
    for h_name, h_path in zip(chirp_names, chirp_paths):
        chirp_path_info[h_name] = h_path

    ## SLOPE data
    slope_dir = '/media/mule/Projects/NASA/THP/Data/TrainingDatasets/SLOPEclipped'
    slope_paths = sorted(glob(slope_dir + '/*.tif'))

    ## PreFlood data
    preflood_dir = '/media/mule/Projects/NASA/THP/Data/TrainingDatasets/PreFlood'
    preflood_paths = sorted(glob(preflood_dir + '/*.tif'))

    # Find corresponding data paths for each image.
    missing_data = defaultdict(list)
    for image_name in image_names:

        # Get, check, and add label image.
        gen_label_path = os.path.join(ps_label_dir, image_name + '.tif')

        if gen_label_path not in label_paths:
            missing_data['labels'].append(image_name)
        else:
            dataset[image_name]['label_path'] = gen_label_path

        # Get, check, and add DEM image.
        gen_dem_path = os.path.join(dem_dir, image_name + '.tif')

        if gen_dem_path not in dem_paths:
            missing_data['dem'].append(image_name)
        else:
            dataset[image_name]['dem_path'] = gen_dem_path

        # Get, check, and add Flood image.
        # gen_flood_path = os.path.join(flood_dir, image_name+'.tif')

        # if gen_flood_path not in flood_paths:
        #     missing_data['flood'].append(image_name)
        # else:
        #      dataset[image_name]['flood_path'] = gen_flood_path

        # Get, check, and add HAND image.
        # if image_name not in hand_names:
        #     missing_data['hand'].append(image_name)
        # else:
        #      dataset[image_name]['hand_path'] = hand_path_info[image_name]

        # Get, check, and add CHIRPS image.
        # if image_name not in chirp_names:
        #     missing_data['chirps'].append(image_name)
        # else:
        #      dataset[image_name]['chirps_path'] = chirp_path_info[image_name]

        # Get, check, and add SLOPE image.
        gen_slope_path = os.path.join(slope_dir, image_name + '.tif')

        if gen_slope_path not in slope_paths:
            missing_data['slope'].append(image_name)
        else:
            dataset[image_name]['slope_path'] = gen_slope_path

        # Get, check, and add PreFlood image.
        # gen_preflood_path = os.path.join(preflood_dir, image_name+'.tif')

        # if gen_preflood_path not in preflood_paths:
        #     missing_data['preflood'].append(image_name)
        # else:
        #      dataset[image_name]['preflood_path'] = gen_preflood_path
        if len(list(dataset[image_name].keys())) != 4:
            print(image_name)

    print(f'Total number of found images: {len(image_names)}')
    print(
        f'Number of images without labels: {len(missing_data["labels"])} | {missing_data["labels"]}'
    )
    print(
        f'Number of images without DEMs: {len(missing_data["dem"])} | {missing_data["dem"]}'
    )
    print(
        f'Number of images without SLOPE data: {len(missing_data["slope"])} | {missing_data["slope"]}'
    )
    # print(f'Number of images without Flood data: {len(missing_data["flood"])} | {missing_data["flood"]}')
    # print(f'Number of images without HAND data: {len(missing_data["hand"])} | {missing_data["hand"]}')
    # print(f'Number of images without CHIRPS data: {len(missing_data["chirps"])} | {missing_data["chirps"]}')
    # print(f'Number of images without PreFlood data: {len(missing_data["preflood"])} | {missing_data["preflood"]}')

    return dataset


def binarize_water_mask(water_mask):
    canvas = np.zeros(water_mask.shape, dtype='uint8')
    x, y = np.where(water_mask > 1)
    canvas[x, y] = 1
    return canvas


def main():
    # Initialize KWCOCO dataset
    save_dir = '/media/mule/Projects/NASA/THP/Data'
    save_name = 'thp_nasa.kwcoco.json'
    save_path = os.path.join(save_dir, save_name)
    dset = kwcoco.CocoDataset()

    dataset_paths = get_dataset_paths()

    # Add water as class for data.
    cid = dset.add_category('water')

    # Add images, labels, and auxilary data to kwcoco file.
    for image_data in tqdm(dataset_paths.values(),
                           desc='Creating KWCOCO dataset'):
        image_path = image_data['image_path']
        label_path = image_data['label_path']
        dem_path = image_data['dem_path']
        slope_path = image_data['slope_path']

        image_file = rasterio.open(image_path)

        gid = dset.add_image(image_path,
                             height=image_file.height,
                             width=image_file.width,
                             channels='RGB_NIR')

        # Add label image.
        label_file = rasterio.open(label_path)
        label_img = label_file.read(1)
        bin_water_mask = binarize_water_mask(label_img)
        anno_mask = kwimage.structs.mask.Mask(bin_water_mask, format='c_mask')

        aid = dset.add_annotation(gid,
                                  category_id=cid,
                                  segmentation=anno_mask.to_coco())
        warp_func = None
        if (image_file.height != label_file.height) or (image_file.width !=
                                                        label_file.width):
            # Check that there is only a scale difference.
            height_ratio = image_file.height / label_file.height
            width_ratio = image_file.width / label_file.width
            # print(f'Image size: [{image_file.width},{image_file.height}]')
            # print(f'Label size: [{label_file.width},{label_file.height}]')
            # print(f'Height | Width Ratios: {height_ratio} | {width_ratio}')
            warp_func = {'scale': (width_ratio, height_ratio)}
        dset.add_auxiliary_item(gid,
                                label_path,
                                channels='label',
                                width=label_file.width,
                                height=label_file.height,
                                warp_aux_to_img=warp_func)

        # Add DEM image.
        dem_file = rasterio.open(dem_path)
        warp_func = None
        if (image_file.height != dem_file.height) or (image_file.width !=
                                                      dem_file.width):
            # Check that there is only a scale difference.
            height_ratio = image_file.height / dem_file.height
            width_ratio = image_file.width / dem_file.width
            # print(f'Image size: [{image_file.width},{image_file.height}]')
            # print(f'DEM size: [{dem_file.width},{dem_file.height}]')
            # print(f'Height | Width Ratios: {height_ratio} | {width_ratio}')
            warp_func = {'scale': (width_ratio, height_ratio)}
        dset.add_auxiliary_item(gid,
                                dem_path,
                                channels='DEM',
                                width=dem_file.width,
                                height=dem_file.height,
                                warp_aux_to_img=warp_func)

        slope_file = rasterio.open(dem_path)
        warp_func = None
        if (image_file.height != slope_file.height) or (image_file.width !=
                                                        slope_file.width):
            # Check that there is only a scale difference.
            height_ratio = image_file.height / slope_file.height
            width_ratio = image_file.width / slope_file.width
            # print(f'Image size: [{image_file.width},{image_file.height}]')
            # print(f'Slope size: [{slope_file.width},{slope_file.height}]')
            # print(f'Height | Width Ratios: {height_ratio} | {width_ratio}')
            warp_func = {'scale': (width_ratio, height_ratio)}
        dset.add_auxiliary_item(gid,
                                slope_path,
                                channels='slope',
                                width=slope_file.width,
                                height=slope_file.height,
                                warp_aux_to_img=warp_func)

    # Save KWCOCO file.
    dset.dump(save_path)


if __name__ == '__main__':
    main()
