import numpy as np
import rasterio
import os
from glob import glob
from tqdm import tqdm
import shutil
import tempfile

def georeference_tiff(unref_tiff_path, reference_tiff_path, output_tiff_path):
    with rasterio.open(reference_tiff_path) as reference:
        transform = reference.transform
        crs = reference.crs

    with rasterio.open(unref_tiff_path) as target:
        profile = target.profile
        profile.update({
            'crs': crs,
            'transform': transform
        })

        with rasterio.open(output_tiff_path, 'w', **profile) as dst:
            dst.write(target.read(1), 1)

def compress_tiff_to_int8(input_path, output_path, compression='lzw'):
    with rasterio.open(input_path) as src:
        # Read the data from the first band
        data = src.read(1)
        
        # Convert data to int8
        data_int8 = data.astype(np.int8)
        
        # Adjust the no-data value to int8
        nodata = src.nodata
        if nodata is not None:
            nodata_int8 = np.int8(nodata)
            data_int8[data == nodata] = nodata_int8
        else:
            nodata_int8 = np.int8(0)  # Assuming 0 as no-data value if not set
            
        # Update the metadata for writing
        meta = src.meta.copy()
        meta.update(dtype=rasterio.uint8, nodata=nodata_int8, compress=compression)
        
        # Delete the existing output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # Write the modified data to a new TIFF file with compression
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data_int8, 1)
            dst.update_tags(ns='rio_overview', resampling='nearest')

def multiply_bands(geo_refed_file, mask_file, cloud_mask_file, output_file):
    with rasterio.open(geo_refed_file) as src1, rasterio.open(mask_file) as src2, rasterio.open(cloud_mask_file) as src3:
        band1 = src1.read(1)
        mask = src2.read(1)
        cloud_mask = src3.read(1)

        result = band1.copy()
        result[band1 == 0] = 1
        result[band1 == 255] = 2
        result[cloud_mask == 0] = 3
        result[mask == 0] = 0

        out_meta = src1.meta.copy()

        # Write to a temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmpfile:
            temp_output_file = tmpfile.name

        with rasterio.open(temp_output_file, "w", **out_meta) as dst:
            dst.nodata = 0
            dst.write(result, 1)

    # Compress the temporary file and overwrite the original output file
    compress_tiff_to_int8(temp_output_file, output_file, compression='lzw')

    # Remove the temporary file
    os.remove(temp_output_file)

# Open the geo-referenced TIFF file
def create_mask(root_folder, save_mask_folder):
    event_list = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
    for event in tqdm(event_list):
        img_list = glob(os.path.join(root_folder, event, "*_SR.tif"))

        for img in tqdm(img_list):
            img_name = img.split("\\")[-1].split('.')[0]
            input_file1 = img

            with rasterio.open(input_file1) as src1:
                data1 = src1.read()
                profile1 = src1.profile

            # Create a new 1-band array and set its values to 0
            rows, cols = data1.shape[1], data1.shape[2]
            result_data = np.zeros((rows, cols), dtype=np.uint8)

            # Iterate through each band and set result_data to 1 if there is data in any band
            nodata_value = profile1["nodata"]
            for band in data1:
                result_data = np.where((band != nodata_value) | (result_data == 1), 1, 0)

            # Update the profile for the 1-band output TIFF file
            profile1["count"] = 1
            profile1["dtype"] = "uint8"

            # Save the result to a new geo-referenced TIFF file
            save_folder = save_mask_folder + "\\" + event
            if not os.path.exists(save_folder): os.makedirs(save_folder)
            output_file = save_folder + "\\" + img_name + "_mask.tif"
            with rasterio.open(output_file, 'w', **profile1) as dst:
                dst.write(result_data, 1)

root_folder_prediction = r"E:\Zhijie_PL_Pipeline\Infered_result\RGV_2019_test"
GT_folder = r"E:\Zhijie_PL_Pipeline\DATA\RGV_2019_test"
save_folder_mask = r"E:\Zhijie_PL_Pipeline\Infered_result\RGV_2019_test_mask"
save_folder_final = r"E:\Zhijie_PL_Pipeline\Infered_result\RGV_2019_test_georef_cloud_final"

# Georeference TIFFs and Create Masks
events = [f for f in os.listdir(root_folder_prediction) if not f.startswith('.')]
create_mask(GT_folder, save_folder_mask)
for event in tqdm(events):
    event_pred_folder = os.path.join(root_folder_prediction, event)
    event_gt_folder = os.path.join(GT_folder, event)
    event_mask_folder = os.path.join(save_folder_mask, event)
    event_final_folder = os.path.join(save_folder_final, event)
    
    if not os.path.exists(event_mask_folder):
        os.makedirs(event_mask_folder)
    if not os.path.exists(event_final_folder):
        os.makedirs(event_final_folder)

    imgs = glob(os.path.join(event_pred_folder, "*.tif"))
    for img in imgs:
        output_georef_path = os.path.join(event_final_folder, os.path.basename(img))
        georeference_tiff(img, os.path.join(event_gt_folder, os.path.basename(img)), output_georef_path)

        mask_name = os.path.basename(output_georef_path).replace(".tif", "_mask.tif")
        mask_path = os.path.join(event_mask_folder, mask_name)
        udm_name = os.path.basename(output_georef_path).replace("AnalyticMS_SR", "udm2")
        udm_path = os.path.join(event_gt_folder, udm_name)
        multiply_bands(output_georef_path, mask_path, udm_path, output_georef_path)
