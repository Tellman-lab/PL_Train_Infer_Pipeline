import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score, accuracy_score
from PIL import Image
from osgeo import gdal

def load_image(file_path, is_tif=False):
    if is_tif:
        dataset = gdal.Open(file_path)
        band = dataset.GetRasterBand(1)
        array = band.ReadAsArray()
        return array
    else:
        return np.array(Image.open(file_path))

def calculate_metrics(ground_truth, prediction):
    # Flatten the arrays to 1D for metric calculation
    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    
    # Overall Accuracy
    oa = accuracy_score(ground_truth, prediction)
    
    # F1 Score
    f1 = f1_score(ground_truth, prediction, average='weighted')
    
    # Intersection over Union
    iou = jaccard_score(ground_truth, prediction, average='weighted')
    
    return oa, f1, iou

def main(folder_a, folder_b, output_csv):
    results = []
    
    for file_name in os.listdir(folder_a):
        if file_name.endswith('.jpg'):
            # Construct file paths
            ground_truth_path = os.path.join(folder_a, file_name)
            prediction_path = os.path.join(folder_b, file_name.replace('.jpg', '.tif'))
            
            if not os.path.exists(prediction_path):
                print(f"Prediction file not found for {file_name}")
                continue
            
            # Load images
            ground_truth = load_image(ground_truth_path)
            prediction = load_image(prediction_path, is_tif=True)
            
            # Calculate metrics
            oa, f1, iou = calculate_metrics(ground_truth, prediction)
            
            # Store results
            results.append({
                'Image': file_name,
                'Overall Accuracy': oa,
                'F1 Score': f1,
                'IoU': iou
            })
    
    # Create a DataFrame
    df = pd.DataFrame(results)
    
    # Calculate overall metrics
    overall_metrics = pd.DataFrame([{
        'Image': 'Overall',
        'Overall Accuracy': df['Overall Accuracy'].mean(),
        'F1 Score': df['F1 Score'].mean(),
        'IoU': df['IoU'].mean()
    }])
    
    # Concatenate overall metrics
    df = pd.concat([df, overall_metrics], ignore_index=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Define the directories and output file
ground_truth_dir = r'E:\RGV_DATA\RGV_30_labels'
prediction_dir = r'E:\Zhijie_PL_Pipeline\Infered_result\rexnet_thponly_attention_globalNorm_ep41_RGV\RGV_240603_30'
output_csv = prediction_dir.split('\\RGV_240603_30')[0] + '\\accuracy.csv'

# Run the main function
main(ground_truth_dir, prediction_dir, output_csv)
