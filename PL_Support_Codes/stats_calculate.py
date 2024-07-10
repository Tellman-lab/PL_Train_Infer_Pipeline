import os
import argparse
from glob import glob

from tqdm import tqdm

from st_water_seg.predict import predict
from st_water_seg.utils.utils_misc import load_config

root_folder = "/home/zhijiezhang/Tranined_models_new_label/combined_csdap_PS_high+low_Aug"
summary_stats_folder = os.path.join(root_folder, 'summary_stats')
if not os.path.exists(summary_stats_folder): os.makedirs(summary_stats_folder)
exlude_list = ['summary_stats_orig', '.hydra', 'hydra', 'cross_validate.log', 'summary_stats']
folder_list = os.listdir(root_folder)
folder_list = [folder for folder in folder_list if folder not in exlude_list]
iou_image_list = []
F1_image_list = []
iou_region_list = []
F1_region_list = []

for folder in folder_list:
    print(folder)
    path_F1_image = glob(root_folder+'/'+folder+'/predictions/*/*/*/ranked_images_F1-score.txt')
    path_F1_region = glob(root_folder+'/'+folder+'/predictions/*/*/*/ranked_regions_F1-Score.txt')
    path_iou_image = glob(root_folder+'/'+folder+'/predictions/*/*/*/ranked_images_mIoU.txt')
    path_iou_region = glob(root_folder+'/'+folder+'/predictions/*/*/*/ranked_regions_iou.txt')
    
    with open(path_F1_image[0], 'r') as f:
        for _ in range(2):
            f.readline()
        for line in f:
            line = line.strip('\n')
            F1_image_list.append(line)
    # print(F1_image_list) 
    
    with open(path_iou_image[0], 'r') as f:
        for _ in range(2):
            f.readline()
        for line in f:
            line = line.strip('\n')
            iou_image_list.append(line)
    # print(F1_image_list) 
    
    with open(path_F1_region[0], 'r') as f:
        for _ in range(2):
            f.readline()
        for line in f:
            line = line.strip('\n')
            F1_region_list.append(line)
    # print(F1_image_list) 
    
    with open(path_iou_region[0], 'r') as f:
        for _ in range(2):
            f.readline()
        for line in f:
            line = line.strip('\n')
            iou_region_list.append(line)
            
total_iou = 0
total_F1 = 0
with open(os.path.join(summary_stats_folder, 'all_img_F1.txt'), 'w') as f1, open(os.path.join(summary_stats_folder, 'all_img_iou.txt'), 'w') as iou:
    for i in range(len(F1_image_list)):
        img_f1 = F1_image_list[i].split(' ')[1]
        img_f1 = float(img_f1.split('%')[0])
        iou_f1 = iou_image_list[i].split(' ')[1]
        iou_f1 = float(iou_f1.split('%')[0])
        total_iou += iou_f1
        total_F1 += img_f1
        f1.write(F1_image_list[i]+'\n')
        iou.write(iou_image_list[i]+'\n')

total_iou_region = 0
total_f1_region = 0
with open(os.path.join(summary_stats_folder, 'region_stats.txt'), 'w') as f:
    for i in range(len(F1_region_list)):
        img_f1 = F1_region_list[i].split(' ')[1]
        img_f1 = float(img_f1.split('%')[0])
        iou_f1 = iou_region_list[i].split(' ')[1]
        iou_f1 = float(iou_f1.split('%')[0])
        total_iou_region += iou_f1
        total_f1_region += img_f1
    avg_F1_region = total_f1_region/len(F1_region_list)
    avg_iou_region = total_iou_region/len(F1_region_list)
    avg_F1_img = total_F1/len(F1_image_list)
    avg_iou_img = total_iou/len(F1_image_list)
    
    f.write('Avg F1 region scores: {}\n'.format(avg_F1_region))
    f.write('-'*8+'\n')
    
    f.write('Avg IoU region scores: {}\n'.format(avg_iou_region))
    f.write('-'*8+'\n')
    
    f.write('Avg F1 scores from all img: {}\n'.format(avg_F1_img))
    f.write('-'*8+'\n')
    
    f.write('Avg IoU scores from all img: {}\n'.format(avg_iou_img))
    f.write('-'*8+'\n')
    
    f.write('F1 region scores:\n')
    f.write('-'*20+'\n')
    for line in F1_region_list:
        f.write(line+'\n')
    f.write('-'*20+'\n')
    f.write('-'*20+'\n')
    f.write('IoU region scores:\n')
    f.write('-'*20+'\n')
    for line in iou_region_list:
        f.write(line+'\n')