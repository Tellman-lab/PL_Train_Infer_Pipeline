import os
import argparse
from glob import glob

from tqdm import tqdm

from st_water_seg.predict import predict
from st_water_seg.utils.utils_misc import load_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cross_val_exp_dir', type=str)
    parser.add_argument('--predict_images', default=True, action='store_true')
    args = parser.parse_args()

    # Get all cross-validation subexperiments.
    ## Check if directory exists.
    if os.path.exists(args.cross_val_exp_dir) is False:
        print(f'Directory: {args.cross_val_exp_dir} not found.')

    ## Get all sub folders.
    exp_dirs = glob(args.cross_val_exp_dir + '/*/')
    print("exp_dirs orig:\n")
    print(exp_dirs)
    exp_dirs = [item for item in exp_dirs if item != (args.cross_val_exp_dir+"/hydra/") and item != (args.cross_val_exp_dir+"/.hydra/") and item != (args.cross_val_exp_dir+"/summary_stats_orig/")]
    print("exp_dirs:\n")
    print(exp_dirs)
    # exp_dirs = ['/home/zhijiezhang/Tranined_models_new_label/trained_models/Combined_csda_PS_high+low_aug/Bolivia/']

    for exp_dir in tqdm(exp_dirs, desc='Cross-val Predictions',
                        colour='green'):
        print(exp_dir)
        # Get configuration file.
        cfg_path = os.path.join(exp_dir, 'config.yaml')
        exp_cfg = load_config(cfg_path)

        # Get best model checkpoint path.
        ckpt_paths = sorted(
            glob(os.path.join(exp_dir, 'checkpoints') + '/*.ckpt'))

        best_ckpt_path = None
        best_iou_score = 0
        for ckpt_path in ckpt_paths:
            ckpt_name = os.path.split(ckpt_path)[1]
            iou_score = float(ckpt_name.split('=')[-1][:-5])

            if iou_score >= best_iou_score:
                best_ckpt_path = ckpt_path
                best_iou_score = iou_score

        # Print out message.
        region_name = exp_dir.split('/')[-2]
        print('\n' + '=' * 40)
        print(f'Region name: {region_name}')
        print(f'Best IoU score: {best_iou_score}')
        print(f'Checkpoint path: {best_ckpt_path}')
        print('=' * 40 + '\n')

        # Run prediction for this subexperiment.
        predict(cfg=exp_cfg,
                experiment_dir=exp_dir,
                checkpoint_path=best_ckpt_path,
                eval_dataset_name=exp_cfg.dataset.name,
                predict_images=args.predict_images,
                eval_region=region_name)
    
    # # summary_stats_folder = os.path.join(args.cross_val_exp_dir, 'summary_stats')
    # root_folder = args.cross_val_exp_dir
    # summary_stats_folder = os.path.join(root_folder, 'summary_stats_coincide_new')
    # if not os.path.exists(summary_stats_folder): os.makedirs(summary_stats_folder)
    # exlude_list = ['summary_stats_orig', '.hydra', 'hydra', 'cross_validate.log', 'summary_stats']
    # folder_list = os.listdir(root_folder)
    # folder_list = [folder for folder in folder_list if folder not in exlude_list]
    # iou_image_list = []
    # F1_image_list = []
    # iou_region_list = []
    # F1_region_list = []

    # for folder in folder_list:
    #     print(folder)
    #     path_F1_image = glob(root_folder+'/'+folder+'/predictions_coincide_new/*/*/*/ranked_images_F1-score.txt')
    #     path_F1_region = glob(root_folder+'/'+folder+'/predictions_coincide_new/*/*/*/ranked_regions_F1-Score.txt')
    #     path_iou_image = glob(root_folder+'/'+folder+'/predictions_coincide_new/*/*/*/ranked_images_mIoU.txt')
    #     path_iou_region = glob(root_folder+'/'+folder+'/predictions_coincide_new/*/*/*/ranked_regions_iou.txt')
        
    #     with open(path_F1_image[0], 'r') as f:
    #         for _ in range(2):
    #             f.readline()
    #         for line in f:
    #             line = line.strip('\n')
    #             F1_image_list.append(line)
    #     # print(F1_image_list) 
        
    #     with open(path_iou_image[0], 'r') as f:
    #         for _ in range(2):
    #             f.readline()
    #         for line in f:
    #             line = line.strip('\n')
    #             iou_image_list.append(line)
    #     # print(F1_image_list) 
        
    #     with open(path_F1_region[0], 'r') as f:
    #         for _ in range(2):
    #             f.readline()
    #         for line in f:
    #             line = line.strip('\n')
    #             F1_region_list.append(line)
    #     # print(F1_image_list) 
        
    #     with open(path_iou_region[0], 'r') as f:
    #         for _ in range(2):
    #             f.readline()
    #         for line in f:
    #             line = line.strip('\n')
    #             iou_region_list.append(line)
                
    # total_iou = 0
    # total_F1 = 0
    # with open(os.path.join(summary_stats_folder, 'all_img_F1.txt'), 'w') as f1, open(os.path.join(summary_stats_folder, 'all_img_iou.txt'), 'w') as iou:
    #     for i in range(len(F1_image_list)):
    #         img_f1 = F1_image_list[i].split(' ')[1]
    #         img_f1 = float(img_f1.split('%')[0])
    #         iou_f1 = iou_image_list[i].split(' ')[1]
    #         iou_f1 = float(iou_f1.split('%')[0])
    #         total_iou += iou_f1
    #         total_F1 += img_f1
    #         f1.write(F1_image_list[i]+'\n')
    #         iou.write(iou_image_list[i]+'\n')

    # total_iou_region = 0
    # total_f1_region = 0
    # with open(os.path.join(summary_stats_folder, 'region_stats.txt'), 'w') as f:
    #     for i in range(len(F1_region_list)):
    #         img_f1 = F1_region_list[i].split(' ')[1]
    #         img_f1 = float(img_f1.split('%')[0])
    #         iou_f1 = iou_region_list[i].split(' ')[1]
    #         iou_f1 = float(iou_f1.split('%')[0])
    #         total_iou_region += iou_f1
    #         total_f1_region += img_f1
    #     avg_F1_region = total_f1_region/len(F1_region_list)
    #     avg_iou_region = total_iou_region/len(F1_region_list)
    #     avg_F1_img = total_F1/len(F1_image_list)
    #     avg_iou_img = total_iou/len(F1_image_list)
        
    #     f.write('Avg F1 region scores: {}\n'.format(avg_F1_region))
    #     f.write('-'*8+'\n')
        
    #     f.write('Avg IoU region scores: {}\n'.format(avg_iou_region))
    #     f.write('-'*8+'\n')
        
    #     f.write('Avg F1 scores from all img: {}\n'.format(avg_F1_img))
    #     f.write('-'*8+'\n')
        
    #     f.write('Avg IoU scores from all img: {}\n'.format(avg_iou_img))
    #     f.write('-'*8+'\n')
        
    #     f.write('F1 region scores:\n')
    #     f.write('-'*20+'\n')
    #     for line in F1_region_list:
    #         f.write(line+'\n')
    #     f.write('-'*20+'\n')
    #     f.write('-'*20+'\n')
    #     f.write('IoU region scores:\n')
    #     f.write('-'*20+'\n')
    #     for line in iou_region_list:
    #         f.write(line+'\n')
