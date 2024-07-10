import os
import json
import argparse
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from scipy.special import softmax

from st_water_seg.models import build_model
from st_water_seg.datasets import build_dataset
from st_water_seg.datasets.utils import generate_image_slice_object
from st_water_seg.utils.utils_image import ImageStitcher_v2 as ImageStitcher
from st_water_seg.tools import load_cfg_file, create_gif, create_conf_matrix_pred_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('--eval_dataset_name', type=str)
    parser.add_argument('--predict_images',
                        default=True,
                        action='store_true',
                        help='Create image predictions')
    parser.add_argument('--eval_region',
                        type=str,
                        help='TODO hotfix for cross-val config save issue.')
    parser.add_argument(
        '--eval_dataset_split',
        type=str,
        default='test',
        help='The dataset split to evaluate on. Default: None')
    parser.add_argument(
        '--n_workers',
        type=int,
        default=None,
        help=
        'Number of CPU cores to utilize. Set to 0 for debugging. Default: The number of workers in the config file.'
    )
    args = parser.parse_args()

    # Get config path from checkpoint path and experiment structure.
    experiment_dir = '/'.join(args.checkpoint_path.split('/')[:-2])
    cfg_path = os.path.join(experiment_dir, '.hydra', 'config.yaml')
    if os.path.exists(cfg_path) is False:
        cfg_path = os.path.join(experiment_dir, 'hydra', 'config.yaml')
    cfg = load_cfg_file(cfg_path)

    # Detemine which dataset to predict model on.
    if args.eval_dataset_name is None:
        eval_dataset_name = cfg.dataset.name
    else:
        eval_dataset_name = args.eval_dataset_name

    # Overwrite number of workers if specified.
    if args.n_workers is None:
        n_workers = cfg.n_workers
    else:
        n_workers = args.n_workers

    predict(cfg,
            experiment_dir,
            args.checkpoint_path,
            eval_dataset_name=eval_dataset_name,
            predict_images=args.predict_images,
            eval_region=args.eval_region,
            eval_dataset_split=args.eval_dataset_split,
            n_workers=n_workers)

################################################s
# def save_image_stats(image_stats, pred_dir, metric_name):
#     """_summary_

#     Args:
#         image_stats (dict): Keys are the image names and the values are a list of metric values for each crop.
#         pred_dir (str): The path to save the generated result files to.
#         metric_name (str): Name of metric, used for file saving.
#     """

#     # Get average metric per image.
#     # TODO: Need to average by number of pixels in crop of image.
#     per_image_metric_values = [
#         np.mean(metric_values) for metric_values in image_stats.values()
#     ]

#     # Order images and metric values based on the average metric value per image.
#     sorted_image_paths = [
#         p for _, p in sorted(
#             zip(per_image_metric_values, list(image_stats.keys())))
#     ][::-1]
#     sorted_img_metric_values = sorted(per_image_metric_values)[::-1]

#     # Get image names.
#     image_names = [os.path.split(p)[1][:-4] for p in sorted_image_paths]

#     # Create and save values to file.
#     ranked_image_path = os.path.join(pred_dir,
#                                      f'ranked_images_{metric_name}.txt')
#     with open(ranked_image_path, 'w') as f:
#         f.write(f'Ranked image {metric_name} \n')
#         f.write('---------------------- \n')
#         for img_name, metric_value in zip(image_names,
#                                           sorted_img_metric_values):
#             f.write(f'{img_name}: {metric_value*100}% \n')


# def save_region_stats(region_stats, pred_dir, metric_name):
#     # Aggregate region metric scores.
#     region_scores = [np.mean(metric) for metric in region_stats.values()]

#     ## Order image paths based on f1-scores.
#     sorted_regions = [
#         p for _, p in sorted(zip(region_scores, list(region_stats.keys())))
#     ][::-1]
#     sorted_region_scores = sorted(region_scores)[::-1]

#     ## Save ranked region f1-scores.
#     ranked_region_path = os.path.join(pred_dir,
#                                       f'ranked_regions_{metric_name}.txt')
#     with open(ranked_region_path, 'w') as f:
#         f.write(f'Ranked region {metric_name} \n')
#         f.write('---------------------- \n')
#         for region_name, metric in zip(sorted_regions, sorted_region_scores):
#             f.write(f'{region_name}: {metric*100}% \n')
########################################################

def predict(cfg,
            experiment_dir,
            checkpoint_path,
            eval_dataset_name,
            predict_images=False,
            eval_region=None,
            eval_dataset_split='test',
            n_workers=0):
    # Load dataset.
    slice_params = generate_image_slice_object(cfg.crop_height, cfg.crop_width,
                                               cfg.crop_stride)

    if eval_region:
        cfg.eval_region = eval_region

    if cfg.dataset.dataset_kwargs is None:
        cfg.dataset.dataset_kwargs = {}

    if hasattr(cfg, 'seed_num') is False:
        cfg.seed_num = None

    eval_dataset = build_dataset(eval_dataset_name,
                                 eval_dataset_split,
                                 slice_params,
                                 sensor=cfg.dataset.sensor,
                                 channels=cfg.dataset.channels,
                                 norm_mode=cfg.norm_mode,
                                 eval_region=cfg.eval_region,
                                 ignore_index=cfg.ignore_index,
                                 seed_num=cfg.seed_num,
                                #  train_split_pct=cfg.train_split_pct,
                                 train_split_pct=0.8,
                                 **cfg.dataset.dataset_kwargs)

    model = build_model(cfg.model.name,
                        eval_dataset.n_channels,
                        eval_dataset.n_classes,
                        cfg.lr,
                        log_image_iter=cfg.log_image_iter,
                        to_rgb_fcn=eval_dataset.to_RGB,
                        ignore_index=eval_dataset.ignore_index,
                        **cfg.model.model_kwargs)

    # Load weights.
    model = model.load_from_checkpoint(checkpoint_path,
                                       in_channels=eval_dataset.n_channels,
                                       n_classes=eval_dataset.n_classes,
                                       lr=cfg.lr)
    model._set_model_to_eval()

    # Get device.
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = model.to(device)

    # Create folder to save predictions into.
    chkpt_name = checkpoint_path.split('/')[-1].split('.')[:-1][0]
    if cfg.eval_region is None:
        # TODO:
        pred_dir = os.path.join(experiment_dir, 'predictions_FP10mOrig',
                                eval_dataset_name,
                                f'split_pct_{cfg.train_split_pct}', chkpt_name)
    else:
        pred_dir = os.path.join(experiment_dir, 'predictions_FP10mOrig',
                                eval_dataset_name, cfg.eval_region, chkpt_name)
    os.makedirs(pred_dir, exist_ok=True)
    # TODO:

    # Prediction loop.
    image_stats_f1, region_stats_f1 = defaultdict(list), defaultdict(list)
    image_stats_iou, region_stats_iou = defaultdict(list), defaultdict(list)
    pbar = tqdm(range(eval_dataset.__len__()),
                desc='Model prediction',
                colour='green')

    rgb_canvases, pred_canvases, gt_canvases, class_pred_canvases = {}, {}, {}, {}
    with torch.no_grad():
        for example_index in pbar:
            # Get example from dataset.
            example = eval_dataset.__getitem__(example_index,
                                               output_metadata=True)

            # Add dimension to input data.
            example['image'] = torch.tensor(example['image'])[None]
            example['target'] = torch.tensor(example['target'])[None]
            if 'dem' in example.keys():
                example['dem'] = torch.tensor(example['dem'])[None]
            if 'slope' in example.keys():
                example['slope'] = torch.tensor(example['slope'])[None]
            if 'preflood' in example.keys():
                example['preflood'] = torch.tensor(example['preflood'])[None]
            if 'pre_post_difference' in example.keys():
                example['pre_post_difference'] = torch.tensor(
                    example['pre_post_difference'])[None]
            if 'hand' in example.keys():
                example['hand'] = torch.tensor(example['hand'])[None]

            # Send input data to device.
            for key, value in example.items():
                if isinstance(value, torch.Tensor):
                    example[key] = value.to(device)

            # Pass data into model and get predictions.
            output = model(example)

            # Compute metrics.
            pred = output.argmax(dim=1)
            # pred = softmax(output, axis=1)
            # pred = argmax
            flat_pred, flat_target = pred.flatten(), example['target'].flatten(
            )
            ######################################################
            metrics = model.test_metrics(flat_pred, flat_target)
            model.test_metrics.update(flat_pred, flat_target)
            ####################################################

            # # Track image and region F1-Scores.
            region_name = example['metadata']['region_name']

            ########################################################
            # # TODO: Make into a function.
            # try:
            #     f1_score_metric = metrics['test_F1Score'].item()
            #     jaccard_metric = metrics['test_JaccardIndex'].item()
            # except KeyError:
            #     f1_score_metric = metrics['test_MulticlassF1Score'].item()
            #     jaccard_metric = metrics['test_MulticlassJaccardIndex'].item()
            # image_stats_f1[example['metadata']['image_path']].append(f1_score_metric)
            # image_stats_iou[example['metadata']['image_path']].append(jaccard_metric)
                
            # try:
            #     region_stats_f1[region_name].append(
            #         f1_score_metric)
            #     region_stats_iou[region_name].append(
            #         jaccard_metric)
            # except KeyError:
            #     # No region name information.
            #     pass
            ###############################################################################

            if predict_images:
                # Get image name.
                image_name = os.path.splitext(
                    os.path.split(example['metadata']['image_path'])[1])[0]

                if region_name not in rgb_canvases.keys():
                    # Create image stitching classes.
                    region_save_dir = os.path.join(pred_dir,
                                                   'image_predictions',
                                                   region_name)
                    os.makedirs(region_save_dir, exist_ok=True)
                    rgb_canvases[region_name] = ImageStitcher(
                        region_save_dir,
                        image_type_name='rgb',
                        save_backend='PIL',
                        save_ext='.png')
                    pred_canvases[region_name] = ImageStitcher(
                        region_save_dir,
                        image_type_name='pred_softmax',
                        save_backend='PIL',
                        save_ext='.png')
                    gt_canvases[region_name] = ImageStitcher(
                        region_save_dir,
                        image_type_name='gt',
                        save_backend='PIL',
                        save_ext='.png')
                    class_pred_canvases[region_name] = ImageStitcher(
                        region_save_dir,
                        image_type_name='pred_class',
                        save_backend='tifffile',
                        save_ext='.tif')
                    # rgb_canvases[region_name] = ImageStitcher(
                    #     region_save_dir,
                    #     image_type_name='rgb',
                    #     save_backend='PIL',
                    #     save_ext='.png')
                    # pred_canvases[region_name] = ImageStitcher(
                    #     region_save_dir,
                    #     image_type_name='pred_softmax',
                    #     save_backend='PIL',
                    #     save_ext='.png')
                    # gt_canvases[region_name] = ImageStitcher(
                    #     region_save_dir,
                    #     image_type_name='gt',
                    #     save_backend='PIL',
                    #     save_ext='.png')
                    # class_pred_canvases[region_name] = ImageStitcher(
                    #     region_save_dir,
                    #     image_type_name='pred_class',
                    #     save_backend='tifffile',
                    #     save_ext='.tif')

                # Get images to save.
                ## Get prediction image.
                # pred = output[0].argmax(dim=0).cpu().numpy()
                pred = output[0].cpu().numpy()
                pred = rearrange(pred, 'c h w -> h w c')

                ## Convert to probability from logits.
                # breakpoint()
                # pass
                pred = softmax(pred, axis=-1)

                ## Class predictions.
                water_pred = np.zeros([pred.shape[0], pred.shape[1]],
                                      dtype='uint8')
                
                x, y = np.where(pred.argmax(axis=-1) == 1)
                # x, y = np.where(pred.argmax(dim=1) == 1)
            
                water_pred[x, y] = 1

                ## Get ground truth image.
                target = example['target'][0].cpu().numpy().astype('uint8')
                up_target = np.zeros(target.shape, dtype='uint8')
                x, y = np.where(target == 1)
                up_target[x, y] = 1

                ## Create CM image.
                # color_pred_image = create_conf_matrix_pred_image(pred, target)

                ## Get RGB image.
                ## Unnormalize image.
                image = example['image'][0].cpu()
                image = (image * example['std'][0]) + example['mean'][0]

                rgb_image = eval_dataset.to_RGB(image.numpy())
                # rgb_image = (rgb_image * 255).astype('uint8')

                # Add crop to canvases.
                crop_params = example['metadata']['crop_params']
                pred_canvases[region_name].add_image(pred, image_name,
                                                     crop_params,
                                                     crop_params.og_height,
                                                     crop_params.og_width)
                # breakpoint()
                # pass
                class_pred_canvases[region_name].add_image(
                    water_pred, image_name, crop_params, crop_params.og_height,
                    crop_params.og_width)
                # breakpoint()
                # pass
                rgb_canvases[region_name].add_image(rgb_image, image_name,
                                                    crop_params,
                                                    crop_params.og_height,
                                                    crop_params.og_width)
                gt_canvases[region_name].add_image(up_target, image_name,
                                                   crop_params,
                                                   crop_params.og_height,
                                                   crop_params.og_width)

        # Save prediction images.
        if predict_images:
            for region_name in pred_canvases.keys():
                # pred_img_canvases = pred_canvases[
                    # region_name].get_combined_images()
                # class_pred_canvases = class_pred_canvases[
                #     region_name].get_combined_images()
                # gt_img_canvases = gt_canvases[region_name].get_combined_images(
                # )
                # rgb_img_canvases = rgb_canvases[
                    # region_name].get_combined_images()
                save_paths, _, _ = class_pred_canvases[
                    region_name].save_images(save_class=True)
                save_paths, _, _ = pred_canvases[
                    region_name].save_images(save_class=False)
                _, _, _ = gt_canvases[region_name].save_images(save_class=False)
                _, _, _ = rgb_canvases[region_name].save_images(save_class=False)

                # for (img_name,
                #      pred_img), save_path in zip(pred_img_canvases.items(),
                #                                  save_paths):
                #     # Compute and save confusion matrix image.
                #     gt_img = gt_img_canvases[img_name]
                #     cm_img = create_conf_matrix_pred_image(
                #         pred_img.argmax(axis=-1), np.ceil(gt_img))
                #     save_path = os.path.join(
                #         '/'.join(save_path.split('/')[:-1]), 'cm.png')
                #     Image.fromarray(cm_img).save(save_path)

                #     # Create gif.
                #     gif_save_path = os.path.join(
                #         '/'.join(save_path.split('/')[:-1]), 'rgb_cm.gif')
                #     rgb_img = rgb_img_canvases[img_name]
                #     rgb_img = (rgb_img * 255).astype('uint8')
                #     create_gif([rgb_img, cm_img], gif_save_path)
##########################################################################################
        # # Save final metrics.
        # all_metrics = model.test_metrics.compute()
        # all_metrics['eval_dataset'] = eval_dataset_name
        # save_metrics_path = os.path.join(pred_dir, 'metrics.json')
        # for key, value in all_metrics.items():
        #     if isinstance(value, torch.Tensor):
        #         all_metrics[key] = value.item()
        # json.dump(all_metrics, open(save_metrics_path, 'w'), indent=4)

        # # Save ranked list of best and worst images and regions predicted.
        # save_image_stats(image_stats_f1, pred_dir, 'F1-score')
        # save_image_stats(image_stats_iou, pred_dir, 'mIoU')

        # if len(region_stats_iou.keys()) > 0:
        #     save_region_stats(region_stats_f1, pred_dir, 'F1-Score')
        #     save_region_stats(region_stats_iou, pred_dir, 'iou')
###############################################################################################


if __name__ == '__main__':
    main()
