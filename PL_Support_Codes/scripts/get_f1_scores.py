import os
import json

import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from tifffile import tifffile
# from sklearn.metrics import precision_recall_fscore_support
import numpy as np
# import sklearn


def load_label_image(path):
    label = tifffile.imread(label_path)

    # Binarize label values to not-flood-water (0) and flood-water (1).
    height, width = label.shape
    # Value mapping:
    # 2: No water
    # 4: Low confidence water
    # 6: High confidence water

    # Get positive water label.

    binary_label = np.zeros([height, width], dtype='uint8')
    x, y = np.where(label == 3)
    binary_label[x, y] = 1

    return binary_label


if __name__ == '__main__':
    # Get prediction paths.
    pred_base_dir = '/media/mule/Projects/NASA/BlackSky/Data/chipped/1m/'
    pred_paths = sorted(glob(pred_base_dir + '/*.tif'))

    # Get label paths.
    label_base_dir = '/media/mule/Projects/NASA/BlackSky/Data/chipped/labels/'
    label_paths = sorted(glob(label_base_dir + '/*.tif'))

    metrics = {}
    f1_all = 0
    iou_all = 0
    n=0
    for pred_path, label_path in tqdm(zip(pred_paths, label_paths)):
        # Load prediction image.
        pred = tifffile.imread(pred_path)

        # Load label image.
        target = load_label_image(label_path)

        TP = np.sum(np.logical_and(target.flatten() == 1, pred.flatten() == 1))
        FP = np.sum(np.logical_and(target.flatten() == 0, pred.flatten() == 1))
        FN = np.sum(np.logical_and(target.flatten() == 1, pred.flatten() == 0))

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        F1 = 2 * (precision * recall) / (precision + recall)


        # # Compute metrics.
        # pr, rc, f1, _ = precision_recall_fscore_support(target.flatten(),
        #                                                 pred.flatten(),
        #                                                 average='binary',
        #                                                 pos_label=1)
        intersection = np.logical_and(target.flatten(), pred.flatten())
        union = np.logical_or(target.flatten(), pred.flatten())
        iou = np.sum(intersection) / np.sum(union)



        # Get name of image.
        img_name = os.path.splitext(os.path.split(pred_path)[1])[0]
        metrics[img_name] = {'precision': precision, 'recall': recall, 'f1_score': F1, 'iou':iou}
        f1_all = f1_all+F1
        iou_all = iou_all+iou
        n = n+1

        # Save RGB version of label image.
        label_save_path = os.path.splitext(label_path)[0] + '.png'
        Image.fromarray((target * 255).astype('uint8')).save(label_save_path)

    # Save metrics
    save_path = os.path.join(pred_base_dir, 'metrics.json')
    json.dump(metrics, open(save_path, 'w'), indent=2, sort_keys=True)
    print(f1_all)
    print(iou_all)
    print(n)
