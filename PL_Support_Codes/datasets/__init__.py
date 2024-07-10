from collections import defaultdict

import torch
import numpy as np

from PL_Support_Codes.datasets.thp import THP_Dataset
from PL_Support_Codes.datasets.s1f11 import S1F11
from PL_Support_Codes.datasets.csdap import CSDAP_Dataset
from PL_Support_Codes.datasets.utils import get_dset_path
from PL_Support_Codes.datasets.combined import Combined_Dataset
from PL_Support_Codes.datasets.batch_infer import Batch_Infer_Dataset

DATASETS = {
    'csdap': CSDAP_Dataset,
    'thp': THP_Dataset,
    'combined': Combined_Dataset,
    's1f11': S1F11,
    'batch_infer': Batch_Infer_Dataset,
}


def tensors_and_lists_collate_fn(data):
    list_key_names = ['metadata']

    out_data = defaultdict(list)
    for ex in data:
        for k, v in ex.items():
            if isinstance(v, np.ndarray):
                v = torch.tensor(v)
            out_data[k].append(v)
        
    for k, v in out_data.items():
        if k in list_key_names:
            out_data[k] = v
        else:
            out_data[k] = torch.stack(v, dim=0)
    
    return out_data


def build_dataset(dset_name, split, slice_params, eval_region, sensor,
                  channels, n_classes, **kwargs):
    dset_root_dir = get_dset_path(dset_name)

    try:
        # Only directly input required parameters.
        dataset = DATASETS[dset_name](dset_root_dir,
                                      split,
                                      slice_params,
                                      channels=channels,
                                      n_classes=n_classes,
                                      eval_region=eval_region,
                                      sensor=sensor,
                                      **kwargs)
    except KeyError:
        raise KeyError(
            f'DATASETS dictionary does not contain a dataset class for dataset name "{dset_name}"'
        )
    return dataset
