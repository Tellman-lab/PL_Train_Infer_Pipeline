import os
import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning import Trainer
from pytorch_lightning.tuner.lr_finder import _LRFinder
# from lightning.pytorch.tuner import Tuner
from pytorch_lightning.tuner import Tuner

from PL_Support_Codes.models import build_model
from PL_Support_Codes.datasets import build_dataset
from PL_Support_Codes.datasets.utils import generate_image_slice_object
from PL_Support_Codes.utils.utils_misc import generate_innovation_script

class PrintLearningRateCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Access the current learning rate from the optimizer
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

# @hydra.main(version_base="1.1", config_path="conf", config_name="config")
def fit_model(cfg: DictConfig, overwrite_exp_dir: str = None) -> str:
    torch.set_default_tensor_type(torch.FloatTensor)

    resume_training = False
    # Get experiment directory.
    if overwrite_exp_dir is None:
        exp_dir = os.getcwd()
    else:
        exp_dir = overwrite_exp_dir

    # Load dataset.
    slice_params = generate_image_slice_object(cfg.crop_height, cfg.crop_width,
                                               cfg.crop_stride)

    if cfg.dataset_kwargs is None:
        cfg.dataset_kwargs = {}

    train_dataset = build_dataset(cfg.dataset_name,
                                  'train',
                                  slice_params,
                                  sensor=cfg.dataset_sensor,
                                  channels=cfg.dataset_channels,
                                  n_classes=cfg.model_n_classes,
                                  norm_mode=cfg.norm_mode,
                                  eval_region=cfg.eval_region,
                                  ignore_index=cfg.ignore_index,
                                  seed_num=cfg.seed_num,
                                  train_split_pct=cfg.train_split_pct,
                                  transforms=cfg.transforms,
                                  **cfg.dataset_kwargs)
    valid_dataset = build_dataset(cfg.dataset_name,
                                  'valid',
                                  slice_params,
                                  sensor=cfg.dataset_sensor,
                                  channels=cfg.dataset_channels,
                                  n_classes=cfg.model_n_classes,
                                  norm_mode=cfg.norm_mode,
                                  eval_region=cfg.eval_region,
                                  ignore_index=cfg.ignore_index,
                                  seed_num=cfg.seed_num,
                                  train_split_pct=cfg.train_split_pct,
                                  **cfg.dataset_kwargs)

    # Create dataloaders.
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.n_workers)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.n_workers)


    # Create model.
    model = build_model(cfg.model_name,
                        train_dataset.n_channels,
                        train_dataset.n_classes,
                        cfg.lr,
                        log_image_iter=cfg.log_image_iter,
                        to_rgb_fcn=train_dataset.to_RGB,
                        ignore_index=train_dataset.ignore_index,
                        model_used=cfg.model_used,
                        model_loss_fn_a=cfg.model_loss_fn_a,
                        model_loss_fn_b=cfg.model_loss_fn_b,
                        model_loss_fn_a_ratio=cfg.model_loss_fn_a_ratio,
                        model_loss_fn_b_ratio=cfg.model_loss_fn_b_ratio,
                        **cfg.model_kwargs)

    # Create logger.
    logger = pl.loggers.TensorBoardLogger(
        save_dir=os.path.join(exp_dir, 'tensorboard_logs'))
    print_lr_callback = PrintLearningRateCallback()
    # Train model.
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(exp_dir, 'checkpoints'),
        save_top_k=cfg.save_topk_models,
        save_last=True,
        every_n_epochs=1,
        mode='max',
        monitor="val_MulticlassJaccardIndex", 
        filename="model-{epoch:02d}-{val_MulticlassJaccardIndex:.4f}") # val_MulticlassJaccardIndex change its type to string
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    

    trainer = pl.Trainer(max_epochs=cfg.n_epochs,
                         accelerator="gpu", #this will automatically find CUDA or MPS devices
                         devices=1,
                         default_root_dir=exp_dir,
                         callbacks=[checkpoint_callback, lr_monitor, print_lr_callback],
                         logger=logger,
                         profiler=cfg.profiler,
                         limit_train_batches=cfg.limit_train_batches,
                         limit_val_batches=cfg.limit_val_batches)
    # # try:
    tuner = Tuner(trainer)
########original_lr_setting########
    lr_finder = tuner.lr_find(model, train_loader, valid_loader,min_lr=1e-6, max_lr=9e-4, num_training=100)
    ##############
    # lr_finder = tuner.lr_find(model, train_loader, valid_loader,min_lr=1e-8, max_lr=8e-7, num_training=100)
    suggested_lr = lr_finder.suggestion()
    print("Suggested Learning Rate:", suggested_lr)
    model.hparams.lr = suggested_lr 

    if resume_training:
        trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader,
                ckpt_path=r"E:\Zhijie_PL_Pipeline\Trained_model\RexNet_Unet_csda_2thp\checkpoints\model-epoch=23-val_MulticlassJaccardIndex=0.8515.ckpt")
    else:
        trainer.fit(model=model,
                    train_dataloaders=train_loader,
                    val_dataloaders=valid_loader)


    # # Return best model path.
    # return trainer.checkpoint_callback.best_model_path


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def config_fit_func(cfg: DictConfig):
    fit_model(cfg)


if __name__ == '__main__':
    config_fit_func()
