import torch.nn as nn
import torch.optim as optim

import torch
import numpy as np
import torchmetrics
from einops import rearrange
import pytorch_lightning as pl

from PL_Support_Codes.models.unet import UNet_Orig
from PL_Support_Codes.models.unet import UNet_CBAM
from PL_Support_Codes.models.unet import ReXNetV1
from PL_Support_Codes.models.unet import RexnetUNet
# from PL_Support_Codes.models.unet import UNet
from PL_Support_Codes.tools import create_conf_matrix_pred_image
from ._functional import soft_dice_score, to_tensor
from ._functional import focal_loss_with_logits
# class DiceLoss(nn.Module):
#     def __init__(self, ignore_index=None):
#         super().__init__()
#         self.ignore_index = ignore_index

#     def forward(self, logits, true):
#         smooth = 1.0
#         logits = torch.softmax(logits, dim=1)
#         device = logits.device
#         true = true.long()
        
#         # Create one-hot encoding for true labels
#         true_one_hot = torch.eye(logits.size(1), device=device)[true]
#         true_one_hot = true_one_hot.permute(0, 3, 1, 2)

#         if self.ignore_index is not None:
#             mask = true != self.ignore_index
#             logits = logits * mask.unsqueeze(1)
#             true_one_hot = true_one_hot * mask.unsqueeze(1)
        
#         intersection = torch.sum(logits * true_one_hot, dim=(0, 2, 3))
#         cardinality = torch.sum(logits + true_one_hot, dim=(0, 2, 3))
#         dice_loss = (2. * intersection + smooth) / (cardinality + smooth)
#         return (1 - dice_loss).mean()

#: Loss binary mode suppose you are solving binary segmentation task.
#: That mean yor have only one class which pixels are labled as **1**,
#: the rest pixels are background and labeled as **0**.
#: Target mask shape - (N, H, W), model output mask shape (N, 1, H, W).
BINARY_MODE: str = "binary"

#: Loss multiclass mode suppose you are solving multi-**class** segmentation task.
#: That mean you have *C = 1..N* classes which have unique label values,
#: classes are mutually exclusive and all pixels are labeled with theese values.
#: Target mask shape - (N, H, W), model output mask shape (N, C, H, W).
MULTICLASS_MODE: str = "multiclass"

#: Loss multilabel mode suppose you are solving multi-**label** segmentation task.
#: That mean you have *C = 1..N* classes which pixels are labeled as **1**,
#: classes are not mutually exclusive and each class have its own *channel*,
#: pixels in each channel which are not belong to class labeled as **0**.
#: Target mask shape - (N, C, H, W), model output mask shape (N, C, H, W).
MULTILABEL_MODE: str = "multilabel"

    
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2, ignore_index=None, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.ignore_index = ignore_index
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         # Ensure inputs are of float type and targets are long (for one_hot and nll_loss)
#         inputs = inputs.float()
#         targets = targets.long()

#         # Handle ignore_index by masking
#         if self.ignore_index is not None:
#             mask = targets != self.ignore_index
#             inputs = inputs * mask.unsqueeze(1)
#             targets = targets * mask

#         # Calculate softmax over the inputs
#         BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha, ignore_index=self.ignore_index)

#         # Get the probabilities of the targets
#         pt = torch.exp(-BCE_loss)

#         # Calculate Focal Loss
#         focal_loss = ((1 - pt) ** self.gamma) * BCE_loss

#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss
from functools import partial
from ._functional import focal_loss_with_logits
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from ._functional import soft_dice_score, to_tensor
class FocalLoss(_Loss):
    def __init__(
        self,
        # mode: str,
        alpha: Optional[float] = None,
        gamma: Optional[float] = 2.0,
        ignore_index: Optional[int] = None,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
    ):
        """Compute Focal loss

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            alpha: Prior probability of having positive value in target.
            gamma: Power factor for dampening weight (focal strength).
            ignore_index: If not None, targets may contain values to be ignored.
                Target values equal to ignore_index will be ignored from loss computation.
            normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
            reduced_threshold: Switch to reduced focal loss. Note, when using this mode you
                should use `reduction="sum"`.

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        # assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = "multiclass"
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.focal_loss_fn(y_pred, y_true)

        elif self.mode == MULTICLASS_MODE:
            num_classes = y_pred.size(1)
            loss = 0

            # Filter anchors with -1 label from loss computation
            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index

            for cls in range(num_classes):
                cls_y_true = (y_true == cls).long()
                cls_y_pred = y_pred[:, cls, ...]

                if self.ignore_index is not None:
                    cls_y_true = cls_y_true[not_ignored]
                    cls_y_pred = cls_y_pred[not_ignored]

                loss += self.focal_loss_fn(cls_y_pred, cls_y_true)

        return loss






class DiceLoss(_Loss):
    def __init__(
        self,
        # mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        # assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = 'multiclass'
        # if classes is not None:
        #     assert (
        #         mode != BINARY_MODE
        #     ), "Masking classes is not supported with mode=binary"
        #     classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot(
                    (y_true * mask).to(torch.long), num_classes
                )  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # N, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(
            y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(
        self, output, target, smooth=0.0, eps=1e-7, dims=None
    ) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)
class WaterSegmentationModel(pl.LightningModule):

    def __init__(self,
                 in_channels,
                 n_classes,
                 lr,
                 log_image_iter=50,
                 to_rgb_fcn=None,
                 ignore_index=None,
                 model_used=None,
                 model_loss_fn_a=None,
                 model_loss_fn_b=None,
                 model_loss_fn_a_ratio=None,
                 model_loss_fn_b_ratio=None,
                 optimizer_name=None):
        super().__init__()
        self.lr = lr
        self.model_used = model_used
        self.model_loss_fn_a = model_loss_fn_a
        self.model_loss_fn_b = model_loss_fn_b
        self.model_loss_fn_a_ratio = model_loss_fn_a_ratio
        self.model_loss_fn_b_ratio = model_loss_fn_b_ratio
        self.optimizer_name = optimizer_name
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.ignore_index = ignore_index
        #TODO：
        self.any_validation_steps_executed = False
        self.any_test_steps_executed = False
        #TODO：

        # Build model.
        self._build_model()

        # Get metrics.
        # if self.ignore_index == -1:
        #     self.ignore_index = self.n_classes - 1
        self.tracked_metrics = self._get_tracked_metrics()
# TODO: find all loss funcs here
#https://pytorch.org/docs/stable/nn.html#loss-functions
        LOSS_FUNCS ={
            'cross_entropy': nn.CrossEntropyLoss,
            'dice': DiceLoss,
            'focal': FocalLoss,
            'l1': nn.L1Loss,
            'mse': nn.MSELoss
        }
        
        # Get loss function.
        self.loss_func_a = LOSS_FUNCS[self.model_loss_fn_a](ignore_index=self.ignore_index)
        self.loss_func_b = LOSS_FUNCS[self.model_loss_fn_b](ignore_index=self.ignore_index)

        # Log images hyperparamters.
        self.to_rgb_fcn = to_rgb_fcn
        self.log_image_iter = log_image_iter
        print("!!!!!!!!!!!!")
        print("!!!!!!!!!!!!")
        print("Model used: ",model_used)
        print("n_classes: ", n_classes)
        print("in_channels: ", in_channels)
        print("ignore_index: ", ignore_index)
        print("optimizer_name: ",optimizer_name)
        print(lr)
        print("!!!!!!!!!!!!")
        print("!!!!!!!!!!!!")



    def _get_tracked_metrics(self, average_mode='micro'):
        metrics = torchmetrics.MetricCollection([
            torchmetrics.F1Score(task="multiclass",num_classes=self.n_classes,ignore_index=self.ignore_index,average='micro'),
            torchmetrics.JaccardIndex(task="multiclass",
                                      num_classes=self.n_classes,
                                      ignore_index=self.ignore_index,
                                      average='micro'),
            torchmetrics.Accuracy(task="multiclass",
                                  num_classes=self.n_classes,
                                  ignore_index=self.ignore_index,
                                  average='micro'),
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def _compute_metrics(self, conf, target):
        # conf: [batch_size, n_classes, height, width]
        # target: [batch_size, height, width]

        pred = conf.argmax(dim=1)
        flat_pred, flat_target = pred.flatten(), target.flatten()

        batch_metrics = {}
        for metric_name, metric_func in self.tracked_metrics.items():
            metric_value = metric_func(flat_pred, flat_target)
            metric_value = torch.nan_to_num(metric_value)
            batch_metrics[metric_name] = metric_value.item()
        return batch_metrics

    def _build_model(self):
        # Build models.
        if type(self.in_channels) is dict:
            n_in_channels = 0
            for feature_channels in self.in_channels.values():
                n_in_channels += feature_channels
        MODELS_USED = {
            'unet_orig': UNet_Orig,
            'unet_cbam': UNet_CBAM,
            'rexnet': ReXNetV1,
            'rexnet_unet':RexnetUNet
        }
        print("Model used!!!!!!!!!: ",MODELS_USED[self.model_used])
        self.model = MODELS_USED[self.model_used](n_in_channels, self.n_classes)

    def forward(self, batch):
        images = batch['image']
        output = self.model(images)
        return output

    def _set_model_to_train(self):
        self.model.train()

    def _set_model_to_eval(self):
        self.model.eval()

    def training_step(self, batch, batch_idx):
        self._set_model_to_train()
        images, target = batch['image'], batch['target']
        output = self.forward(batch)
        loss_a = self.loss_func_a(output, target)
        loss_b = self.loss_func_b(output, target)

        loss = self.model_loss_fn_a_ratio * 1 * loss_a + self.model_loss_fn_b_ratio * loss_b
        if torch.isnan(loss):
            # Happens when all numbers are ignore numbers.
            loss = torch.nan_to_num(loss)
        pred = output.argmax(dim=1)
        flat_pred, flat_target = pred.flatten(), batch['target'].flatten()
        metric_output = self.train_metrics(flat_pred, flat_target)

        metric_output['train_loss_combined'] = loss
        metric_output['train_loss_a'] = loss_a
        metric_output['train_loss_b'] = loss_b
        self.log_dict(metric_output,
                      prog_bar=True,
                      on_step=True,
                      on_epoch=True)

        if False:
        # if (batch_idx % self.log_image_iter) == 0:
            # Unnormalize images
            images = (images * batch['std']) + batch['mean']
            for b in range(images.shape[0]):
                # Convert input image to RGB.
                input_image = images[b].detach().cpu().numpy()
                rgb_image = self.to_rgb_fcn(input_image)

                # Generate prediction image
                prediction = output[b].detach().argmax(dim=0).cpu().numpy()
                ground_truth = target[b].detach().cpu().numpy()
                cm_image = create_conf_matrix_pred_image(
                    prediction, ground_truth) / 255.0

                # Create title for logged image.
                # str_title = f'train_e{str(self.current_epoch).zfill(3)}_b{str(b).zfill(3)}.png'
                str_title = f'train_i{str(batch_idx).zfill(4)}_b{str(b).zfill(3)}.png'

                self.log_image_to_tensorflow(str_title, rgb_image, cm_image)

        return loss

    def validation_step(self, batch, batch_idx):
        self.any_validation_steps_executed = True
        self._set_model_to_eval()
        images, target = batch['image'], batch['target']
        output = self.forward(batch)

        loss = self.model_loss_fn_a_ratio * self.loss_func_a(output, target) + self.model_loss_fn_b_ratio * self.loss_func_b(output, target)
        if torch.isnan(loss):
            # Happens when all numbers are ignore numbers.
            loss = torch.nan_to_num(loss)

        pred = output.argmax(dim=1)
        flat_pred, flat_target = pred.flatten(), batch['target'].flatten()
        metric_output = self.valid_metrics(flat_pred, flat_target)
        # metric_output['val_MulticlassF1Score'] = metric_output['val_MulticlassF1Score'].item()
        # metric_output['val_MulticlassJaccardIndex'] = metric_output['val_MulticlassJaccardIndex'].item()

        # metric_output['val_MulticlassAccuracy'] = metric_output['val_MulticlassAccuracy'].item()
        self.valid_metrics.update(flat_pred, flat_target)

        # Log metrics and loss.
        metric_output['valid_loss'] = loss

        for key, value in  metric_output.items():
            if isinstance(value, torch.Tensor):
                metric_output[key] = value.item()
            else:
                metric_output[key] = value

        self.log_dict(metric_output,
                      prog_bar=True,
                      on_step=True,
                      on_epoch=True)

        if False:
        # if (batch_idx % self.log_image_iter) == 0:
            # Unnormalize images
            images = (images * batch['std']) + batch['mean']
            for b in range(images.shape[0]):
                # Convert input image to RGB.
                input_image = images[b].detach().cpu().numpy()
                rgb_image = self.to_rgb_fcn(input_image)

                # Generate prediction image
                prediction = output[b].detach().argmax(dim=0).cpu().numpy()
                ground_truth = target[b].detach().cpu().numpy()
                cm_image = create_conf_matrix_pred_image(
                    prediction, ground_truth) / 255.0

                # Create title for logged image.
                # str_title = f'valid_e{str(self.current_epoch).zfill(3)}_b{str(b).zfill(3)}.png'
                str_title = f'valid_i{str(batch_idx).zfill(4)}_b{str(b).zfill(3)}.png'

                self.log_image_to_tensorflow(str_title, rgb_image, cm_image)

    def test_step(self, batch, batch_idx):
        self.any_test_steps_executed = True
        self._set_model_to_eval()
        output = self.forward(batch)
        target = batch['target']

        loss = self.model_loss_fn_a_ratio * self.loss_func_a(output, target) + self.model_loss_fn_b_ratio * self.loss_func_b(output, target)

        # Track metrics.
        pred = output.argmax(dim=1)
        flat_pred, flat_target = pred.flatten(), batch['target'].flatten()
        self.test_metrics.update(flat_pred, flat_target)


        # Log metrics and loss.
        self.log_dict({'test_loss': loss},
                      prog_bar=True,
                      on_step=True,
                      on_epoch=True)

    def configure_optimizers(self):
        OPTIMIZERS = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'adamw': optim.AdamW,
            'adamax': optim.Adamax,
            'adadelta': optim.Adadelta,
            'adagrad': optim.Adagrad,
            'rmsprop': optim.RMSprop,
            'rprop': optim.Rprop,
            'asgd': optim.ASGD,
            'lbfgs': optim.LBFGS,
            'sparse_adam': optim.SparseAdam,
            'radam': optim.RAdam,
            'nadam': optim.NAdam,
        }
        optimizer = OPTIMIZERS[self.optimizer_name](self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        return optim_dict
    
    def on_before_batch_transfer(self, batch, dataloader_idx=0):
    # Function to convert tensors to float32, leaving other data types unchanged
        def to_float32(item):
            if isinstance(item, torch.Tensor):
                return item.float()
            elif isinstance(item, (list, tuple)):
                return type(item)(to_float32(x) for x in item)
            elif isinstance(item, dict):
                return {key: to_float32(value) for key, value in item.items()}
            else:
                return item

        return to_float32(batch)

# # TODO:
#     def validation_epoch_end(self, validation_step_outputs):
#         if len(validation_step_outputs) == 0:
#             self.test_f1_score = 0
#             self.test_iou = 0
#             self.test_acc = 0
#         else:
#             metric_output = self.valid_metrics.compute()
#             self.log_dict(metric_output)
# # TODO:
    

#     def test_epoch_end(self, test_step_outputs) -> None:
#         if len(test_step_outputs) == 0:
#             pass
#         else:
#             metric_output = self.test_metrics.compute()
#             self.log_dict(metric_output)

#             self.f1_score = metric_output['test_F1Score'].item()
#             self.acc = metric_output['test_Accuracy'].item()
#             self.iou = metric_output['test_JaccardIndex'].item()

    def on_validation_epoch_end(self):
        if not self.any_validation_steps_executed:
            # Handle case where no validation steps were executed
            self.log("val_no_steps", True)  # Example of logging a custom flag or handling as needed
        else:
            # Compute and log metrics as usual
            metric_output = self.valid_metrics.compute()
            self.log_dict(metric_output)
            self.valid_metrics.reset()
    
    # Reset the tracker for the next epoch
        self.any_validation_steps_executed = False

    def on_test_epoch_end(self):
        if not self.any_test_steps_executed:
            # Handle case where no test steps were executed
            self.log("test_no_steps", True)  # Example of logging a custom flag or handling as needed
        else:
            # Compute and log metrics as usual
            metric_output = self.test_metrics.compute()
            self.log_dict(metric_output)
            self.f1_score = metric_output['test_F1Score'].item()
            self.acc = metric_output['test_Accuracy'].item()
            self.iou = metric_output['test_JaccardIndex'].item()
            self.test_metrics.reset()
        
        # Reset the tracker for the next usage
        self.any_test_steps_executed = False



    def log_image_to_tensorflow(self, str_title, rgb_image, cm_image):
        """_summary_

        Args:
            str_title (str): Title for the image.
            rgb_image (np.array): A np.array of shape [height, width, 3].
            cm_image (np.array): A np.array of shape [height, width, 3].
        """

        # Combine images together.
        log_image = np.concatenate((rgb_image, cm_image), axis=0).transpose(
            (2, 0, 1))
        self.logger.experiment.add_image(str_title, log_image,
                                         self.global_step)