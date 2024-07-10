import torch

from PL_Support_Codes.models.water_seg_model import WaterSegmentationModel


class EarlyFusionModel(WaterSegmentationModel):

    def __init__(self,
                 in_channels,
                 n_classes,
                 lr,
                 log_image_iter=50,
                 to_rgb_fcn=None,
                 ignore_index=None,
                 model_used = None,
                 model_loss_fn_a=None,
                 model_loss_fn_b=None,
                 model_loss_fn_a_ratio=None,
                 model_loss_fn_b_ratio=None,
                 optimizer_name=None):
        self.model_used = model_used
        self.optimizer_name = optimizer_name
        self.model_loss_fn_a = model_loss_fn_a
        self.model_loss_fn_b = model_loss_fn_b
        self.model_loss_fn_a_ratio = model_loss_fn_a_ratio
        self.model_loss_fn_b_ratio = model_loss_fn_b_ratio
        super().__init__(in_channels,
                         n_classes,
                         lr,
                         log_image_iter,
                         to_rgb_fcn,
                         ignore_index=ignore_index,
                         model_used=model_used,
                         model_loss_fn_a=model_loss_fn_a,
                         model_loss_fn_b=model_loss_fn_b,
                         model_loss_fn_a_ratio=model_loss_fn_a_ratio,
                         model_loss_fn_b_ratio=model_loss_fn_b_ratio,
                         optimizer_name=optimizer_name)

    def forward(self, batch):
        images = batch['image']

        extra_features = []
        if 'dem' in list(batch.keys()):
            extra_features.append(batch['dem'])

        if 'slope' in list(batch.keys()):
            extra_features.append(batch['slope'])

        if 'preflood' in list(batch.keys()):
            extra_features.append(batch['preflood'])

        if 'pre_post_difference' in list(batch.keys()):
            extra_features.append(batch['pre_post_difference'])

        if 'hand' in list(batch.keys()):
            extra_features.append(batch['hand'])

        for extra_feature in extra_features:
            images = torch.concat([images, extra_feature], dim=1)

        output = self.model(images)
        return output


if __name__ == '__main__':
    in_channels = {'ms_image': 4, 'dem': 1, 'slope': 1}
    n_classes = 2
    lr = 1e-4
    bs = 4
    img_size = [bs, 4, 64, 64]
    dem_size = [bs, 1, 64, 64]
    slope_size = [bs, 1, 64, 64]
    model = EarlyFusionModel(in_channels, n_classes, lr)

    ex_input = {
        'image': torch.zeros(img_size),
        'dem': torch.ones(dem_size),
        'slope': torch.ones(slope_size)
    }
    model.forward(ex_input)
