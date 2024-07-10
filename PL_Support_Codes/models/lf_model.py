import torch
import torchmetrics
import torch.nn as nn

from PL_Support_Codes.models.unet import UNetEncoder, UNetDecoder
from PL_Support_Codes.models.water_seg_model import WaterSegmentationModel


class LateFusionModel(WaterSegmentationModel):

    def __init__(self,
                 in_channels,
                 n_classes,
                 lr,
                 log_image_iter=50,
                 to_rgb_fcn=None,
                 ignore_index=None,
                 model_used=None,
                 optimizer_name='adam',
                 feat_fusion='concat_conv'):
        self.feat_fusion = feat_fusion
        super().__init__(in_channels,
                         n_classes,
                         lr,
                         log_image_iter=log_image_iter,
                         to_rgb_fcn=to_rgb_fcn,
                         optimizer_name=optimizer_name,
                         ignore_index=ignore_index,
                         model_used=model_used)

    def _build_model(self):
        # Build models.
        self.encoders = nn.ModuleDict()
        if type(self.in_channels) is dict:
            n_in_channels = 0
            for input_name, feature_channels in self.in_channels.items():
                n_in_channels += feature_channels
                self.encoders[input_name] = UNetEncoder(feature_channels)

        self.decoder = UNetDecoder(self.n_classes)

        if self.feat_fusion == 'concat_conv':
            unet_feat_sizes = [64, 128, 256, 512, 512]
            self.concat_convs = nn.ModuleList()
            for fs in unet_feat_sizes:
                self.concat_convs.append(
                    nn.Conv2d(fs * len(self.in_channels), fs, 1, 1))

    def _set_model_to_train(self):
        self.encoders.train()
        self.decoder.train()
        self.concat_convs.train()

    def _set_model_to_eval(self):
        self.encoders.eval()
        self.decoder.eval()
        self.concat_convs.eval()

    def forward(self, batch):

        image_feats = self.encoders['ms_image'](batch['image'])

        extra_features = []
        if 'dem' in list(batch.keys()):
            extra_features.append(self.encoders['dem'](batch['dem']))

        if 'slope' in list(batch.keys()):
            extra_features.append(self.encoders['slope'](batch['slope']))

        if 'preflood' in list(batch.keys()):
            extra_features.append(self.encoders['preflood'](batch['preflood']))

        if 'pre_post_difference' in list(batch.keys()):
            extra_features.append(self.encoders['pre_post_difference'](
                batch['pre_post_difference']))

        if 'hand' in list(batch.keys()):
            extra_features.append(self.encoders['hand'](batch['hand']))

        for extra_feature in extra_features:
            for i, (a_feat,
                    b_feat) in enumerate(zip(image_feats, extra_feature)):
                image_feats[i] = torch.concat([a_feat, b_feat], dim=1)

        # Combine features.
        if self.feat_fusion == 'concat_conv':
            up_image_feat = []
            for img_feat, concat_conv in zip(image_feats, self.concat_convs):
                up_image_feat.append(concat_conv(img_feat))
        else:
            raise NotImplementedError

        output = self.decoder(up_image_feat)
        return output


if __name__ == '__main__':
    in_channels = {'ms_image': 4, 'dem': 1, 'slope': 1}
    n_classes = 2
    lr = 1e-4
    bs = 4
    img_size = [bs, 4, 64, 64]
    dem_size = [bs, 1, 64, 64]
    slope_size = [bs, 1, 64, 64]
    model = LateFusionModel(in_channels, n_classes, lr)

    ex_input = {
        'image': torch.zeros(img_size),
        'dem': torch.ones(dem_size),
        'slope': torch.ones(slope_size)
    }
    model.forward(ex_input)
