import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from math import ceil

###################################################################################################################################
# This part is used for Unet_Original
###################################################################################################################################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels,
                                         in_channels // 2,
                                         kernel_size=2,
                                         stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#TODO:
class UNet_Orig(nn.Module):
    # https://github.com/milesial/Pytorch-UNet/tree/master/unet
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_Orig, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def encode(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        feats = [x1, x2, x3, x4, x5]

        return feats

    def decode(self, feats):
        x1, x2, x3, x4, x5 = feats
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
###################################################################################################################################
# This is the end of Unet_Original
###################################################################################################################################



###################################################################################################################################
# This part is used for rexnet
###################################################################################################################################
# Memory-efficient Siwsh using torch.jit.script borrowed from the code in (https://twitter.com/jeremyphoward/status/1188251041835315200)
# Currently use memory-efficient SiLU as default:
USE_MEMORY_EFFICIENT_SiLU = True

if USE_MEMORY_EFFICIENT_SiLU:
    @torch.jit.script
    def silu_fwd(x):
        return x.mul(torch.sigmoid(x))


    @torch.jit.script
    def silu_bwd(x, grad_output):
        x_sigmoid = torch.sigmoid(x)
        return grad_output * (x_sigmoid * (1. + x * (1. - x_sigmoid)))


    class SiLUJitImplementation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return silu_fwd(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            return silu_bwd(x, grad_output)


    def silu(x, inplace=False):
        return SiLUJitImplementation.apply(x)

else:
    def silu(x, inplace=False):
        return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class SiLU(nn.Module):
    def __init__(self, inplace=True):
        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return silu(x, self.inplace)


def ConvBNAct(out, in_channels, channels, kernel=1, stride=1, pad=0,
              num_group=1, active=True, relu6=False):
    out.append(nn.Conv2d(in_channels, channels, kernel,
                         stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    if active:
        out.append(nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))


def ConvBNSiLU(out, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1):
    out.append(nn.Conv2d(in_channels, channels, kernel,
                         stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    out.append(SiLU(inplace=True))


class SE(nn.Module):
    def __init__(self, in_channels, channels, se_ratio=12):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, channels // se_ratio, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels // se_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // se_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, channels, t, stride, use_se=True, se_ratio=12,
                 **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels

        out = []
        if t != 1:
            dw_channels = in_channels * t
            ConvBNSiLU(out, in_channels=in_channels, channels=dw_channels)
        else:
            dw_channels = in_channels

        ConvBNAct(out, in_channels=dw_channels, channels=dw_channels, kernel=3, stride=stride, pad=1,
                  num_group=dw_channels, active=False)

        if use_se:
            out.append(SE(dw_channels, dw_channels, se_ratio))

        out.append(nn.ReLU6())
        ConvBNAct(out, in_channels=dw_channels, channels=channels, active=False, relu6=True)
        self.out = nn.Sequential(*out)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out[:, 0:self.in_channels] += x

        return out


class ReXNetV1(nn.Module):
    def __init__(self, input_ch=16, final_ch=180, width_mult=1.5, depth_mult=1.0, classes=1000,
                 use_se=True,
                 se_ratio=12,
                 dropout_ratio=0.2,
                 bn_momentum=0.9):
        super(ReXNetV1, self).__init__()


        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        use_ses = [False, False, True, True, True, True]

        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum([[element] + [1] * (layers[idx] - 1)
                       for idx, element in enumerate(strides)], [])
        if use_se:
            use_ses = sum([[element] * layers[idx] for idx, element in enumerate(use_ses)], [])
        else:
            use_ses = [False] * sum(layers[:])
        ts = [1] * layers[0] + [6] * sum(layers[1:])

        self.depth = sum(layers[:]) * 3
        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = input_ch / width_mult if width_mult < 1.0 else input_ch

        features = []
        in_channels_group = []
        channels_group = []

        # The following channel configuration is a simple instance to make each layer become an expand layer.
        for i in range(self.depth // 3):
            if i == 0:
                in_channels_group.append(int(round(stem_channel * width_mult)))
                channels_group.append(int(round(inplanes * width_mult)))
            else:
                in_channels_group.append(int(round(inplanes * width_mult)))
                inplanes += final_ch / (self.depth // 3 * 1.0)
                channels_group.append(int(round(inplanes * width_mult)))

        ConvBNSiLU(features, 3, int(round(stem_channel * width_mult)), kernel=3, stride=2, pad=1)

        for block_idx, (in_c, c, t, s, se) in enumerate(zip(in_channels_group, channels_group, ts, strides, use_ses)):
            features.append(LinearBottleneck(in_channels=in_c,
                                             channels=c,
                                             t=t,
                                             stride=s,
                                             use_se=se, se_ratio=se_ratio))

        pen_channels = int(1280 * width_mult)
        ConvBNSiLU(features, c, pen_channels)

        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)
        self.output = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv2d(pen_channels, classes, 1, bias=True))
        
    def extract_features(self, x):
        return self.features[:-1](x)
    
    def forward(self, x):
        x = self.features(x)
        x = self.output(x).flatten(1)
        return x

###################################################################################################################################
# This is the end of Unet_Original
###################################################################################################################################





###################################################################################################################################
# This part is used for UNet with CBAM
###################################################################################################################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # print("*"*25)
        # print(x.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


#TODO: 
class UNet_CBAM(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_CBAM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Load a pre-trained resnet model
        self.base_model = models.resnet18(pretrained=True)

        # Modify the first convolution layer to accept 4-channel input
        self.base_model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Layers from the pre-trained model
        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4

        # Decoder layers
        # self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.upconv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        # self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # Additional conv layer after concatenation
        # self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # Additional conv layer after concatenation
        # self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)   # Additional conv layer after concatenation
        # self.upconv4 = nn.ConvTranspose2d(64,64, kernel_size=2, stride=2)
        # # self.conv4 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        # # self.upconv5 = nn.ConvTranspose2d(32,16, kernel_size=2, stride=2)

        # self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # self.upconv5 = nn.ConvTranspose2d(64,2, kernel_size=2, stride=2)
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # Additional conv layer after concatenation
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # Additional conv layer after concatenation
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)   # Additional conv layer after concatenation
        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(64,64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        # self.conv4 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        # self.upconv5 = nn.ConvTranspose2d(32,16, kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upconv5 = nn.Sequential(
            nn.ConvTranspose2d(64,2, kernel_size=2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU())

        self.cbam1 = CBAM(512)
        self.cbam2 = CBAM(256)
        self.cbam3 = CBAM(128)
        self.cbam4 = CBAM(128)
        # Final classification layer
        self.out =nn.LogSoftmax(dim=1)
        self.classifier = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        att = True
        # Encoder (ResNet-18)
        # input torch.Size([1, 4, 1024, 1024])
        x1 = self.base_model.conv1(x)
        # x1.shape torch.Size([1, 64, 512, 512])
        x1 = self.base_model.bn1(x1)

        x1 = self.base_model.relu(x1)

        x1_ = self.base_model.maxpool(x1)

        x2 = self.layer1(x1_)
        # x2.shape torch.Size([1, 64, 256, 256])
        x3 = self.layer2(x2)
        # x3.shape torch.Size([1, 128, 128, 128])
        x4 = self.layer3(x3)
        # x4.shape torch.Size([1, 256, 64, 64])
        x5 = self.layer4(x4)
        # x5.shape torch.Size([1, 512, 32, 32])

        # Decoder
        # x = self.upconv1(x5)
        # x = torch.cat([x, x4], dim=1)
        # x = self.upconv2(x)
        # x = torch.cat([x, x3], dim=1)
        # x = self.upconv3(x)
        # x = torch.cat([x, x2], dim=1)
        # x = self.upconv4(x)
        # x = torch.cat([x, x1], dim=1)

        x = self.upconv1(x5) # torch.Size([1, 256, 64, 64])
        # ([15, 256, 20, 20])
        # breakpoint()
        # pass
        # batchnorm
        # activation
        x = self._resize_and_concat(x, x4) # torch.Size([1, 512, 64, 64]) [15, 256, 19, 19])
        if att:
            
            x = self.cbam1(x)
        x = self.conv1(x) 
        x = self.upconv2(x) # torch.Size([1, 128, 128, 128])
        x = self._resize_and_concat(x, x3) # torch.Size([1, 256, 128, 128])
        if att:
            x = self.cbam2(x)

        x = self.conv2(x)
        x = self.upconv3(x) # torch.Size([1, 64, 256, 256])
        x = self._resize_and_concat(x, x2) # torch.Size([1, 128, 256, 256])
        if att:
            x = self.cbam3(x)
        x = self.conv3(x)
        x = self.upconv4(x) # torch.Size([1, 64, 512, 512])
        x = self._resize_and_concat(x, x1) # torch.Size([1, 128, 512, 512])
        if att:
            x = self.cbam4(x)

        
        # Classifier
        x = self.conv4(x)
        x = self.upconv5(x) # torch.Size([1, 2, 1024, 1024])
        x = self.out(x) # torch.Size([1, 2, 1024, 1024])
        # x = self.classifier(x)

        return x
    
    def _resize_and_concat(self, upsampled, bypass):
        # Resize the upsampled tensor to match the size of the bypass tensor
        _, _, H, W = bypass.size()
        upsampled = F.interpolate(upsampled, size=(H, W), mode='bilinear', align_corners=True)
        return torch.cat([upsampled,bypass],dim=1)
###################################################################################################################################
# End of UNet with CBAM
###################################################################################################################################






###################################################################################################################################
# This part is used for late fusion model
###################################################################################################################################
class UNetEncoder(nn.Module):

    def __init__(self, n_channels, bilinear=True, base_feat_channels=64):
        super(UNetEncoder, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        bfc = base_feat_channels
        self.base_feat_channels = base_feat_channels

        self.inc = DoubleConv(n_channels, bfc)
        self.down1 = Down(bfc, bfc * 2)
        self.down2 = Down(bfc * 2, bfc * 4)
        self.down3 = Down(bfc * 4, bfc * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(bfc * 8, (bfc * 16) // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        feats = [x1, x2, x3, x4, x5]

        return feats


class UNetDecoder(nn.Module):

    def __init__(self,
                 n_classes,
                 bilinear=True,
                 channel_factor=1,
                 base_feat_channels=64):
        super(UNetDecoder, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        cf = channel_factor
        bfc = base_feat_channels
        self.base_feat_channels = base_feat_channels

        factor = 2 if bilinear else 1
        self.up1 = Up((bfc * 16) * cf, (bfc * 8) // factor, bilinear)
        self.up2 = Up((bfc * 8) // factor * (cf + 1), (bfc * 4) // factor,
                      bilinear)
        self.up3 = Up((bfc * 4) // factor * (cf + 1), (bfc * 2) // factor,
                      bilinear)
        self.up4 = Up((bfc * 2) // factor * (cf + 1), bfc, bilinear)
        self.outc = OutConv(bfc, n_classes)

    def forward(self, feats):
        x1, x2, x3, x4, x5 = feats
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def get_output_feats(self, feats):
        x1, x2, x3, x4, x5 = feats
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x
###################################################################################################################################
# End of late fusion model
###################################################################################################################################

## fastai implementation of RexnetUNet
import timm
from fastai.vision.models.unet import DynamicUnet

def create_timm_body(arch, pretrained=True, n_in=3, cut=None):
    model = timm.create_model(arch, pretrained=pretrained, features_only=True, in_chans=n_in)
    if cut is None:
        cut = -1
    return nn.Sequential(*list(model.children())[:cut])
class RexnetUNet(nn.Module):
    def __init__(self, n_channels, n_classes,base_model="rexnet_150"):
        super(RexnetUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_model = create_timm_body(base_model, pretrained=True, n_in=n_channels)
        self.model = DynamicUnet(self.base_model, n_classes,(256,256), self_attention=True)
    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    # Create an example input to model.
    b, c, h, w = 2, 3, 256, 256
    fake_image = torch.zeros([b, c, h, w])

    # Create model.
    # model = UNet(c, 2)
    model = RexnetUNet(c, 2)


    # Pass data through model.
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    fake_image = fake_image.to(device)
    model = model.to(device)

    output = model(fake_image)
    print(f'Input shape: [{b} {c} {h} {w}]')
    print(f'Ouput shape: {output.shape}')
