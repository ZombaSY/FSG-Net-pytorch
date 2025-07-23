import torch.nn as nn
import torch
import torch.nn.functional as F

from collections import OrderedDict
from models.backbones import Unet_part
from models.backbones import UNeTPluss
from models.backbones import ResUNet as ResUNets
from models.backbones import SAUNet as SAUNets
from models.backbones import DCSAUUNet
from models.backbones import AGNet as AGNet_parts
from models.backbones import ConvUNeXt as ConvUNeXt_parts
from models.backbones import R2UNet as R2UNet_parts
from models.backbones import FRUNet as FRUNet_parts
from models.backbones import FSGNet as FSGNet_parts
from models.backbones import head
from models.backbones import swin
from models.backbones import HRNet


class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, bilinear=True, **kwargs):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = Unet_part.DoubleConv(in_channels, 64)
        self.down1 = Unet_part.Down(64, 128)
        self.down2 = Unet_part.Down(128, 256)
        self.down3 = Unet_part.Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Unet_part.Down(512, 1024 // factor)
        self.up1 = Unet_part.Up(1024, 512 // factor, bilinear)
        self.up2 = Unet_part.Up(512, 256 // factor, bilinear)
        self.up3 = Unet_part.Up(256, 128 // factor, bilinear)
        self.up4 = Unet_part.Up(128, 64, bilinear)
        self.outc = Unet_part.OutConv(64, n_classes)

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

        return torch.sigmoid(logits)


class UNet2P(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super().__init__()
        self.unet2p = UNeTPluss.UNet_2Plus(in_channels=in_channels, n_classes=n_classes)

    def forward(self, x):
        return self.unet2p(x)


class UNet3P_Deep(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super().__init__()
        self.unet3p = UNeTPluss.UNet_3Plus_DeepSup(in_channels=in_channels, n_classes=n_classes)

    def forward(self, x):
        return self.unet3p(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super().__init__()
        self.resunet = ResUNets.ResUnet(channel=in_channels, n_classes=n_classes)

    def forward(self, x):
        return self.resunet(x)


class ResUNet2P(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super().__init__()
        self.resunet2p = ResUNets.ResUnetPlusPlus(channel=in_channels, n_classes=n_classes)

    def forward(self, x):
        return self.resunet2p(x)


class SAUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, base_c=16, **kwargs):
        super().__init__()
        self.sa_unet = SAUNets.SA_UNet(in_channels=in_channels, num_classes=n_classes, base_c=base_c)

    def forward(self, x):
        return self.sa_unet(x)


class DCSAU_UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super().__init__()
        self.dcsau_unet = DCSAUUNet.DCSAU_UNet(img_channels=in_channels, n_classes=n_classes)

    def forward(self, x):
        return torch.sigmoid(self.dcsau_unet(x))


class AGNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, **kwargs):
        super().__init__()
        self.ag_net = AGNet_parts.AG_Net(in_channels=in_channels, n_classes=n_classes)

    def forward(self, x):
        out = [torch.sigmoid(item) for item in self.ag_net(x)]
        return out


class ATTUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super().__init__()
        self.attu_net = R2UNet_parts.AttU_Net(img_ch=in_channels, output_ch=n_classes)

    def forward(self, x):
        return torch.sigmoid(self.attu_net(x))


class R2UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, **kwargs):
        super().__init__()
        self.r2unet = R2UNet_parts.R2U_Net(img_ch=in_channels, output_ch=n_classes)

    def forward(self, x):
        return torch.sigmoid(self.r2unet(x))


class ConvUNeXt(nn.Module):
    def __init__(self, in_channels, n_classes, base_c=32, **kwargs):
        super().__init__()
        self.convunext = ConvUNeXt_parts.ConvUNeXt(in_channels=in_channels, num_classes=n_classes, base_c=base_c)

    def forward(self, x):
        out = self.convunext(x)
        out = out['out']

        return torch.sigmoid(out)


class FRUNet(nn.Module):
    def __init__(self, in_channels, n_classes, **kwargs):
        super().__init__()
        self.frunet = FRUNet_parts.FR_UNet(num_channels=in_channels, num_classes=n_classes)

    def forward(self, x):
        out = self.frunet(x)

        return torch.sigmoid(out)


class FSGNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 n_classes=1,
                 depths=[3, 3, 9, 3],
                 base_c=64,
                 kernel_size=3,
                 **kwargs):
        super().__init__()
        self.FSGNet = FSGNet_parts.FSGNet(in_channels, n_classes, base_c,
                                          depths=depths, kernel_size=kernel_size)

    def forward(self, x):
        return self.FSGNet(x)


class Swin_t(nn.Module):
    def __init__(self, in_channel, base_c=96):
        super().__init__()

        self.swin_transformer = swin.SwinTransformer(in_chans=in_channel,
                                                     embed_dim=base_c,
                                                     depths=[2, 2, 6, 2],
                                                     num_heads=[3, 6, 12, 24],
                                                     window_size=7,
                                                     mlp_ratio=4.,
                                                     qkv_bias=True,
                                                     qk_scale=None,
                                                     drop_rate=0.,
                                                     attn_drop_rate=0.,
                                                     drop_path_rate=0.3,
                                                     ape=False,
                                                     patch_norm=True,
                                                     out_indices=(0, 1, 2, 3),
                                                     use_checkpoint=False)

    def load_pretrained_imagenet(self, dst):
        pretrained_states = torch.load(dst)['model']
        pretrained_states_backbone = OrderedDict()

        for item in pretrained_states.keys():
            if 'head.weight' == item or 'head.bias' == item or 'norm.weight' == item or 'norm.bias' == item or 'layers.0.blocks.1.attn_mask' == item or 'layers.1.blocks.1.attn_mask' == item or 'layers.2.blocks.1.attn_mask' == item or 'layers.2.blocks.3.attn_mask' == item or 'layers.2.blocks.5.attn_mask' == item:
                continue
            pretrained_states_backbone[item] = pretrained_states[item]

        self.swin_transformer.remove_fpn_norm_layers()  # temporally remove fpn norm layers that not included on public-release model
        self.swin_transformer.load_state_dict(pretrained_states_backbone)
        self.swin_transformer.add_fpn_norm_layers()

    def forward(self, x):
        feat1, feat2, feat3, feat4 = self.swin_transformer(x)
        out_dict = {'feats': [feat1, feat2, feat3, feat4]}

        return out_dict


class Swin_tiny_segmentation(Swin_t):
    def __init__(self, num_class=1, in_channel=3, base_c=96, **kwargs):
        super().__init__(in_channel, base_c)

        self.uper_head = head.M_UPerHead_dsv(in_channels=[base_c, base_c * 2, base_c * 4, base_c * 8],
                                             in_index=[0, 1, 2, 3],
                                             pool_scales=(1, 2, 3, 6),
                                             channels=512,
                                             dropout_ratio=0.1,
                                             num_class=num_class,
                                             align_corners=False,)

    def forward(self, x):
        x_size = x.shape[2:]

        # get segmentation map
        feats = self.swin_transformer(x)

        out_dict = self.uper_head(*feats)
        out_dict['seg'] = F.interpolate(out_dict['seg'], x_size, mode='bilinear', align_corners=False)
        for i in range(len(out_dict['seg_aux'])):
            out_dict['seg_aux'][i] = F.interpolate(out_dict['seg_aux'][i], x_size, mode='bilinear', align_corners=False)
        out_dict['feats'] = feats

        return torch.sigmoid(out_dict['seg'])


class HRNet_t(nn.Module):
    def __init__(self, num_class=1, in_channel=3, base_c=96, **kwargs):
        super().__init__()
        # from config import config
        # from config import update_config
        # config = argparse.

        self.backbone = HRNet.HighResolutionNet()

    def forward(self, x):
        x = self.backbone(x)

        return torch.sigmoid(x)
