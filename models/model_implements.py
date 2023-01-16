import torch.nn as nn
import torch

from models.backbones import Unet_part
from models.backbones import UNeTPluss
from models.backbones import ResUNet as ResUNets
from models.backbones import SAUNet as SAUNets
from models.backbones import DCSAUUNet
from models.backbones import AGNet as AGNet_parts
from models.backbones import ConvUNeXt as ConvUNeXt_parts
from models.backbones import R2UNet as R2UNet_parts
from models.backbones import FRUNet as FRUNet_parts
from models.backbones import neUNet as neUNet_parts


class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, bilinear=True):
        super(UNet, self).__init__()
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
    def __init__(self, in_channels=3, n_classes=1):
        super(UNet2P, self).__init__()
        self.unet2p = UNeTPluss.UNet_2Plus(in_channels=in_channels, n_classes=n_classes)

    def forward(self, x):
        return self.unet2p(x)


class UNet3P_Deep(nn.Module):
    def __init__(self, in_channels=3, n_classes=1):
        super(UNet3P_Deep, self).__init__()
        self.unet3p = UNeTPluss.UNet_3Plus_DeepSup(in_channels=in_channels, n_classes=n_classes)

    def forward(self, x):
        return self.unet3p(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1):
        super(ResUNet, self).__init__()
        self.resunet = ResUNets.ResUnet(channel=in_channels, n_classes=n_classes)

    def forward(self, x):
        return self.resunet(x)


class ResUNet2P(nn.Module):
    def __init__(self, in_channels=3, n_classes=1):
        super(ResUNet2P, self).__init__()
        self.resunet2p = ResUNets.ResUnetPlusPlus(channel=in_channels, n_classes=n_classes)

    def forward(self, x):
        return self.resunet2p(x)


class SAUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, base_c=16):
        super(SAUNet, self).__init__()
        self.sa_unet = SAUNets.SA_UNet(in_channels=in_channels, num_classes=n_classes, base_c=base_c)

    def forward(self, x):
        return self.sa_unet(x)


class DCSAU_UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1):
        super(DCSAU_UNet, self).__init__()
        self.dcsau_unet = DCSAUUNet.DCSAU_UNet(img_channels=in_channels, n_classes=n_classes)

    def forward(self, x):
        return torch.sigmoid(self.dcsau_unet(x))


class AGNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2):
        super(AGNet, self).__init__()
        self.ag_net = AGNet_parts.AG_Net(in_channels=in_channels, n_classes=n_classes)

    def forward(self, x):
        out = [torch.sigmoid(item) for item in self.ag_net(x)]
        return out


class ATTUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1):
        super(ATTUNet, self).__init__()
        self.attu_net = R2UNet_parts.AttU_Net(img_ch=in_channels, output_ch=n_classes)

    def forward(self, x):
        return torch.sigmoid(self.attu_net(x))


class R2UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1):
        super(R2UNet, self).__init__()
        self.r2unet = R2UNet_parts.R2U_Net(img_ch=in_channels, output_ch=n_classes)

    def forward(self, x):
        return torch.sigmoid(self.r2unet(x))


class ConvUNeXt(nn.Module):
    def __init__(self, in_channels, n_classes, base_c=32):
        super(ConvUNeXt, self).__init__()
        self.convunext = ConvUNeXt_parts.ConvUNeXt(in_channels=in_channels, num_classes=n_classes, base_c=base_c)

    def forward(self, x):
        out = self.convunext(x)
        out = out['out']

        return torch.sigmoid(out)


class FRUNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(FRUNet, self).__init__()
        self.frunet = FRUNet_parts.FR_UNet(num_channels=in_channels, num_classes=n_classes)

    def forward(self, x):
        out = self.frunet(x)

        return torch.sigmoid(out)


class neUNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 n_classes=1,
                 depths=[3, 3, 9, 3],
                 base_c=64,
                 kernel_size=3):
        super(neUNet, self).__init__()
        self.neunet = neUNet_parts.neUNet(in_channels, n_classes, base_c,
                                          depths=depths, kernel_size=kernel_size)

    def forward(self, x):
        return self.neunet(x)
