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
from models.backbones import WrinNet as WrinNet_parts
from models.backbones import neUNet as neUNet_parts
from models.backbones import neUNet_v2 as neUNet_parts_v2
from models.backbones import neUNet_v3 as neUNet_parts_v3
from models.backbones import neUNet_v4 as neUNet_parts_v4
from models.backbones import neUNet_v5 as neUNet_parts_v5
from models.backbones import neUNet_v6 as neUNet_parts_v6
from models.backbones import neUNet_v7 as neUNet_parts_v7


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


class StripedWriNet(nn.Module):
    def __init__(self, in_channels, n_classes, base_c=24, **kwargs):
        super().__init__()
        self.wrinnet = WrinNet_parts.StripedWriNet(n_channels=in_channels, n_classes=n_classes, init_c=base_c)

    def forward(self, x):
        out = self.wrinnet(x)

        return torch.sigmoid(out)


class neUNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 n_classes=1,
                 depths=[3, 3, 9, 3],
                 base_c=64,
                 kernel_size=3,
                 **kwargs):
        super().__init__()
        self.neunet = neUNet_parts.neUNet(in_channels, n_classes, base_c,
                                          depths=depths, kernel_size=kernel_size)

    def forward(self, x):
        return self.neunet(x)


class neUNet_v2(nn.Module):
    def __init__(self,
                 in_channels=3,
                 n_classes=1,
                 depths=[3, 3, 9, 3],
                 base_c=64,
                 kernel_size=3,
                 **kwargs):
        super().__init__()
        self.freeze_layer = kwargs['freeze_layer']

        self.neunet = neUNet_parts_v2.neUNet(in_channels, n_classes, base_c, depths=depths, kernel_size=kernel_size, non_linear=kwargs['non_linear'])

        self.train_callback()

    def train_callback(self):
        if self.freeze_layer:
            for p in self.neunet.parameters():
                p.requires_grad = False
            for p in self.neunet.post_projection.parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.neunet(x)


class neUNet_v3(nn.Module):
    # freeze second conv layer and fix with the constant acquired from first conv layer
    def __init__(self,
                 in_channels=3,
                 n_classes=1,
                 depths=[3, 3, 9, 3],
                 base_c=64,
                 kernel_size=3,
                 **kwargs):
        super().__init__()
        self.freeze_layer = kwargs['freeze_layer']

        self.neunet = neUNet_parts_v3.neUNet(in_channels, n_classes, base_c, depths=depths, kernel_size=kernel_size,
                                             non_linear=kwargs['non_linear'], project_dim=kwargs['project_dim'])

        self.train_callback()

    def train_callback(self):
        if self.freeze_layer:
            for p in self.neunet.parameters():
                p.requires_grad = False
            for p in self.neunet.post_projection[0].parameters():
                p.requires_grad = True

    def iteration_callback(self):
        # re-proejection
        self.neunet.post_projection[2].weight.data = torch.transpose(self.neunet.post_projection[0].weight.data, 0, 1)
        for p in self.neunet.post_projection[2].parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.neunet(x)


class neUNet_v4(nn.Module):
    # only use MSA layer from WrinNet
    def __init__(self,
                 in_channels=3,
                 n_classes=1,
                 depths=[3, 3, 9, 3],
                 base_c=64,
                 kernel_size=3,
                 **kwargs):
        super().__init__()
        self.freeze_layer = kwargs['freeze_layer']

        self.neunet = neUNet_parts_v4.neUNet(in_channels, n_classes, base_c, depths=depths, kernel_size=kernel_size)

    def forward(self, x):
        return self.neunet(x)


class neUNet_v5(nn.Module):
    # modify MSA layer with deformable convolution
    def __init__(self,
                 in_channels=3,
                 n_classes=1,
                 depths=[3, 3, 9, 3],
                 base_c=64,
                 kernel_size=3,
                 **kwargs):
        super().__init__()
        self.freeze_layer = kwargs['freeze_layer']

        self.neunet = neUNet_parts_v5.neUNet(in_channels, n_classes, base_c, depths=depths, kernel_size=kernel_size)

    def forward(self, x):
        return self.neunet(x)


class neUNet_v6(nn.Module):
    # modify Guided attention filter with stripe convolution
    def __init__(self,
                 in_channels=3,
                 n_classes=1,
                 depths=[3, 3, 9, 3],
                 base_c=64,
                 kernel_size=3,
                 **kwargs):
        super().__init__()
        self.freeze_layer = kwargs['freeze_layer']

        self.neunet = neUNet_parts_v6.neUNet(in_channels, n_classes, base_c, depths=depths, kernel_size=kernel_size)

    def forward(self, x):
        return self.neunet(x)


class neUNet_v7(nn.Module):
    # TBD.
    def __init__(self,
                 in_channels=3,
                 n_classes=1,
                 depths=[3, 3, 9, 3],
                 base_c=64,
                 kernel_size=3,
                 **kwargs):
        super().__init__()
        self.freeze_layer = kwargs['freeze_layer']

        self.neunet = neUNet_parts_v7.neUNet(in_channels, n_classes, base_c, depths=depths, kernel_size=kernel_size)

    def forward(self, x):
        return self.neunet(x)


class UNet_v2(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, bilinear=True, project_dim=2, non_linear=False, **kwargs):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.post_projection = nn.Sequential(*[
            nn.Conv2d(3, project_dim, kernel_size=1, bias=False),
            nn.ReLU() if non_linear else nn.Identity(),
            nn.Conv2d(project_dim, 3, kernel_size=1, bias=False),
            nn.ReLU() if non_linear else nn.Identity(),
        ])

        # acquired from PCA
        params = [-0.49403217, -0.57345206, -0.65351737, 0.76348513, 0.07347065, -0.6416327]
        param_tensor = torch.Tensor(params[:2 * 3]).view(2, 3, 1, 1)
        self.post_projection[0].weight.data = param_tensor
        self.post_projection[2].weight.data = torch.transpose(param_tensor, 0, 1)

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

        self.freeze_layer = kwargs['freeze_layer']
        self.train_callback()

    def train_callback(self):
        if self.freeze_layer:
            for p in self.parameters():
                p.requires_grad = False
            for p in self.post_projection[0].parameters():
                p.requires_grad = True

    def iteration_callback(self):
        # re-proejection
        self.post_projection[2].weight.data = torch.transpose(self.post_projection[0].weight.data, 0, 1)
        for p in self.post_projection[2].parameters():
            p.requires_grad = False

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
