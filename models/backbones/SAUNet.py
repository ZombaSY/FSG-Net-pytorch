import torch
import torch.nn.functional as F
from torch import nn, Tensor


# https://github.com/Tensor-king/SA_Uet-pytorch/blob/master/model/SA_Unet.py


class DropBlock(nn.Module):
    def __init__(self, block_size: int = 5, p: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: Tensor) -> float:
        """计算gamma
        Args:
            x (Tensor): 输入张量
        Returns:
            Tensor: gamma
        """

        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.size()
        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Last_Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Last_Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            DropBlock(7, 0.18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # 原论文采用的是转置卷积，我们一般用双线性插值
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:  # 采用转置卷积的通道数会减少一半
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        # 为了防止maxpooling后得到的图片尺寸向下取整，不是整数倍
        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
            nn.Sigmoid(),
        )


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, 1, keepdim=True)
        x3 = torch.cat((x1, x2), dim=1)
        x4 = torch.sigmoid(self.conv(x3))
        x = x4 * x
        assert len(x.shape) == 4, f"好像乘不了"
        return x


class SA_UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 bilinear: bool = False,
                 base_c: int = 16):
        super(SA_UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.conv1 = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Last_Down(base_c * 4, base_c * 8)

        self.attn = Attention()
        self.conv2 = nn.Sequential(nn.Conv2d(base_c * 8, base_c * 8, kernel_size=3, padding=1, bias=False),
                                   DropBlock(7, 0.9),
                                   nn.BatchNorm2d(base_c * 8),
                                   nn.ReLU(inplace=True))

        self.up1 = Up(base_c * 8, base_c * 4, bilinear)
        self.up2 = Up(base_c * 4, base_c * 2, bilinear)
        self.up3 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.attn(x4)
        x6 = self.conv2(x5)
        x = self.up1(x6, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.out_conv(x)

        return logits
