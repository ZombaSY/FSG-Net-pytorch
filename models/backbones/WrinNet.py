import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from torchvision.transforms import Resize

# https://github.com/mingyu6yang/striped-wrinet

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class DoubleConvRes(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )
        self.res_connect = nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=False)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        res = self.res_connect(x)
        double_conv = self.double_conv(x)
        return self.act(res+double_conv)


class DoubleConvStriped(nn.Module):
    """Striped Conv"""

    def __init__(self, in_channels, out_channels,kernel_size = 3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size//2), bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)


class MSABlock(nn.Module):
    """MSA block"""

    def __init__(self,channels):
        super().__init__()
        self.strip_conv1 = DoubleConvStriped(channels,channels,kernel_size=3)
        self.strip_conv2 = DoubleConvStriped(channels,channels,kernel_size=7)
        self.strip_conv3 = DoubleConvStriped(channels,channels,kernel_size=11)
        self.conv1x1 = nn.Conv2d(3*channels, 1, kernel_size=1,bias=False)
        self.attn_func = nn.Sigmoid()

    def forward(self, x):
        strip1 = self.strip_conv1(x)
        strip2 = self.strip_conv2(x)
        strip3 = self.strip_conv3(x)
        strip_concat = torch.cat([strip1,strip2,strip3],dim=1)
        attn = self.attn_func(self.conv1x1(strip_concat))
        out = attn*x
        return out

class MSA(nn.Module):
    """MSA"""

    def __init__(self,c1,c2,c3,c4):
        super().__init__()
        self.msa_1 = MSABlock(c1)
        self.msa_2 = MSABlock(c2)
        self.msa_3 = MSABlock(c3)
        self.msa_4 = MSABlock(c4)

    def forward(self, x1,x2,x3,x4):
        x1_ = self.msa_1(x1)
        x2_ = self.msa_2(x2)
        x3_ = self.msa_3(x3)
        x4_ = self.msa_4(x4)
        return x1_,x2_,x3_,x4_

class HorizontalAttention(nn.Module):
    def __init__(self,channels,out_channels):
        super(HorizontalAttention,self).__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.l1 = nn.Linear(self.channels,self.out_channels)
        self.l2 = nn.Linear(self.channels,self.out_channels)
        self.l3 = nn.Linear(self.channels,self.out_channels)

    def forward(self, x,H,W):
        strip_pooling = nn.AdaptiveAvgPool2d((1, W))
        strip_x = strip_pooling(x).reshape(x.shape[0],-1,W)
        strip_x = strip_x.transpose(2,1)  # b w c

        Q = self.l1(strip_x) # b w c
        K = self.l2(strip_x) # b w c
        V = self.l3(strip_x) # b w c
        qk = torch.matmul(Q, K.transpose(2,1))
        qk = qk / math.sqrt(self.out_channels)
        qk = nn.Softmax(dim=-1)(qk)
        qkv = torch.matmul(qk, V)
        qkv = qkv.transpose(2,1)
        qkv = torch.unsqueeze(qkv,dim=2)
        qkv_expend = qkv.expand((-1,-1,H,-1))
        return qkv_expend

class VerticalAttention(nn.Module):
    def __init__(self,channels,out_channels):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.l1 = nn.Linear(self.channels,self.out_channels)
        self.l2 = nn.Linear(self.channels,self.out_channels)
        self.l3 = nn.Linear(self.channels,self.out_channels)


    def forward(self, x,H,W):
        strip_pooling = nn.AdaptiveMaxPool2d((H,1))
        strip_x = strip_pooling(x).reshape(x.shape[0],-1,H)
        strip_x = strip_x.transpose(2,1)  # b H c
        Q = self.l1(strip_x) # b w c
        K = self.l2(strip_x) # b w c
        V = self.l3(strip_x) # b w c
        qk = torch.matmul(Q, K.transpose(2,1))
        qk = qk / math.sqrt(self.out_channels)
        qk = nn.Softmax(dim=-1)(qk)
        qkv = torch.matmul(qk, V)
        qkv = qkv.transpose(2,1)
        qkv = torch.unsqueeze(qkv,dim=3)
        qkv_expend = qkv.expand((-1,-1,-1,W))
        return qkv_expend


class GSA(nn.Module):
    """GSA"""
    def __init__(self,c1,c2,c3,c4,out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(c1+c2+c3+c4, out_channels, kernel_size=1,bias=False)
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.out_channels = out_channels
        self.horizontal_attention = HorizontalAttention(out_channels,self.out_channels)
        self.vertical_attention = VerticalAttention(out_channels,self.out_channels)

    def forward(self, x1,x2,x3,x4):
        t_h, t_w = x1.shape[-2:]
        up = nn.Upsample(size=(t_h, t_w), mode='bilinear', align_corners=True)
        x2_ = up(x2)
        x3_ = up(x3)
        x4_ = up(x4)
        x_concat = torch.cat([x1,x2_,x3_,x4_],dim=1)
        x_concat_ = self.conv1x1(x_concat)
        hor_attn = self.horizontal_attention(x_concat_,t_h, t_w)
        ver_attn = self.vertical_attention(x_concat_,t_h, t_w)
        out = hor_attn+ver_attn+x_concat_
        x1_out = out
        x2_out = Resize(x2.shape[-2:])(out)
        x3_out = Resize(x3.shape[-2:])(out)
        x4_out = Resize(x4.shape[-2:])(out)
        return x1_out,x2_out,x3_out,x4_out



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UPBilinear(nn.Module):
    def __init__(self, in_channels, mid_channels,out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                )
        self.conv = DoubleConv(in_channels+mid_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class StripedWriNet(nn.Module):
    def __init__(self, n_channels, n_classes, init_c=24):
        super(StripedWriNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.init_c = init_c
        # self.inc = Stage0(n_channels, init_c)
        self.inc = DoubleConv(n_channels, init_c)
        self.down1 = Down(init_c, init_c*2)
        self.down2 = Down(init_c*2, init_c*4)
        self.down3 = Down(init_c*4, init_c*8)
        self.down4 = Down(init_c*8, init_c*16)

        self.msa = MSA(init_c,init_c*2,init_c*4,init_c*8)
        self.gsa = GSA(init_c,init_c*2,init_c*4,init_c*8,init_c*2)

        self.up1 = UPBilinear(init_c*16, init_c*2,init_c*8)
        self.up2 = UPBilinear(init_c*8, init_c*2,init_c*4)
        self.up3 = UPBilinear(init_c*4, init_c*2,init_c*2)
        self.up4 = UPBilinear(init_c*2, init_c*2,init_c*1)

        self.outc = OutConv(init_c*1, n_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        import pdb; pdb.set_trace()
        x1, x2, x3, x4 = self.msa(x1,x2,x3,x4)
        x1, x2, x3, x4 = self.gsa(x1,x2,x3,x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        out = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return out




if __name__ == '__main__':
    input = torch.rand([2,3,256,256])
    model = StripedWriNet(n_channels=3, n_classes=2)
    print(model(input).shape)