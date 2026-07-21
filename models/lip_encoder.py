import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.net(x)


class ConvResBlock(nn.Module):
    def __init__(self, channels):
        super(ConvResBlock, self).__init__()
        self.conv = BasicBlock(channels, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out + x)
        return out


class DownResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownResBlock, self).__init__()
        self.conv = BasicBlock(in_channels, out_channels, stride=2)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        residual = self.shortcut(x)
        out = self.relu(out + residual)
        return out


class LipEncoder(nn.Module):
    def __init__(self):
        super(LipEncoder, self).__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(5, 7, 7), stride=(1, 1, 1), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )

        self.conv1 = ConvResBlock(8)
        self.down1 = DownResBlock(8, 16)
        self.conv2 = ConvResBlock(16)
        self.conv3 = ConvResBlock(16)
        self.down2 = DownResBlock(16, 32)
        self.conv4 = ConvResBlock(32)
        self.conv5 = ConvResBlock(32)
        self.down3 = DownResBlock(32, 64)
        self.conv6 = ConvResBlock(64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.frontend3D(x)
        x = x.transpose(1, 2).reshape(B * T, -1, H, W)
        x = self.conv1(x)
        x = self.down1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.down2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.down3(x)
        x = self.conv6(x)
        out = self.avgpool(x)
        out = out.reshape(B, T, -1).transpose(1, 2)
        return out