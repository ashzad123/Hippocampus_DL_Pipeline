"""
3D U-Net model for medical image segmentation
"""

import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    """
    Double convolution block: (Conv3D -> BatchNorm -> ReLU) x 2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """
    Downscaling block: MaxPool -> DoubleConv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2), DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """
    Upscaling block: Upsample -> Concat -> DoubleConv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: from decoder, x2: from encoder (skip connection)
        x1 = self.up(x1)

        # Handle size mismatch due to odd dimensions
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = nn.functional.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
                diffZ // 2,
                diffZ - diffZ // 2,
            ],
        )

        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric segmentation

    Args:
        in_channels: Number of input channels (1 for MRI)
        num_classes: Number of output classes (3 for background + 2 hippocampus parts)
        base_channels: Number of channels in first layer (default: 16 for lighter model)
    """

    def __init__(self, in_channels=1, num_classes=3, base_channels=16):
        super().__init__()

        # Encoder (downsampling path)
        self.inc = DoubleConv3D(in_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.down4 = Down3D(base_channels * 8, base_channels * 16)

        # Decoder (upsampling path)
        self.up1 = Up3D(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up2 = Up3D(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up3 = Up3D(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up4 = Up3D(base_channels * 2 + base_channels, base_channels)

        # Output layer
        self.outc = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)  # base_channels
        x2 = self.down1(x1)  # base_channels * 2
        x3 = self.down2(x2)  # base_channels * 4
        x4 = self.down3(x3)  # base_channels * 8
        x5 = self.down4(x4)  # base_channels * 16 (bottleneck)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)

        return logits

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
