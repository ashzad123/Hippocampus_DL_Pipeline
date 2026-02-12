"""
V-Net (3D) model for medical image segmentation
"""
import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """
    Conv3D -> BatchNorm3D -> PReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock3D(nn.Module):
    """
    Residual block with N conv layers
    """
    def __init__(self, channels, num_convs=2):
        super().__init__()
        layers = [ConvBlock3D(channels, channels) for _ in range(num_convs)]
        self.convs = nn.Sequential(*layers)
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        out = self.convs(x)
        out = out + x
        return self.prelu(out)


class InputTransition(nn.Module):
    """
    Input transition: project input to base channels with residual
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels, kernel_size=5, padding=2)
        self.prelu = nn.PReLU(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        out = self.conv(x)
        # Repeat input channels to match out_channels for residual add
        if self.out_channels % self.in_channels == 0:
            repeat = self.out_channels // self.in_channels
            x_repeat = x.repeat(1, repeat, 1, 1, 1)
            out = out + x_repeat
        return self.prelu(out)


class DownTransition(nn.Module):
    """
    Down transition: strided conv + residual block
    """
    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        self.down_conv = ConvBlock3D(in_channels, out_channels, kernel_size=2, padding=0, stride=2)
        self.res_block = ResidualBlock3D(out_channels, num_convs=num_convs)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.res_block(x)
        return x


class UpTransition(nn.Module):
    """
    Up transition: transposed conv + concat + residual block
    """
    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.prelu = nn.PReLU(out_channels)
        self.conv_after_concat = ConvBlock3D(out_channels * 2, out_channels)
        self.res_block = ResidualBlock3D(out_channels, num_convs=num_convs)

    def forward(self, x, skip):
        x = self.up_conv(x)
        x = self.prelu(x)

        # Pad if needed to match skip connection sizes
        diffZ = skip.size()[2] - x.size()[2]
        diffY = skip.size()[3] - x.size()[3]
        diffX = skip.size()[4] - x.size()[4]

        if diffZ != 0 or diffY != 0 or diffX != 0:
            x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2,
                                      diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([skip, x], dim=1)
        x = self.conv_after_concat(x)
        x = self.res_block(x)
        return x


class OutputTransition(nn.Module):
    """
    Output transition: 1x1x1 conv to logits
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class VNet3D(nn.Module):
    """
    V-Net for 3D medical image segmentation

    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        base_channels: Number of base channels
    """
    def __init__(self, in_channels=1, num_classes=3, base_channels=16):
        super().__init__()
        self.in_tr = InputTransition(in_channels, base_channels)
        self.down1 = DownTransition(base_channels, base_channels * 2, num_convs=1)
        self.down2 = DownTransition(base_channels * 2, base_channels * 4, num_convs=2)
        self.down3 = DownTransition(base_channels * 4, base_channels * 8, num_convs=2)
        self.down4 = DownTransition(base_channels * 8, base_channels * 16, num_convs=2)

        self.up1 = UpTransition(base_channels * 16, base_channels * 8, num_convs=2)
        self.up2 = UpTransition(base_channels * 8, base_channels * 4, num_convs=2)
        self.up3 = UpTransition(base_channels * 4, base_channels * 2, num_convs=1)
        self.up4 = UpTransition(base_channels * 2, base_channels, num_convs=1)

        self.out_tr = OutputTransition(base_channels, num_classes)

    def forward(self, x):
        x1 = self.in_tr(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_tr(x)

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """
    Test the model with a dummy input
    """
    print("=" * 70)
    print("Testing V-Net Model")
    print("=" * 70)

    model = VNet3D(in_channels=1, num_classes=3, base_channels=16)

    print(f"\nModel: V-Net")
    print(f"Input channels: 1 (MRI)")
    print(f"Output classes: 3 (background, anterior, posterior)")
    print(f"Base channels: 16")
    print(f"Trainable parameters: {model.count_parameters():,}")

    print("\nTesting forward pass...")
    print("-" * 70)

    dummy_input = torch.randn(2, 1, 32, 48, 32)
    print(f"Input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    print("Expected shape: (2, 3, 32, 48, 32)")

    assert output.shape == (2, 3, 32, 48, 32), "Output shape mismatch!"

    print("\n" + "=" * 70)
    print("✓ Model test passed!")
    print("=" * 70)

    if torch.backends.mps.is_available():
        print("\nTesting on MPS (Apple Silicon GPU)...")
        device = torch.device("mps")
        model = model.to(device)
        dummy_input = dummy_input.to(device)

        with torch.no_grad():
            output = model(dummy_input)

        print(f"✓ MPS test passed! Output shape: {output.shape}")

    return model


if __name__ == "__main__":
    test_model()
