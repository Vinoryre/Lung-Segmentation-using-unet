import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    A module consisting of two convolutional layers each followed by a ReLU activation.
    The padding ensures the spatial dimensions are preserved.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downsampling block: Applies MaxPooling followed by a DoubleConv block.
    Used to reduce spatial dimensions and increase feature depth.
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)

class Up(nn.Module):
    """
    Upsampling block: Performs bilinear upsampling followed by concatenation with skip connection,
    then applies a DoubleConv block to refine features.
    """
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # After concatenation, the number of input channels doubles
        self.conv = DoubleConv(in_channels + in_channels // 2, out_channels)

    def forward(self, x1, x2):
        # Upsample the input feature map to match the size of the skip connection
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)

        # Handle possible mismatch in dimensions due to rounding
        if x1.size()[2] != x2.size()[2] or x1.size()[3] != x2.size()[3]:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = x1[:, :, diffY // 2: -diffY // 2, diffX // 2: -diffX // 2]

        # Concatennate along the channel axis
        x = torch.cat([x2, x1], dim = 1)
        return self.conv(x)

class UNet(nn.Module):
    """
    The U-Net model for image segmentation.
    Consists of an encoder (down= sampling path), a bottleneck, and a decoder (upsampling path).
    """
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)   # Initial convolution
        self.down1 = Down(64, 128)              # Downsample 1
        self.down2 = Down(128, 256)             # Downsample 2
        self.down3 = Down(256, 512)             # Downsample 3
        self.down4 = Down(512, 1024)            # Bottleneck (deepest layer)
        self.dropout = nn.Dropout(0.5)          # Optional dropout for regularization

        self.up1 = Up(1024, 512)                # Upsample 1
        self.up2 = Up(512, 256)                 # Upsample 2
        self.up3 = Up(256, 128)                 # Upsample 3
        self.up4 = Up(128, 64)                  # Upsample 4
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1) # Final 1x1 convolution to get class scores

    def forward(self, x):
        # Encoder path (downsampling with skip connections)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Apply dropout at the bottleneck
        x5 = self.dropout(x5)

        # Decoder path (upsampling and concatenating with skip connections)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)