import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive 3×3 convolution blocks:
    Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=(64, 128, 256, 512)
    ):
        super().__init__()

        self.encoder_blocks = nn.ModuleList()
        self.decoder_upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        current_channels = in_channels

        for feature in features:
            self.encoder_blocks.append(
                DoubleConv(current_channels, feature)
            )
            current_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(
            features[-1],
            features[-1] * 2
        )

        # Decoder
        current_channels = features[-1] * 2

        for feature in reversed(features):
            self.decoder_upconvs.append(
                nn.ConvTranspose2d(
                    current_channels,
                    feature,
                    kernel_size=2,
                    stride=2
                )
            )

            # feature from upsampling + feature from skip connection
            self.decoder_blocks.append(
                DoubleConv(feature * 2, feature)
            )

            current_channels = feature

        # Final 1×1 convolution
        self.final_conv = nn.Conv2d(
            features[0],
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        skip_connections = []

        # Encoder
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections
        skip_connections = skip_connections[::-1]

        # Decoder
        for upconv, decoder_block, skip in zip(
            self.decoder_upconvs,
            self.decoder_blocks,
            skip_connections
        ):
            x = upconv(x)

            # Handle odd-sized input images
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )

            # Concatenate decoder feature with encoder skip feature
            x = torch.cat([skip, x], dim=1)

            x = decoder_block(x)

        # Segmentation logits
        return self.final_conv(x)


def test_unet():
    model = UNet(
        in_channels=3,
        out_channels=1
    )

    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    assert y.shape == (1, 1, 256, 256), (
        f"Expected output shape (1, 1, 256, 256), "
        f"but received {tuple(y.shape)}"
    )


if __name__ == "__main__":
    test_unet()