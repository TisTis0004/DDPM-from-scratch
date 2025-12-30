import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # the original 2015 paper kept the number of out_channels the same
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.downsample = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downsample.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.upsample.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.upsample.append(DoubleConv(feature * 2, feature))

        self.bottleNeck = DoubleConv(
            in_channels=features[-1], out_channels=features[-1] * 2
        )
        self.finalConv = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downsample:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleNeck(x)

        skip_connections = skip_connections[::-1]  # reverse

        for idx in range(0, len(self.upsample), 2):
            x = self.upsample[idx](x)
            # cause we are stepping 2s and we want to get the skip connections linearly
            skip = skip_connections[idx // 2]

            # dim=1 is adding across the batch dim
            concat_skip = torch.cat((skip, x), dim=1)

            # idx + 1 since we added the doubleconv block after each convtranspose operation
            x = self.upsample[idx + 1](concat_skip)

        return self.finalConv(x)


def test():
    x = torch.randn((64, 1, 28, 28))
    model = UNET(in_channels=1, out_channels=1, features=[64, 128])
    preds = model(x)

    print(preds.shape)
    print(x.shape)

    assert x.shape == preds.shape

    print("All good the shapes are equal ✅")
    print(f"Image shape: {x.shape}")
    print(f"Prediction shape: {preds.shape}")


if __name__ == "__main__":
    test()
