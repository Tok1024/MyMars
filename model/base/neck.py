import torch
import torch.nn as nn
from model.base.components import Conv, C2f


class Neck(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """

    def __init__(self, w, r, n):
        super().__init__()
        self.kernelSize = 3
        self.stride = 2

        # Neck layers
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f1 = C2f(int(512 * w * (1 + r)), int(512 * w), n, shortcut=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f2 = C2f(int(768 * w), int(256 * w), n, shortcut=False)

        self.conv1 = Conv(int(256 * w), int(256 * w), self.kernelSize, self.stride)
        self.c2f3 = C2f(int(768 * w), int(512 * w), n, shortcut=False)
        self.conv2 = Conv(int(512 * w), int(512 * w), self.kernelSize, self.stride)
        self.c2f4 = C2f(int(512 * w * (1 + r)), int(512 * w * r), n, shortcut=False)

    def forward(self, feat1, feat2, feat3):
        """
        Input shape:
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        Output shape:
            C: (B, 512 * w, 40, 40)
            X: (B, 256 * w, 80, 80)
            Y: (B, 512 * w, 40, 40)
            Z: (B, 512 * w * r, 20, 20)
        """
        # First upsampling and processing
        C = self.c2f1(torch.cat((feat2, self.upsample1(feat3)), dim=1))

        # Second upsampling and processing
        X = self.c2f2(torch.cat((feat1, self.upsample1(C)), dim=1))

        # Downsampling and processing
        Y = self.c2f3(torch.cat((C, self.conv1(X)), dim=1))

        # Final processing
        Z = self.c2f4(torch.cat((feat3, self.conv2(Y)), dim=1))

        return C, X, Y, Z