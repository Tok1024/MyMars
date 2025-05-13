import torch.nn as nn
from model.base.components import Conv, SPPF
from model.swinbackbone.components import SwinC2f


class SwinBackbone(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.imageChannel = 3
        self.kernelSize = 3
        self.stride = 2

        # Backbone layers
        self.conv1 = Conv(self.imageChannel, int(64 * w), self.kernelSize, self.stride)

        self.conv2 = Conv(int(64 * w), int(128 * w), self.kernelSize, self.stride)
        self.c2f1 = SwinC2f(int(128 * w), int(128 * w), n, shortcut=True)

        self.conv3 = Conv(int(128 * w), int(256 * w), self.kernelSize, self.stride)
        self.c2f2 = SwinC2f(int(256 * w), int(256 * w), 2 * n, shortcut=True)

        self.conv4 = Conv(int(256 * w), int(512 * w), self.kernelSize, self.stride)
        self.c2f3 = SwinC2f(int(512 * w), int(512 * w), 2 * n, shortcut=True)

        self.conv5 = Conv(int(512 * w), int(512 * w * r), self.kernelSize, self.stride)
        self.c2f4 = SwinC2f(int(512 * w * r), int(512 * w * r), n, shortcut=True)
        self.sppf = SPPF(int(512 * w * r), int(512 * w * r))

    def forward(self, x):
        """
        Input shape: (B, 3, 640, 640)
        Output shape:
            feat0: (B, 128 * w, 160, 160)
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        """
        feat0 = self.c2f1(self.conv2(self.conv1(x)))
        feat1 = self.c2f2(self.conv3(feat0))
        feat2 = self.c2f3(self.conv4(feat1))
        feat3 = self.sppf(self.c2f4(self.conv5(feat2)))
        return None, feat1, feat2, feat3