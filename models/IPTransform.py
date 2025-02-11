import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F

from models.Conv import PointwiseConv


class IPTConv(nn.Module):
    def __init__(self, c1, c2, stride=1, scale=2):
        super().__init__()
        self.scale = scale
        self.stride = stride
        self.PointConv = PointwiseConv(c1=c1 * self.scale**2, c2=c2)

    def forward(self, x):
        x = self.IPTrans(x)
        x = self.PointConv(x)

        return x

    def IPTrans(self, x):
        bs, c, h, w = x.shape
        assert h % self.scale == 0 and w % self.scale == 0, "Height and width must be divisible by scale"

        x = x.view(bs, c, h // self.scale, self.scale, w // self.scale, self.scale)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(bs, c * (self.scale ** 2), h // self.scale, w // self.scale)

        return x


# if __name__ == '__main__':
#     batch_size = 1
#     channels = 3
#     height = 4
#     width = 4
#
#     scale = 2
#
#     input_tensor = torch.arange(0, batch_size * channels * height * width).view(batch_size, channels, height,
#                                                                                 width).float()
#
#     print(input_tensor)
#
#     IPT = IPTConv(c1=channels, c2=channels*scale**2, scale=scale)
#     output_tensor = IPT(input_tensor)
#
#     print("Input Tensor Shape:", input_tensor.shape)
#     print("Output Tensor Shape:", output_tensor.shape)
#     print("Output Tensor: \n", output_tensor)
