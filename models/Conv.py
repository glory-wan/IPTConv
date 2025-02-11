import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Many of these Conv class are from Ultralytics
    https://github.com/ultralytics/ultralytics
"""


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class PointwiseConv(nn.Module):
    """Point-wise convolution (1x1 convolution) to perform dot product between input channels."""

    def __init__(self, c1, c2, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False)  # 1x1 convolution
        self.bn = nn.BatchNorm2d(c2)  # Batch normalization
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply 1x1 convolution, batch normalization, and activation."""
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


if __name__ == "__main__":
    input_tensor = torch.randn(1, 16, 32, 32)

    model = PointwiseConv(c1=16, c2=64)

    output = model(input_tensor)
    print("Output shape:", output.shape)
