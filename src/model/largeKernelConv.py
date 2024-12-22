import torch
import torch.nn as nn
# Large-Kernel Convolutions
class LargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=13, stride=1, padding=6):
        super(LargeKernelConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
