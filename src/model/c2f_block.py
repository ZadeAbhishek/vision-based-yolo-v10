import torch
import torch.nn as nn

class C2fBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_count=2):
        super(C2fBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(bottleneck_count)
        ])

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.use_residual = (in_channels == out_channels)

    def forward(self, x):
        identity = x
        x = self.act(self.bn1(self.conv1(x)))

        bottleneck_output = 0
        for b in self.bottlenecks:
            bottleneck_output += b(x)

        out = self.act(self.bn2(self.conv2(bottleneck_output)))

        if self.use_residual:
            out = out + identity
        return out