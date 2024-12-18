import torch.nn as nn
import torch

class C2fBlock(nn.Module):
    """
    Efficient C2f Block based on Cross-Stage Partial Bottleneck design.
    """
    def __init__(self, in_channels, out_channels, bottleneck_count=2):
        super(C2fBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ) for _ in range(bottleneck_count)
        ])

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        bottleneck_output = sum([b(x) for b in self.bottlenecks])  # Sum of bottleneck outputs
        out = self.relu(self.bn2(self.conv2(bottleneck_output)))
        return out