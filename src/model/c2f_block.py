import torch
import torch.nn as nn
import torch.nn.functional as F

# Replace all in-place operations
class C2fBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottlenecks=2, residual=True):
        super().__init__()
        self.residual = residual
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)  # Ensure inplace=False
        )
        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False)  # Ensure inplace=False
            ) for _ in range(bottlenecks)
        ])

    def forward(self, x):
        y = self.entry(x)
        identity = y if self.residual else None
        for block in self.bottlenecks:
            y = block(y)
        return y + identity if identity is not None else y