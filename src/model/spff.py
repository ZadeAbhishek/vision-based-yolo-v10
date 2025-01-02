import torch.nn as nn

class SPFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Simplified Spatial Pyramid Feature Fusion (SPFF) block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(SPFF, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))