import torch
import torch.nn as nn
import torch.nn.functional as F
from .dualAssignmentHead import DualAssignmentHead
from .partialSelfAttention import PartialSelfAttention
from .largeKernelConv import LargeKernelConv
from .c2f_block import C2fBlock
from .conv_lstm import ConvLSTMBlock
from .spff import SPFF


def dynamic_channel_resize(tensor, target_channels):
    """
    Adjusts the input tensor's channel dimension to match the target using a 1x1 convolution.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C_in, H, W).
        target_channels (int): Desired number of channels.

    Returns:
        torch.Tensor: Resized tensor with the target number of channels.
    """
    current_channels = tensor.size(1)
    if current_channels != target_channels:
        conv = nn.Conv2d(
            in_channels=current_channels,
            out_channels=target_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        ).to(tensor.device)
        tensor = conv(tensor)
    return tensor


class RecurrentYOLOv10(nn.Module):
    def __init__(self, input_channels, num_classes, num_bbox_params=4, hidden_dim=32):
        """
        YOLOv8-style recurrent model with PANet and enhanced feature fusion.

        Args:
            input_channels (int): Number of input channels.
            num_classes (int): Number of object classes.
            num_bbox_params (int): Number of bounding box parameters.
            hidden_dim (int): Hidden dimension for recurrent blocks.
        """
        super(RecurrentYOLOv10, self).__init__()
        self.num_classes = num_classes
        self.num_bbox_params = num_bbox_params

        # Backbone
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.c2f_block1 = C2fBlock(in_channels=32, out_channels=32, bottlenecks=1, residual=True)
        self.lstm1 = ConvLSTMBlock(32, hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, 64, kernel_size=3, padding=1, stride=2)
        self.c2f_block2 = C2fBlock(64, 64, bottlenecks=2, residual=True)
        self.lstm2 = ConvLSTMBlock(64, hidden_dim)

        self.conv4 = nn.Conv2d(hidden_dim, 128, kernel_size=3, padding=1, stride=2)
        self.c2f_block3 = C2fBlock(128, 128, bottlenecks=2, residual=True)
        self.lstm3 = ConvLSTMBlock(128, hidden_dim)

        self.conv5 = nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1, stride=2)
        self.c2f_block4 = C2fBlock(256, 256, bottlenecks=1, residual=True)

        # PANet
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.spff1 = SPFF(256, 128)
        self.c2f_block5 = C2fBlock(128 + 128, 128, bottlenecks=1, residual=True)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.spff2 = SPFF(128, 64)
        self.c2f_block6 = C2fBlock(64 + 64, 64, bottlenecks=1, residual=True)

        self.upsample3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.spff3 = SPFF(64, 32)
        self.c2f_block7 = C2fBlock(32 + 32, 32, bottlenecks=1, residual=True)

        # Detection Head
        self.dual_head = DualAssignmentHead(32, num_classes, num_bbox_params)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)

        B, T, C, H, W = x.shape
        h1, c1, h2, c2, h3, c3 = None, None, None, None, None, None

        for t in range(T):
            frame = x[:, t, :, :, :]
            x1 = self.conv1(frame)
            x2 = self.conv2(x1)
            x3 = self.c2f_block1(x2)
            h1, c1 = self.lstm1(x3, h1, c1)

            x4 = self.conv3(h1)
            x5 = self.c2f_block2(x4)
            h2, c2 = self.lstm2(x5, h2, c2)

            x6 = self.conv4(h2)
            x7 = self.c2f_block3(x6)
            h3, c3 = self.lstm3(x7, h3, c3)

            x8 = self.conv5(h3)
            x9 = self.c2f_block4(x8)

        p1 = self.upsample1(x9)
        h3_resized = dynamic_channel_resize(h3, p1.size(1))
        p1 = torch.cat([self.spff1(p1), h3_resized], dim=1)
        p1 = dynamic_channel_resize(p1, target_channels=256)
        p1 = self.c2f_block5(p1)

        p2 = self.upsample2(p1)
        h2_resized = dynamic_channel_resize(h2, p2.size(1))
        p2 = torch.cat([self.spff2(p2), h2_resized], dim=1)
        p2 = dynamic_channel_resize(p2, target_channels=128)
        p2 = self.c2f_block6(p2)

        p3 = self.upsample3(p2)
        h1_resized = dynamic_channel_resize(h1, p3.size(1))
        p3 = torch.cat([self.spff3(p3), h1_resized], dim=1)
        p3 = dynamic_channel_resize(p3, target_channels=64)
        p3 = self.c2f_block7(p3)

        one_to_one_output, one_to_many_output = self.dual_head(p3)
        return one_to_one_output, one_to_many_output