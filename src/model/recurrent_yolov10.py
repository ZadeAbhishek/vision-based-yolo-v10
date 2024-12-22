import torch
import torch.nn as nn
from .dualAssignmentHead import DualAssignmentHead
from .partialSelfAttention import PartialSelfAttention
from .largeKernelConv import LargeKernelConv
from .c2f_block import C2fBlock
from .conv_lstm import ConvLSTMBlock

class RecurrentYOLOv10(nn.Module):
    def __init__(self, input_channels, num_classes, hidden_dim=64):
        super(RecurrentYOLOv10, self).__init__()
        self.downsample1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.large_kernel_block1 = LargeKernelConv(32, 64)

        self.downsample2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.downsample3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # Initialize the ConvLSTMBlock with proper arguments
        self.recurrent_block = ConvLSTMBlock(in_channels=64, hidden_channels=hidden_dim)

        # C2fBlock to refine features after recurrence
        self.c2f = C2fBlock(in_channels=hidden_dim, out_channels=64, bottleneck_count=2)

        self.psa = PartialSelfAttention(64)
        self.large_kernel_block2 = LargeKernelConv(64, 128)
        self.dual_head = DualAssignmentHead(128, num_classes)

    def forward(self, x):
        # x should be (B, T, C, H, W), but if it's (B, C, H, W), add a time dimension
        if x.dim() == 4:
            x = x.unsqueeze(1)  # Now x is (B, 1, C, H, W)

        B, T, C, H, W = x.shape

        # Initialize states for LSTM
        h, c = None, None

        for t in range(T):
            frame = x[:, t, :, :, :]  # (B, C, H, W)
            frame_feat = self.downsample1(frame)         # (B, 32, H/2, W/2)
            frame_feat = self.large_kernel_block1(frame_feat)
            frame_feat = self.downsample2(frame_feat)    # (B, 64, H/4, W/4)
            frame_feat = self.downsample3(frame_feat)    # (B, 64, H/8, W/8)

            # Pass through ConvLSTMBlock
            h, c = self.recurrent_block(frame_feat, h, c)

        # After processing all timesteps, h is the final hidden state
        out_feat = self.c2f(h)
        out_feat = self.psa(out_feat)
        out_feat = self.large_kernel_block2(out_feat)

        one_to_one_output, one_to_many_output = self.dual_head(out_feat)
        return one_to_one_output, one_to_many_output