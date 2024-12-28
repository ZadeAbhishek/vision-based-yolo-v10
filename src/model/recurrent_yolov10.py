# model/recurrent_yolov10.py

import torch.nn as nn
from .dualAssignmentHead import DualAssignmentHead
from .partialSelfAttention import PartialSelfAttention
from .largeKernelConv import LargeKernelConv
from .c2f_block import C2fBlock
from .conv_lstm import ConvLSTMBlock

class RecurrentYOLOv10(nn.Module):
    def __init__(self, input_channels, num_classes, num_bbox_params=4, hidden_dim=64):
        """
        Initializes the RecurrentYOLOv10 model.

        Args:
            input_channels (int): Number of input channels.
            num_classes (int): Number of object classes.
            num_bbox_params (int): Number of bounding box parameters.
            hidden_dim (int): Hidden dimension for recurrent blocks.
        """
        super(RecurrentYOLOv10, self).__init__()
        self.num_classes = num_classes
        self.num_bbox_params = num_bbox_params
        self.hidden_dim = hidden_dim

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
        self.dual_head = DualAssignmentHead(128, num_classes, num_bbox_params)

    def forward(self, x):
        # x should be (B, T, C_in, H, W), but if it's (B, C, H, W), add a time dimension
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
    
    
'''Input: (B, T, C, H, W)
   |
   v
Temporal Dimension Handling:
If 4D (B, C, H, W), add time dimension (T=1)
   |
   v
For each frame (T steps):
   |
   v
+---------------------+
| Convolutional Layers|
+---------------------+
   |
   v
   [Downsample1] (Conv2D: C -> 32, H/W -> H/2, W/2)
   |
   v
   [LargeKernelConv1] (32 -> 64)
   |
   v
   [Downsample2] (Conv2D: 64 -> 64, H/2 -> H/4, W/2 -> W/4)
   |
   v
   [Downsample3] (Conv2D: 64 -> 64, H/4 -> H/8, W/4 -> W/8)
   |
   v
+-----------------------+
| ConvLSTM Block        |
| Temporal Feature Fusion|
+-----------------------+
   |     Hidden State (h), Cell State (c)
   v
End of temporal loop:
   |
   v
Final hidden state (h)
   |
   v
+--------------------+
| Feature Refinement |
+--------------------+
   |
   v
   [C2fBlock] (Hidden Dim -> 64)
   |
   v
   [PartialSelfAttention] (Focus on key regions)
   |
   v
   [LargeKernelConv2] (64 -> 128)
   |
   v
+-----------------------+
| DualAssignmentHead    |
| (One-to-One Output,   |
|  One-to-Many Output)  |
+-----------------------+
   |
   v
Outputs:
1. One-to-One Output (e.g., Global Classification)
2. One-to-Many Output (e.g., Object Detection) '''