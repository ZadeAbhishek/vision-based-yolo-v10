import torch
import torch.nn as nn
from .c2f_block import C2fBlock
from .conv_lstm import ConvLSTMBlock

class RecurrentYOLOv8(nn.Module):
    def __init__(self, input_channels, num_classes, max_boxes):
        super(RecurrentYOLOv8, self).__init__()
        self.num_classes = num_classes
        self.max_boxes = max_boxes  # Keep this if needed, but it's no longer used the same way.

        self.downsample1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.c2f_block = C2fBlock(32, 64, bottleneck_count=2)
        self.conv_lstm = ConvLSTMBlock(64, 64)
        self.classification_head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.bbox_head = nn.Conv2d(64, 4, kernel_size=1)

    def forward(self, x):
        """
        x shape: [B, B(in channels), H, W]
        Output:
          class_logits: [B, num_classes, H_out, W_out]
          bbox_preds:   [B, 4, H_out, W_out]
        """
        x = self.downsample1(x)
        x = self.c2f_block(x)
        h, c = None, None
        h, c = self.conv_lstm(x, h, c)
        class_logits = self.classification_head(h) # [B, num_classes, H_out, W_out]
        bbox_preds = self.bbox_head(h)              # [B, 4, H_out, W_out]
        return class_logits, bbox_preds