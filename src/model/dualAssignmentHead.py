import torch
import torch.nn as nn
# Dual Assignment Head
class DualAssignmentHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DualAssignmentHead, self).__init__()
        self.one_to_one = nn.Conv2d(in_channels, 1, kernel_size=1)  # One-to-One
        self.one_to_many = nn.Conv2d(in_channels, num_classes + 4, kernel_size=1)  # One-to-Many

    def forward(self, x):
        one_to_one_output = self.one_to_one(x)  # Objectness prediction
        one_to_many_output = self.one_to_many(x)  # Class + Bounding Box prediction
        return one_to_one_output, one_to_many_output