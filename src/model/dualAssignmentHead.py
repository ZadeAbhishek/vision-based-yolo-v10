# model/dualAssignmentHead.py

import torch.nn as nn

class DualAssignmentHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_bbox_params=4):
        """
        Initializes the DualAssignmentHead.

        Args:
            in_channels (int): Number of input channels from the previous layer.
            num_classes (int): Number of object classes.
            num_bbox_params (int): Number of bounding box parameters (e.g., 4 for x, y, w, h).
        """
        super(DualAssignmentHead, self).__init__()
        self.num_classes = num_classes
        self.num_bbox_params = num_bbox_params

        # Example convolutional layers (adjust based on your architecture)
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        # Final convolution to predict objectness, class scores, and bbox parameters
        self.final_conv = nn.Conv2d(
            256, 
            1 + self.num_classes + self.num_bbox_params,  # Objectness + Class Scores + BBox Params
            kernel_size=1
        )

    def forward(self, x):
        """
        Forward pass of the DualAssignmentHead.

        Args:
            x (torch.Tensor): Input feature map of shape [Batch, in_channels, H, W].

        Returns:
            tuple: 
                - one_to_one_output (torch.Tensor): Objectness score map.
                - one_to_many_output (torch.Tensor): Class scores and bbox parameters.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        detection_output = self.final_conv(x)
        
        # Split into one_to_one and one_to_many outputs
        one_to_one_output = detection_output[:, :1, :, :]  # Objectness
        one_to_many_output = detection_output[:, 1:, :, :]  # Class Scores + BBox Params
        return one_to_one_output, one_to_many_output