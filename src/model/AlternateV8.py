import torch
import torch.nn as nn
import torch.nn.functional as F


def dynamic_resize(tensor, target_size, target_channels=None):
    """
    Dynamically resize a tensor to match the target size and optionally match channels.

    Args:
        tensor (torch.Tensor): Input tensor to resize.
        target_size (tuple): Target size as (B, C, H, W).
        target_channels (int, optional): If provided, adjusts the channels using a 1x1 convolution.

    Returns:
        torch.Tensor: Resized tensor.
    """
    current_size = tensor.size()

    # Adjust spatial dimensions (H, W) using interpolation
    if current_size[2:] != target_size[2:]:
        tensor = F.interpolate(tensor, size=target_size[2:], mode="nearest")
    
    # Adjust channel dimension (C) if necessary
    if target_channels is not None and current_size[1] != target_channels:
        conv = nn.Conv2d(
            in_channels=current_size[1],
            out_channels=target_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        ).to(tensor.device)
        tensor = conv(tensor)

    return tensor


class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv_xh = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, h_prev=None, c_prev=None):
        B, C, H, W = x.shape
        if h_prev is None:
            h_prev = torch.zeros(B, self.conv_xh.out_channels // 4, H, W, device=x.device)
            c_prev = torch.zeros(B, self.conv_xh.out_channels // 4, H, W, device=x.device)

        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv_xh(combined)
        i, f, g, o = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class C2fBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottlenecks=2, residual=True):
        super().__init__()
        self.residual = residual
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(bottlenecks)
        ])

    def forward(self, x):
        y = self.entry(x)
        identity = y if self.residual else None
        for block in self.bottlenecks:
            y = block(y)
        return y + identity if identity is not None else y


class SPFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class ReYOLOv8s(nn.Module):
    def __init__(self, in_channels=5,NUM_CLASSES=2):
        super().__init__()

        # Recurrent Backbone
        self.conv1  = nn.Conv2d(in_channels, 24, 3, padding=1)
        self.conv2  = nn.Conv2d(24, 48, 3, padding=1)
        self.c2f3   = C2fBlock(48, 48, bottlenecks=2, residual=True)
        self.lstm4  = ConvLSTMBlock(48, 48)
        self.conv5  = nn.Conv2d(48, 88, 3, padding=1)
        self.c2f6   = C2fBlock(88, 88, bottlenecks=3, residual=True)
        self.lstm7  = ConvLSTMBlock(88, 88)
        self.conv8  = nn.Conv2d(88, 176, 3, padding=1)
        self.c2f9   = C2fBlock(176, 176, bottlenecks=3, residual=True)
        self.lstm10 = ConvLSTMBlock(176, 176)
        self.conv11 = nn.Conv2d(176, 344, 3, padding=1)
        self.c2f12  = C2fBlock(344, 344, bottlenecks=2, residual=True)
        self.lstm13 = ConvLSTMBlock(344, 344)
        self.spff14 = SPFF(344, 344)

        # PANet
        self.upsample15 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f17 = C2fBlock(520, 176, bottlenecks=2, residual=False)
        self.upsample18 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f20 = C2fBlock(264, 88, bottlenecks=2, residual=False)
        self.conv21 = nn.Conv2d(88, 88, kernel_size=3, padding=1)
        self.c2f23 = C2fBlock(264, 176, bottlenecks=2, residual=False)
        self.conv24 = nn.Conv2d(176, 176, kernel_size=3, padding=1)
        self.c2f26 = C2fBlock(520, 344, bottlenecks=2, residual=False)

        # Detection Head
        self.detect27 = nn.Conv2d(88 + 176 + 344, 80, kernel_size=1)
        self.NUM_CLASSES = NUM_CLASSES

    def forward(self, x):
     # Backbone
     x1 = self.conv1(x)
     x2 = self.conv2(x1)
     x3 = self.c2f3(x2)
     x4, c4 = self.lstm4(x3)
     x5 = self.conv5(x4)
     x6 = self.c2f6(x5)
     x7, c7 = self.lstm7(x6)
     x8 = self.conv8(x7)
     x9 = self.c2f9(x8)
     x10, c10 = self.lstm10(x9)
     x11 = self.conv11(x10)
     x12 = self.c2f12(x11)
     x13, c13 = self.lstm13(x12)
     x14 = self.spff14(x13)

    # PANet
     x15 = self.upsample15(x14)
     x10_resized = dynamic_resize(x10, x15.size(), target_channels=176)
     x16 = torch.cat([x15, x10_resized], dim=1)
     x17 = self.c2f17(x16)
     x18 = self.upsample18(x17)
     x7_resized = dynamic_resize(x7, x18.size(), target_channels=88)
     x19 = torch.cat([x18, x7_resized], dim=1)
     x20 = self.c2f20(x19)
     x21 = self.conv21(x20)
     x17_resized = dynamic_resize(x17, x21.size(), target_channels=176)
     x22 = torch.cat([x21, x17_resized], dim=1)
     x23 = self.c2f23(x22)
     x24 = self.conv24(x23)
     x14_resized = dynamic_resize(x14, x24.size(), target_channels=344)
     x25 = torch.cat([x24, x14_resized], dim=1)
     x26 = self.c2f26(x25)

     # Detection Head
     x20_resized = dynamic_resize(x20, x26.size(), target_channels=88)
     x23_resized = dynamic_resize(x23, x26.size(), target_channels=176)
     detect_in = torch.cat([x20_resized, x23_resized, x26], dim=1)
     out = self.detect27(detect_in)

     # Split output into class logits and bounding box predictions
     B, C, H, W = out.shape
     class_logits = out[:, :self.NUM_CLASSES, :, :]  # First NUM_CLASSES channels for class predictions
     bbox_preds = out[:, self.NUM_CLASSES:, :, :]    # Remaining channels for bounding boxes

     return class_logits, bbox_preds


# # Test with dummy input
# if __name__ == "__main__":
#     model = ReYOLOv8s(in_channels=5)
#     dummy_input = torch.randn(2, 5, 64, 64)  # Batch=2, Channels=5, H=W=64
#     output = model(dummy_input)
#     print("Output shape:", output.shape)