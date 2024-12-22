import torch
import torch.nn as nn

class PartialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(PartialSelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        hw = h * w

        # Reshape and permute to form (b, hw, c)
        query = self.query(x).view(b, c, hw).permute(0, 2, 1)   # (b, hw, c)
        key = self.key(x).view(b, c, hw)                        # (b, c, hw)
        value = self.value(x).view(b, c, hw).permute(0, 2, 1)   # (b, hw, c)

        # Attention: (b, hw, hw)
        # Query * Key -> (b, hw, hw)
        attention = torch.bmm(query, key)  # (b, hw, c) * (b, c, hw)
        attention = self.softmax(attention / (hw))  # scale by hw to stabilize

        # (b, hw, c) = (b, hw, hw) * (b, hw, c)
        out = torch.bmm(attention, value)

        # Reshape back to (b, c, h, w)
        out = out.permute(0, 2, 1).view(b, c, h, w)

        return out + x  # Residual connection