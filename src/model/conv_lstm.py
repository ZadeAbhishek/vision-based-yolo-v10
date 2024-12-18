import torch
import torch.nn as nn

class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ConvLSTMBlock, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv_xh = nn.Conv2d(
            in_channels + hidden_channels, 
            4 * hidden_channels, 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(4 * hidden_channels)
        # Add a projection layer to ensure consistent output channels
        self.project_out = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)

    def forward(self, x, h=None, c=None):
        if h is None or c is None:
            h = torch.zeros(x.size(0), self.hidden_channels, x.size(2), x.size(3), device=x.device)
            c = torch.zeros_like(h)

        combined = torch.cat([x, h], dim=1)  # Combine input and hidden state
        gates = self.bn(self.conv_xh(combined))
        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)

        # Update cell and hidden states
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        # Ensure output has consistent channels
        h_next = self.project_out(h_next)

        return h_next, c_next