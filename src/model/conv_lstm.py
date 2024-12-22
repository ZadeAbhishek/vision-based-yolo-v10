import torch
import torch.nn as nn

class ConvLSTMBlock(nn.Module):
    """
    A single ConvLSTM block. This processes spatial data (H, W) over time steps, 
    maintaining a hidden (h) and cell (c) state across frames.

    Arguments:
        in_channels (int): Number of input channels at each time step.
        hidden_channels (int): Number of hidden channels in the LSTM cell.
    """
    def __init__(self, in_channels, hidden_channels):
        super(ConvLSTMBlock, self).__init__()
        self.hidden_channels = hidden_channels

        # Conv that produces all gate values: i, f, o, g
        self.conv_xh = nn.Conv2d(
            in_channels + hidden_channels, 
            4 * hidden_channels, 
            kernel_size=3, 
            padding=1, 
            bias=True
        )
        self.project_out = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)

    def forward(self, x, h=None, c=None):
        # Initialize hidden and cell states if not provided
        if h is None or c is None:
            h = torch.zeros(
                x.size(0), self.hidden_channels, x.size(2), x.size(3), device=x.device, dtype=x.dtype
            )
            c = torch.zeros_like(h)

        combined = torch.cat([x, h], dim=1)
        gates = self.conv_xh(combined)
        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        h_next = self.project_out(h_next)

        return h_next, c_next