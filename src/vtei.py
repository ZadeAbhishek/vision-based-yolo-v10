import numpy as np
import torch

def generate_vtei_with_rps(events, H, W, B, t0, tN, suppress_prob=0.1, pos_prob=0.5):
    """
    Converts event streams into VTEI tensors with RPS.

    Args:
        events: List of tuples [(x, y, polarity, timestamp)]
        H, W: Image dimensions (height, width)
        B: Number of temporal bins
        t0, tN: Start and end timestamps of event stream
        suppress_prob: Probability to suppress polarities
        pos_prob: Probability to suppress positive polarities

    Returns:
        VTEI tensor of shape (1, B, H, W)
    """
    VTEI = np.zeros((B, H, W), dtype=np.int8)

    for x, y, polarity, t in events:
        bin_idx = int((t - t0) / (tN - t0) * B)
        bin_idx = min(max(bin_idx, 0), B - 1)
        VTEI[bin_idx, y, x] = polarity

    r1, r2 = np.random.rand(), np.random.rand()
    if r1 < suppress_prob:
        if r2 < pos_prob:
            VTEI[VTEI > 0] = 0  # Suppress positive polarities
        else:
            VTEI[VTEI < 0] = 0  # Suppress negative polarities

    return torch.tensor(VTEI, dtype=torch.float32).unsqueeze(0)