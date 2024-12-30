import numpy as np
import torch

def generate_vtei_with_rps(events, H, W, B, t0, tN, suppress_prob=0.1, pos_prob=0.5):
    """
    Generates a VTEI (Voxel of Time-Surface Event Image) with optional random polarity suppression (RPS).

    Args:
        events (list of tuples): List of (x, y, polarity, t).
            - x (int): x-coordinate in [0, W-1].
            - y (int): y-coordinate in [0, H-1].
            - polarity (int): event polarity (+1 or -1).
            - t (float): event timestamp.
        H (int): Image height.
        W (int): Image width.
        B (int): Number of temporal bins (channels).
        t0 (float): Earliest timestamp (min of t).
        tN (float): Latest timestamp (max of t).
        suppress_prob (float): Probability of suppressing either positive or negative polarities.
        pos_prob (float): Probability of suppressing positive vs negative if suppression is triggered.

    Returns:
        torch.Tensor:
            A float32 tensor of shape (B, H, W). Each bin corresponds to a time-slice,
            and each pixel stores +1/-1 based on events.
            If random suppression occurs, either all positive or all negative polarities
            are zeroed out.
    """
    # Initialize the VTEI volume
    VTEI = np.zeros((B, H, W), dtype=np.int8)

    # Safeguard against divide-by-zero if t0 == tN
    time_range = tN - t0
    if time_range == 0:
        # All events happen at the same timestamp; assign them to bin 0
        for (x, y, polarity, t) in events:
            if 0 <= x < W and 0 <= y < H:
                VTEI[0, y, x] = polarity
    else:
        # Distribute events across the B bins based on their timestamps
        for (x, y, polarity, t) in events:
            if 0 <= x < W and 0 <= y < H:
                bin_idx = int((t - t0) / time_range * B)
                bin_idx = min(max(bin_idx, 0), B - 1)
                VTEI[bin_idx, y, x] = polarity

    # Random suppression of either positive or negative events
    r1, r2 = np.random.rand(), np.random.rand()
    if r1 < suppress_prob:
        if r2 < pos_prob:
            # Suppress positives
            VTEI[VTEI > 0] = 0
        else:
            # Suppress negatives
            VTEI[VTEI < 0] = 0

    # Convert to float32 tensor before returning
    return torch.tensor(VTEI, dtype=torch.float32)


# if __name__ == "__main__":
#     # Example events: [x, y, polarity, t]
#     events = [
#         (10, 20, 1, 0.1),
#         (15, 25, -1, 0.15),
#         (10, 20, 1, 0.2),
#         (30, 40, 1, 0.3),
#         (30, 40, -1, 0.35),
#     ]

#     H, W = 64, 64  # Spatial dimensions
#     B = 8          # Temporal bins
#     t0, tN = 0.0, 0.4  # Timestamps range

#     # Generate VTEI
#     vtei = generate_vtei_with_rps(events, H, W, B, t0, tN, suppress_prob=0.2, pos_prob=0.5)
#     print("Generated VTEI Shape:", vtei.shape)
#     print("VTEI Tensor:\n", vtei)