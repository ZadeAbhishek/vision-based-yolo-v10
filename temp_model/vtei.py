# src/vtei.py

import numpy as np
import torch

def generate_vtei_with_rps(events, H, W, B, t0, tN, suppress_prob=0.1, pos_prob=0.5):
    """
    Generates VTEI tensor with random polarity suppression.

    Args:
        events (list of tuples): Each tuple contains (x, y, p, t).
        H (int): Height of the event frame.
        W (int): Width of the event frame.
        B (int): Number of temporal bins (channels).
        t0 (float): Start time.
        tN (float): End time.
        suppress_prob (float): Probability to suppress events.
        pos_prob (float): Probability to retain positive polarities.

    Returns:
        np.ndarray: VTEI tensor of shape (B, H, W).
    """
    # Initialize VTEI tensor
    vtei = np.zeros((B, H, W), dtype=np.float32)

    # Define time bins
    bin_size = (tN - t0) / B
    bin_edges = [t0 + i * bin_size for i in range(B + 1)]

    for event in events:
        x, y, p, t = event
        if x < 0 or x >= W or y < 0 or y >= H:
            continue  # Skip out-of-bounds events

        # Random suppression
        if np.random.rand() < suppress_prob:
            continue

        # Polarity-based suppression
        if p == 1 and np.random.rand() > pos_prob:
            continue

        # Assign to the appropriate bin
        for b in range(B):
            if bin_edges[b] <= t < bin_edges[b + 1]:
                vtei[b, y, x] += 1  # Increment event count
                break
        else:
            # If t == tN, assign to the last bin
            vtei[-1, y, x] += 1

    return vtei