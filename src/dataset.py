import os
import numpy as np
import torch
from torch.utils.data import Dataset
from vtei import generate_vtei_with_rps  # Ensure this is in your PYTHONPATH or same directory

class PEDRoDataset(Dataset):
    """
    A PyTorch Dataset to load event data from .npy files and corresponding YOLO labels.
    Assumes the following directory structure:
        data_dir/numpy/<split>/*.npy  (event data)
        data_dir/yolo/<split>/*.txt   (YOLO labels)
        data_dir/<split>.txt          (list of sample names)
    Args:
        data_dir (str): Path to the root PEDRo dataset directory.
        split (str): Which subset to load: 'train', 'val', or 'test'.
        H (int): Height of the event frame (e.g. 260).
        W (int): Width of the event frame (e.g. 346).
        B (int): Number of temporal bins (channels) for the VTEI (e.g. 5).
        transform (callable, optional): Optional transform to apply to (vtei, labels).
        suppress_prob (float): Probability used by `generate_vtei_with_rps`.
        pos_prob (float): Probability used by `generate_vtei_with_rps`.
    """

    def __init__(
        self,
        data_dir,
        split="train",
        H=260,
        W=346,
        B=5,
        transform=None,
        suppress_prob=0.1,
        pos_prob=0.5
    ):
        self.event_dir = os.path.join(data_dir, "numpy", split)
        self.label_dir = os.path.join(data_dir, "yolo", split)
        self.split_file = os.path.join(data_dir, f"{split}.txt")

        self.H, self.W, self.B = H, W, B
        self.transform = transform
        self.suppress_prob = suppress_prob
        self.pos_prob = pos_prob

        # Read the list of sample names from <split>.txt
        with open(self.split_file, "r") as f:
            self.samples = [line.strip() for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            vtei (torch.Tensor): Shape [B, H, W].
            labels (list of list[float]): Each inner list is
                [class_id, x_center, y_center, width, height].
        """
        sample_name = self.samples[idx]

        # Load event data (shape: (N, 4) => [t, x, y, p])
        event_path = os.path.join(self.event_dir, f"{sample_name}.npy")
        events = np.load(event_path)  # shape (N,4)

        # Separate each column
        t = events[:, 0]
        x = events[:, 1].astype(int)
        y = events[:, 2].astype(int)
        p = events[:, 3]

        # Convert polarity if needed.
        # If your data is strictly {0,1}, convert 0->-1:
        # polarity = np.where(p == 1, 1, -1)
        #
        # If your data is already {+1,-1}, you can just cast:
        polarity = p.astype(int)

        # Determine global min & max timestamps
        t0, tN = t.min(), t.max()

        # Combine into a list of (x, y, polarity, time)
        event_list = [(x[i], y[i], polarity[i], t[i]) for i in range(len(t))]

        # Generate a VTEI of shape (B, H, W)
        vtei = generate_vtei_with_rps(
               events=event_list,
               H=self.H,
               W=self.W,
               B=self.B,
               t0=t0,
               tN=tN,
               suppress_prob=self.suppress_prob,
               pos_prob=self.pos_prob
)

        # Load YOLO labels: [class, x_center, y_center, w, h]
        label_path = os.path.join(self.label_dir, f"{sample_name}.txt")
        labels = self._load_yolo_labels(label_path)

        # Optional transform
        if self.transform:
            vtei, labels = self.transform(vtei, labels)

        return vtei, labels

    def _load_yolo_labels(self, label_path):
        """
        Loads YOLO format labels from a .txt file, one box per line:
            class_id x_center y_center width height
        Returns a list of bounding boxes, each box is a list of floats.
        """
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    # Example: "0 0.55 0.45 0.20 0.10"
                    vals = line.strip().split()
                    if len(vals) == 5:  # Expect 5 values
                        boxes.append([float(val) for val in vals])
                    else:
                        # If there's an unexpected format, you can handle or ignore it
                        print(f"Warning: Label line in {label_path} has {len(vals)} values, expected 5.")
        return boxes