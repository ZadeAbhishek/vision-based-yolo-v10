import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from vtei import generate_vtei_with_rps  # Ensure this module is available in your PYTHONPATH


class PEDRoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        H=256,
        W=256,
        C_in=5,
        T=5,
        transform=None,
        suppress_prob=0.1,
        pos_prob=0.5
    ):
        self.event_dir = os.path.join(data_dir, "numpy", split)
        self.label_dir = os.path.join(data_dir, "yolo", split)
        self.split_file = os.path.join(data_dir, f"{split}.txt")

        self.H, self.W, self.C_in, self.T = H, W, C_in, T
        self.transform = transform
        self.suppress_prob = suppress_prob
        self.pos_prob = pos_prob

        if not os.path.isdir(self.event_dir):
            raise FileNotFoundError(f"Event directory {self.event_dir} does not exist.")
        
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label directory {self.label_dir} does not exist.")

        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file {self.split_file} does not exist.")
        
        with open(self.split_file, "r") as f:
            self.samples = [line.strip() for line in f if line.strip()]
        
        if not self.samples:
            raise ValueError(f"No samples found in split file {self.split_file}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]

        # Load event data
        event_path = os.path.join(self.event_dir, f"{sample_name}.npy")
        events = np.load(event_path)
        t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        events = list(zip(x.astype(int), y.astype(int), p.astype(int), t))

        # Generate VTEI tensor
        vtei = generate_vtei_with_rps(events, self.H, self.W, self.C_in, t0=t.min(), tN=t.max())
        vtei = torch.tensor(vtei, dtype=torch.float32)

        # Combine x and y into separate channels to preserve spatial information
        combined_vtei = torch.zeros((4, self.H, self.W), dtype=torch.float32)
        combined_vtei[0] = vtei[0]  # Use x (spatial information) as channel 1
        combined_vtei[1] = vtei[1]  # Use y (spatial information) as channel 2
        combined_vtei[2] = vtei[2]  # Use t (time) as channel 3
        combined_vtei[3] = vtei[3]  # Use i (intensity) as channel 4

        # Debugging: Print shape before padding
        print(f"Shape before padding for {sample_name}: {combined_vtei.shape}")

        # Pad the tensor to ensure it has exactly T temporal frames
        combined_vtei = self.pad_tensor(combined_vtei)

        # Debugging: print tensor shape after padding
        print(f"Shape after padding for {sample_name}: {combined_vtei.shape}")

        # Load YOLO-style labels
        label_path = os.path.join(self.label_dir, f"{sample_name}.txt")
        labels = self._load_yolo_labels(label_path)

        if self.transform:
            combined_vtei = self.transform(combined_vtei)

        return combined_vtei, torch.tensor(labels, dtype=torch.float32)

    def pad_tensor(self, tensor):
        """
        Pads the tensor to ensure it has exactly T temporal frames, 
        and ensures the spatial dimensions (height and width) are consistent.
        """
        current_t, current_h, current_w = tensor.shape
        
        # Calculate padding for each dimension
        padding_t = self.T - current_t if current_t < self.T else 0
        padding_h = self.H - current_h if current_h < self.H else 0
        padding_w = self.W - current_w if current_w < self.W else 0

        padded_tensor = F.pad(
            tensor.unsqueeze(0),  # Add a batch dimension for padding
            (0, padding_w, 0, padding_h, 0, padding_t),  # Pad dimensions
            "constant", 
            0
        ).squeeze(0)  # Remove the batch dimension after padding

        return padded_tensor

    def _load_yolo_labels(self, label_path):
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    vals = list(map(float, line.strip().split()))
                    if len(vals) == 5:
                        boxes.append(vals)
        return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((1, 5), dtype=torch.float32)
