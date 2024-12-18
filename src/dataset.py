import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PEDRoDataset(Dataset):
    def __init__(self, data_dir, split="train", H=260, W=346, B=5, transform=None):
        self.event_dir = os.path.join(data_dir, "numpy", split)
        self.label_dir = os.path.join(data_dir, "yolo", split)
        self.split_file = os.path.join(data_dir, f"{split}.txt")
        self.H, self.W, self.B = H, W, B
        self.transform = transform

        with open(self.split_file, "r") as f:
            self.samples = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        event_path = os.path.join(self.event_dir, f"{sample_name}.npy")
        events = np.load(event_path)

        vtei = self._generate_vtei(events)

        label_path = os.path.join(self.label_dir, f"{sample_name}.txt")
        labels = self._load_yolo_labels(label_path)

        if self.transform:
            vtei, labels = self.transform(vtei, labels)

        return torch.tensor(vtei, dtype=torch.float32), labels

    def _generate_vtei(self, events):
        t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        VTEI = np.zeros((self.B, self.H, self.W), dtype=np.int8)

        t_min, t_max = t.min(), t.max()
        for i in range(len(t)):
            bin_idx = int((t[i] - t_min) / (t_max - t_min) * self.B)
            bin_idx = min(bin_idx, self.B - 1)
            if 0 <= x[i] < self.W and 0 <= y[i] < self.H:
                VTEI[bin_idx, y[i], x[i]] = 1 if p[i] == 1 else -1
        return VTEI

    def _load_yolo_labels(self, label_path):
        boxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                boxes.append([float(x) for x in line.strip().split()])
        return boxes