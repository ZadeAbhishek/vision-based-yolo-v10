import os
import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET


class PEDRoDataset(Dataset):
    def __init__(self, data_dir, split="train", H=260, W=346, B=5, transform=None):
        self.event_dir = os.path.join(data_dir, "numpy", split)
        self.label_dir_yolo = os.path.join(data_dir, "yolo", split)
        self.label_dir_xml = os.path.join(data_dir, "xml", split)
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

        # Generate VTEI from events
        vtei = self._generate_vtei(events)

        # Load YOLO-style labels (if needed)
        label_path_yolo = os.path.join(self.label_dir_yolo, f"{sample_name}.txt")
        labels_yolo = self._load_yolo_labels(label_path_yolo)

        # Load Pascal-VOC XML labels
        label_path_xml = os.path.join(self.label_dir_xml, f"{sample_name}.xml")
        labels_xml = self._load_pascal_voc_labels(label_path_xml)

        # Apply transformations (if any)
        if self.transform:
            vtei, labels_yolo, labels_xml = self.transform(vtei, labels_yolo, labels_xml)

        return torch.tensor(vtei, dtype=torch.float32), labels_yolo, labels_xml

    def _generate_vtei(self, events):
        """
        Converts event-based data to a Volume of Ternary Event Images (VTEI).
        """
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
        """
        Loads YOLO-style labels.
        Format: [class_id, x_center, y_center, width, height]
        """
        boxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                boxes.append([float(x) for x in line.strip().split()])
        return boxes

    def _load_pascal_voc_labels(self, label_path):
        """
        Parses PASCAL-VOC XML labels into a human-readable format.
        """
        boxes = []
        tree = ET.parse(label_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            boxes.append({
                "class": class_name,
                "bbox": [xmin, ymin, xmax, ymax]
            })
        return boxes