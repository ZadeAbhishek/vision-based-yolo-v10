import os
import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import logging

# Configure logging for the dataset module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class PEDRoDataset(Dataset):
    def __init__(self, data_dir, split="train", H=260, W=346, B=5, transform=None, limit=None):
        """
        Args:
            data_dir (str): Base directory containing the dataset.
            split (str): One of 'train', 'val', or 'test'.
            H (int): Image height.
            W (int): Image width.
            B (int): Number of temporal bins (channels).
            transform (callable, optional): Optional transform to be applied on a sample.
            limit (int, optional): Maximum number of samples to load.
        """
        self.event_dir = os.path.join(data_dir, "numpy", split)
        self.label_dir_yolo = os.path.join(data_dir, "yolo", split)
        self.label_dir_xml = os.path.join(data_dir, "xml", split)
        self.split_file = os.path.join(data_dir, f"{split}.txt")
        self.H, self.W, self.B = H, W, B
        self.transform = transform

        # Verify existence of split file
        if not os.path.exists(self.split_file):
            logging.error(f"Split file not found: {self.split_file}")
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        # Load sample names from split file
        with open(self.split_file, "r") as f:
            self.samples = [line.strip() for line in f.readlines()]

        # Apply dataset size limit if specified
        if limit:
            if limit > len(self.samples):
                logging.warning(f"Requested limit {limit} exceeds available samples {len(self.samples)}. Using full dataset.")
            else:
                self.samples = self.samples[:limit]
                logging.info(f"Loaded {len(self.samples)} samples for split '{split}' with a limit of {limit}.")

        else:
            logging.info(f"Loaded {len(self.samples)} samples for split '{split}'.")

        # Extract unique categories from the dataset
        self.object_categories = self._extract_categories()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        event_path = os.path.join(self.event_dir, f"{sample_name}.npy")

        # Load event data
        if not os.path.exists(event_path):
            logging.error(f"Event file not found: {event_path}")
            raise FileNotFoundError(f"Event file not found: {event_path}")
        events = np.load(event_path)

        # Generate VTEI from events
        vtei = self._generate_vtei(events)

        # Load YOLO-style labels (if available)
        label_path_yolo = os.path.join(self.label_dir_yolo, f"{sample_name}.txt")
        labels_yolo = self._load_yolo_labels(label_path_yolo)

        # Load Pascal-VOC XML labels (if available)
        label_path_xml = os.path.join(self.label_dir_xml, f"{sample_name}.xml")
        labels_xml = self._load_pascal_voc_labels(label_path_xml)

        # Apply transformations (if any)
        if self.transform:
            vtei, labels_yolo, labels_xml = self.transform(vtei, labels_yolo, labels_xml)

        return torch.tensor(vtei, dtype=torch.float32), {"yolo": labels_yolo, "xml": labels_xml}

    def _generate_vtei(self, events):
        """
        Converts event-based data to a Volume of Ternary Event Images (VTEI).
        Args:
            events (np.ndarray): Array of events with shape [N, 4] where each event is [t, x, y, p].
        Returns:
            VTEI (np.ndarray): Generated VTEI with shape [B, H, W].
        """
        t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        VTEI = np.zeros((self.B, self.H, self.W), dtype=np.int8)

        t_min, t_max = t.min(), t.max()
        for i in range(len(t)):
            # Normalize time to [0, B-1]
            bin_idx = int((t[i] - t_min) / (t_max - t_min) * self.B)
            bin_idx = min(bin_idx, self.B - 1)
            if 0 <= x[i] < self.W and 0 <= y[i] < self.H:
                VTEI[bin_idx, y[i], x[i]] = 1 if p[i] == 1 else -1
            else:
                logging.debug(f"Event out of bounds: x={x[i]}, y={y[i]}")
        return VTEI

    def _load_yolo_labels(self, label_path):
        """
        Loads YOLO-style labels.
        Format: [class_id, x_center, y_center, width, height]
        Args:
            label_path (str): Path to the YOLO label file.
        Returns:
            boxes (list): List of YOLO label dictionaries.
        """
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    tokens = line.strip().split()
                    if len(tokens) != 5:
                        logging.warning(f"Invalid YOLO label format in {label_path}: {line.strip()}")
                        continue
                    class_id, x_center, y_center, width, height = tokens
                    boxes.append({
                        "category_id": int(class_id),
                        "bbox": [float(x_center), float(y_center), float(width), float(height)]
                    })
        else:
            logging.debug(f"YOLO label file not found: {label_path}")
        return boxes

    def _load_pascal_voc_labels(self, label_path):
        """
        Parses PASCAL-VOC XML labels into a human-readable format.
        Args:
            label_path (str): Path to the Pascal-VOC XML label file.
        Returns:
            boxes (list): List of Pascal-VOC label dictionaries.
        """
        boxes = []
        if os.path.exists(label_path):
            try:
                tree = ET.parse(label_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    name = obj.find("name").text
                    bndbox = obj.find("bndbox")
                    xmin = int(float(bndbox.find("xmin").text))
                    ymin = int(float(bndbox.find("ymin").text))
                    xmax = int(float(bndbox.find("xmax").text))
                    ymax = int(float(bndbox.find("ymax").text))
                    boxes.append({
                        "name": name,
                        "bbox": [xmin, ymin, xmax, ymax]
                    })
            except ET.ParseError as e:
                logging.error(f"Error parsing XML file {label_path}: {e}")
        else:
            logging.debug(f"Pascal-VOC XML label file not found: {label_path}")
        return boxes

    def _extract_categories(self):
        """
        Extracts unique object categories from YOLO and Pascal-VOC XML labels.
        Returns:
            categories (list): Sorted list of unique categories.
        """
        categories = set()

        # Extract from YOLO labels
        for split in ['train', 'val', 'test']:
            yolo_dir = os.path.join(self.label_dir_yolo, split)
            if os.path.exists(yolo_dir):
                for label_file in os.listdir(yolo_dir):
                    if label_file.endswith(".txt"):
                        label_path = os.path.join(yolo_dir, label_file)
                        with open(label_path, "r") as f:
                            for line in f.readlines():
                                tokens = line.strip().split()
                                if len(tokens) >= 1:
                                    try:
                                        class_id = int(tokens[0])
                                        categories.add(str(class_id))
                                    except ValueError:
                                        logging.warning(f"Invalid class ID in {label_path}: {tokens[0]}")

        # Extract from Pascal-VOC XML labels
        for split in ['train', 'val', 'test']:
            xml_dir = os.path.join(self.label_dir_xml, split)
            if os.path.exists(xml_dir):
                for label_file in os.listdir(xml_dir):
                    if label_file.endswith(".xml"):
                        label_path = os.path.join(xml_dir, label_file)
                        try:
                            tree = ET.parse(label_path)
                            root = tree.getroot()
                            for obj in root.findall("object"):
                                name = obj.find("name").text
                                categories.add(name)
                        except ET.ParseError as e:
                            logging.error(f"Error parsing XML file {label_path}: {e}")

        return sorted(categories)