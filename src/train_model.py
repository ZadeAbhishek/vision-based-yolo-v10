import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from model.AlternateV8 import ReYOLOv8s

# Device configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Dataset configuration
DATA_DIR = "data/PEDRo/"
BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 0.01
H, W, B = 64, 64, 5  # Image height, width, temporal bins
NUM_CLASSES = None  # Dynamically determined later


def custom_collate_fn(batch):
    """
    Custom collate function to maintain the structure of VTEIs and labels.
    """
    vteis, labels = zip(*batch)
    vteis = torch.stack(vteis)
    # Maintain label structure for batching
    batched_labels = {"yolo": [], "xml": []}
    for label in labels:
        batched_labels["yolo"].extend(label["yolo"])
        batched_labels["xml"].extend(label["xml"])
    return vteis, batched_labels

def build_targets(class_logits, bbox_preds, labels, num_classes):
    """
    Placeholder for YOLO-style target assignment.
    """
    B, _, H_out, W_out = class_logits.shape
    obj_target = torch.zeros(B, 1, H_out, W_out, device=class_logits.device)
    class_target = torch.zeros(B, num_classes, H_out, W_out, device=class_logits.device)
    bbox_target = torch.zeros(B, 4, H_out, W_out, device=class_logits.device)

    # Implement YOLO-style target assignment logic here if needed.
    return obj_target, class_target, bbox_target


class PEDRoDataset(Dataset):
    def __init__(self, data_dir, split, H=64, W=64, B=5):
        self.event_dir = os.path.join(data_dir, "numpy", split)
        self.label_dir_txt = os.path.join(data_dir, "yolo", split)
        self.label_dir_xml = os.path.join(data_dir, "xml", split)
        self.H, self.W, self.B = H, W, B
        self.samples = [f.replace(".npy", "") for f in os.listdir(self.event_dir) if f.endswith(".npy")]
        self.object_categories = self._extract_categories()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        event_path = os.path.join(self.event_dir, f"{sample_name}.npy")
        events = np.load(event_path)

        vtei = self._generate_vtei(events)

        # Load labels from both TXT and XML
        label_txt_path = os.path.join(self.label_dir_txt, f"{sample_name}.txt")
        label_xml_path = os.path.join(self.label_dir_xml, f"{sample_name}.xml")
        yolo_labels = self._load_yolo_labels(label_txt_path)
        xml_labels = self._load_xml_labels(label_xml_path)

        return torch.tensor(vtei, dtype=torch.float32), {"yolo": yolo_labels, "xml": xml_labels}

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
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    tokens = [float(x) for x in line.strip().split()]
                    category_id = int(tokens[0])
                    bbox = tokens[1:]
                    boxes.append({"category_id": category_id, "bbox": bbox})
        return boxes

    def _load_xml_labels(self, label_path):
        labels = []
        if os.path.exists(label_path):
            tree = ET.parse(label_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                name = obj.find("name").text
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                labels.append({"name": name, "bbox": [xmin, ymin, xmax, ymax]})
        return labels

    def _extract_categories(self):
        categories = set()

        # Extract from TXT labels
        if os.path.exists(self.label_dir_txt):
            for label_file in os.listdir(self.label_dir_txt):
                if label_file.endswith(".txt"):
                    with open(os.path.join(self.label_dir_txt, label_file), "r") as f:
                        for line in f.readlines():
                            category_id = int(line.split()[0])
                            categories.add(str(category_id))

        # Extract from XML labels
        if os.path.exists(self.label_dir_xml):
            for label_file in os.listdir(self.label_dir_xml):
                if label_file.endswith(".xml"):
                    tree = ET.parse(os.path.join(self.label_dir_xml, label_file))
                    root = tree.getroot()
                    for obj in root.findall("object"):
                        categories.add(obj.find("name").text)

        return sorted(categories)


def evaluate_model(model, loader, obj_loss_fn, class_loss_fn, bbox_loss_fn, device, num_classes, phase="Validation"):
    model.eval()
    total_loss, total_obj_loss, total_class_loss, total_bbox_loss = 0, 0, 0, 0
    with torch.no_grad():
        for vtei, labels in loader:
            vtei = vtei.to(device)
            class_logits, bbox_preds = model(vtei)

            obj_target, class_target, bbox_target = build_targets(class_logits, bbox_preds, labels, num_classes)

            objectness_pred = class_logits[:, 0:1, :, :]
            class_pred = class_logits[:, 1:, :, :]
            loss_obj = obj_loss_fn(objectness_pred, obj_target)

            obj_mask = (obj_target.squeeze(1) == 1)
            class_indices = class_target.argmax(dim=1)
            class_pred_obj = class_pred.permute(0, 2, 3, 1)[obj_mask]
            class_gt_obj = class_indices[obj_mask]
            loss_cls = class_loss_fn(class_pred_obj, class_gt_obj) if class_pred_obj.numel() > 0 else torch.tensor(0.0, device=device)

            bbox_pred_obj = bbox_preds.permute(0, 2, 3, 1)[obj_mask]
            bbox_gt_obj = bbox_target.permute(0, 2, 3, 1)[obj_mask]
            loss_bbox = bbox_loss_fn(bbox_pred_obj, bbox_gt_obj) if bbox_pred_obj.numel() > 0 else torch.tensor(0.0, device=device)

            loss = loss_obj + loss_cls + loss_bbox
            total_loss += loss.item()
            total_obj_loss += loss_obj.item()
            total_class_loss += loss_cls.item()
            total_bbox_loss += loss_bbox.item()

            print(f"{phase}: Detected Objects (XML): {[label['name'] for label in labels['xml']]}")
            print(f"{phase}: Detected Objects (YOLO): {[OBJECT_CATEGORIES[int(label['category_id'])] for label in labels['yolo']]}")

    print(f"{phase} Loss: {total_loss / len(loader):.4f}")
    print(f"  - Objectness Loss: {total_obj_loss / len(loader):.4f}")
    print(f"  - Class Loss: {total_class_loss / len(loader):.4f}")
    print(f"  - Bounding Box Loss: {total_bbox_loss / len(loader):.4f}")



def train_model():
    train_dataset = PEDRoDataset(data_dir=DATA_DIR, split="train", H=H, W=W, B=B)
    val_dataset = PEDRoDataset(data_dir=DATA_DIR, split="val", H=H, W=W, B=B)
    test_dataset = PEDRoDataset(data_dir=DATA_DIR, split="test", H=H, W=W, B=B)

    global OBJECT_CATEGORIES, NUM_CLASSES
    OBJECT_CATEGORIES = train_dataset.object_categories
    NUM_CLASSES = len(OBJECT_CATEGORIES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    model = ReYOLOv8s(in_channels=B).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

    obj_loss_fn = nn.BCEWithLogitsLoss()
    class_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()

    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (vtei, labels) in enumerate(train_loader):
            vtei = vtei.to(DEVICE)
            class_logits, bbox_preds = model(vtei)

            obj_target, class_target, bbox_target = build_targets(class_logits, bbox_preds, labels, NUM_CLASSES)

            objectness_pred = class_logits[:, 0:1, :, :]
            class_pred = class_logits[:, 1:, :, :]
            loss_obj = obj_loss_fn(objectness_pred, obj_target)

            obj_mask = (obj_target.squeeze(1) == 1)
            class_indices = class_target.argmax(dim=1)
            class_pred_obj = class_pred.permute(0, 2, 3, 1)[obj_mask]
            class_gt_obj = class_indices[obj_mask]
            loss_cls = class_loss_fn(class_pred_obj, class_gt_obj) if class_pred_obj.numel() > 0 else torch.tensor(0.0, device=DEVICE)

            bbox_pred_obj = bbox_preds.permute(0, 2, 3, 1)[obj_mask]
            bbox_gt_obj = bbox_target.permute(0, 2, 3, 1)[obj_mask]
            loss_bbox = bbox_loss_fn(bbox_pred_obj, bbox_gt_obj) if bbox_pred_obj.numel() > 0 else torch.tensor(0.0, device=DEVICE)

            loss = loss_obj + loss_cls + loss_bbox
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Access and log detected objects
            print(f"Train: Detected Objects (XML): {[obj['name'] for obj in labels['xml']]}")
            print(f"Train: Detected Objects (YOLO): {[OBJECT_CATEGORIES[int(obj['category_id'])] for obj in labels['yolo']]}")

        evaluate_model(model, val_loader, obj_loss_fn, class_loss_fn, bbox_loss_fn, DEVICE, NUM_CLASSES, phase="Validation")

    print("Testing...")
    evaluate_model(model, test_loader, obj_loss_fn, class_loss_fn, bbox_loss_fn, DEVICE, NUM_CLASSES, phase="Test")


if __name__ == "__main__":
    train_model()