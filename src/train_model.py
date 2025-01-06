import os
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import PEDRoDataset  # Ensure this path is correct
from model.AlternateV8 import ReYOLOv8s  # Ensure this path is correct

# ---------------------------
# Configure Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logging.info("Logging configuration complete.")

# ---------------------------
# Device Configuration
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")

# ---------------------------
# Hyperparameters and Configuration
# ---------------------------
DATA_DIR = "data/PEDRo/"
BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 5e-5
H, W, B = 260, 346, 5
train_limit = 2000
val_limit = 2000
test_limit = 2000

# ---------------------------
# Custom Collate Function
# ---------------------------
def custom_collate_fn(batch):
    vteis, labels = zip(*batch)
    vteis = torch.stack(vteis)
    batched_labels = {"yolo": [], "xml": []}
    for label in labels:
        batched_labels["yolo"].extend(label["yolo"])
        batched_labels["xml"].extend(label["xml"])
    return vteis, batched_labels

# ---------------------------
# Placeholder for YOLO-style Target Assignment
# ---------------------------
def build_targets(class_logits, bbox_preds, labels, num_classes):
    B, _, H_out, W_out = class_logits.shape
    obj_target = torch.zeros(B, 1, H_out, W_out, device=class_logits.device)
    class_target = torch.zeros(B, num_classes, H_out, W_out, device=class_logits.device)
    bbox_target = torch.zeros(B, 4, H_out, W_out, device=class_logits.device)
    # TODO: Implement YOLO-style target assignment logic here.
    return obj_target, class_target, bbox_target

# ---------------------------
# Evaluation Function
# ---------------------------
def evaluate_model(model, loader, obj_loss_fn, class_loss_fn, bbox_loss_fn, device, num_classes, phase="Validation"):
    model.eval()
    total_loss, total_obj_loss, total_class_loss, total_bbox_loss = 0, 0, 0, 0
    with torch.no_grad():
        for vtei, labels in loader:
            vtei = vtei.to(device)
            try:
                class_logits, bbox_preds = model(vtei)
            except Exception as e:
                logging.error(f"Model forward pass error during evaluation: {e}")
                continue

            obj_target, class_target, bbox_target = build_targets(class_logits, bbox_preds, labels, num_classes)
            objectness_pred = class_logits[:, :1, :, :]
            class_pred = class_logits[:, 1:, :, :]
            loss_obj = obj_loss_fn(objectness_pred, obj_target)

            obj_mask = (obj_target.squeeze(1) == 1)
            if obj_mask.sum() > 0:
                class_pred_obj = class_pred.permute(0, 2, 3, 1)[obj_mask]
                class_gt_obj = class_target.argmax(dim=1)[obj_mask]
                loss_cls = class_loss_fn(class_pred_obj, class_gt_obj)
                bbox_pred_obj = bbox_preds.permute(0, 2, 3, 1)[obj_mask]
                bbox_gt_obj = bbox_target.permute(0, 2, 3, 1)[obj_mask]
                loss_bbox = bbox_loss_fn(bbox_pred_obj, bbox_gt_obj)
            else:
                loss_cls = torch.tensor(0.0, device=device)
                loss_bbox = torch.tensor(0.0, device=device)

            loss = loss_obj + loss_cls + loss_bbox
            total_loss += loss.item()
            total_obj_loss += loss_obj.item()
            total_class_loss += loss_cls.item()
            total_bbox_loss += loss_bbox.item()

    avg_loss = total_loss / len(loader)
    logging.info(f"{phase} Loss: {avg_loss:.4f} | Objectness: {total_obj_loss/len(loader):.4f} | "
                 f"Classification: {total_class_loss/len(loader):.4f} | BBox: {total_bbox_loss/len(loader):.4f}")
    return avg_loss

# ---------------------------
# Training Function
# ---------------------------
def train_model():
    train_dataset = PEDRoDataset(DATA_DIR, split="train", H=H, W=W, B=B, limit=train_limit)
    val_dataset = PEDRoDataset(DATA_DIR, split="val", H=H, W=W, B=B, limit=val_limit)
    test_dataset = PEDRoDataset(DATA_DIR, split="test", H=H, W=W, B=B, limit=test_limit)
    OBJECT_CATEGORIES = train_dataset.object_categories
    NUM_CLASSES = len(OBJECT_CATEGORIES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    model = ReYOLOv8s(in_channels=B, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    obj_loss_fn = nn.BCEWithLogitsLoss()
    class_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()
    scaler = torch.cuda.amp.GradScaler()

    logging.info("Starting training...")
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_obj_loss, total_class_loss, total_bbox_loss = 0, 0, 0, 0
        for vtei, labels in train_loader:
            vtei = vtei.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                try:
                    class_logits, bbox_preds = model(vtei)
                except RuntimeError as e:
                    logging.error(f"Error during forward pass: {e}")
                    continue

                obj_target, class_target, bbox_target = build_targets(class_logits, bbox_preds, labels, NUM_CLASSES)
                objectness_pred = class_logits[:, :1, :, :]
                class_pred = class_logits[:, 1:, :, :]
                loss_obj = obj_loss_fn(objectness_pred, obj_target)

                obj_mask = (obj_target.squeeze(1) == 1)
                if obj_mask.sum() > 0:
                    class_pred_obj = class_pred.permute(0, 2, 3, 1)[obj_mask]
                    class_gt_obj = class_target.argmax(dim=1)[obj_mask]
                    loss_cls = class_loss_fn(class_pred_obj, class_gt_obj)
                    bbox_pred_obj = bbox_preds.permute(0, 2, 3, 1)[obj_mask]
                    bbox_gt_obj = bbox_target.permute(0, 2, 3, 1)[obj_mask]
                    loss_bbox = bbox_loss_fn(bbox_pred_obj, bbox_gt_obj)
                else:
                    loss_cls = torch.tensor(0.0, device=DEVICE)
                    loss_bbox = torch.tensor(0.0, device=DEVICE)

                loss = loss_obj + loss_cls + loss_bbox

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_obj_loss += loss_obj.item()
            total_class_loss += loss_cls.item()
            total_bbox_loss += loss_bbox.item()

        logging.info(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {total_loss/len(train_loader):.4f}")

        val_loss = evaluate_model(model, val_loader, obj_loss_fn, class_loss_fn, bbox_loss_fn, DEVICE, NUM_CLASSES, phase="Validation")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(DATA_DIR, "best_model.pth"))
            logging.info(f"Saved best model with validation loss: {val_loss:.4f}")

    logging.info("Training complete. Starting test evaluation...")
    evaluate_model(model, test_loader, obj_loss_fn, class_loss_fn, bbox_loss_fn, DEVICE, NUM_CLASSES, phase="Test")

if __name__ == '__main__':
    train_model()