import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PEDRoDataset
from model.recurrent_yolov8 import RecurrentYOLOv8
import torch.nn as nn

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/PEDRo/"
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.01
H, W, B = 260, 346, 5
NUM_CLASSES = 2
MAX_BOXES = 50  # Not used the same way as before, but keep if needed.

def custom_collate_fn(batch):
    """
    Custom collate function that stacks VTEIs and keeps labels unpadded.
    This allows YOLO-style target assignment later.
    """
    vteis, labels = zip(*batch)
    vteis = torch.stack(vteis)
    # labels is a list of [N_boxes, 5], do not pad here.
    return vteis, labels

def build_targets(class_logits, bbox_preds, labels, num_classes):
    """
    Placeholder for YOLO-style target assignment.
    This function must:
    - Convert ground-truth boxes and classes into target tensors aligned with the output grid.
    - Return obj_target, class_target, bbox_target of shapes:
      obj_target:   [B, 1, H_out, W_out]
      class_target: [B, num_classes, H_out, W_out]
      bbox_target:  [B, 4, H_out, W_out]
    """
    B, _, H_out, W_out = class_logits.shape

    obj_target = torch.zeros(B, 1, H_out, W_out, device=class_logits.device)
    class_target = torch.zeros(B, num_classes, H_out, W_out, device=class_logits.device)
    bbox_target = torch.zeros(B, 4, H_out, W_out, device=class_logits.device)

    # Implement YOLO target assignment logic here.
    # For each image:
    #   1. Convert normalized (x,y,w,h) in labels to grid coordinates.
    #   2. Identify the cell (cx, cy) responsible for each box.
    #   3. Set obj_target[b,0,cy,cx] = 1
    #   4. Set class_target[b,:,cy,cx] = one-hot class vector
    #   5. Set bbox_target[b,:,cy,cx] = [tx, ty, tw, th] (encoded bbox params)

    return obj_target, class_target, bbox_target

def validate_model(model, val_loader, obj_loss_fn, class_loss_fn, bbox_loss_fn, device, num_classes):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for vtei, labels in val_loader:
            vtei = vtei.to(device)
            class_logits, bbox_preds = model(vtei)

            obj_target, class_target, bbox_target = build_targets(class_logits, bbox_preds, labels, num_classes)

            # In this example, assume channel 0 of class_logits is objectness
            # and channels [1:] are class predictions
            objectness_pred = class_logits[:,0:1,:,:]
            class_pred = class_logits[:,1:,:,:]

            loss_obj = obj_loss_fn(objectness_pred, obj_target)

            obj_mask = (obj_target.squeeze(1) == 1)
            class_indices = class_target.argmax(dim=1)
            class_pred_obj = class_pred.permute(0,2,3,1)[obj_mask]
            class_gt_obj = class_indices[obj_mask]
            if class_pred_obj.numel() > 0:
                loss_cls = class_loss_fn(class_pred_obj, class_gt_obj)
            else:
                loss_cls = torch.tensor(0.0, device=device)

            bbox_pred_obj = bbox_preds.permute(0,2,3,1)[obj_mask]
            bbox_gt_obj = bbox_target.permute(0,2,3,1)[obj_mask]
            if bbox_pred_obj.numel() > 0:
                loss_bbox = bbox_loss_fn(bbox_pred_obj, bbox_gt_obj)
            else:
                loss_bbox = torch.tensor(0.0, device=device)

            loss = loss_obj + loss_cls + loss_bbox
            total_loss += loss.item()

    print(f"Validation Loss: {total_loss / len(val_loader):.4f}")

def train_model():
    # Load datasets
    train_dataset = PEDRoDataset(data_dir=DATA_DIR, split="train", H=H, W=W, B=B)
    val_dataset = PEDRoDataset(data_dir=DATA_DIR, split="val", H=H, W=W, B=B)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate_fn
    )

    model = RecurrentYOLOv8(input_channels=B, num_classes=NUM_CLASSES, max_boxes=MAX_BOXES).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

    # Example YOLO losses
    obj_loss_fn = nn.BCEWithLogitsLoss()  # Objectness
    class_loss_fn = nn.CrossEntropyLoss() # Classification
    bbox_loss_fn = nn.SmoothL1Loss()      # Bounding boxes (placeholder, YOLO often uses CIoU/GIoU)

    print(f"Using device: {DEVICE}")
    print("Starting training...")

    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (vtei, labels) in enumerate(train_loader):
            vtei = vtei.to(DEVICE)
            class_logits, bbox_preds = model(vtei)

            obj_target, class_target, bbox_target = build_targets(class_logits, bbox_preds, labels, NUM_CLASSES)

            # Assume first channel is objectness
            objectness_pred = class_logits[:,0:1,:,:]
            class_pred = class_logits[:,1:,:,:]

            loss_obj = obj_loss_fn(objectness_pred, obj_target)

            obj_mask = (obj_target.squeeze(1) == 1)
            class_indices = class_target.argmax(dim=1)
            class_pred_obj = class_pred.permute(0,2,3,1)[obj_mask]
            class_gt_obj = class_indices[obj_mask]
            if class_pred_obj.numel() > 0:
                loss_cls = class_loss_fn(class_pred_obj, class_gt_obj)
            else:
                loss_cls = torch.tensor(0.0, device=DEVICE)

            bbox_pred_obj = bbox_preds.permute(0,2,3,1)[obj_mask]
            bbox_gt_obj = bbox_target.permute(0,2,3,1)[obj_mask]
            if bbox_pred_obj.numel() > 0:
                loss_bbox = bbox_loss_fn(bbox_pred_obj, bbox_gt_obj)
            else:
                loss_bbox = torch.tensor(0.0, device=DEVICE)

            loss = loss_obj + loss_cls + loss_bbox
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        validate_model(model, val_loader, obj_loss_fn, class_loss_fn, bbox_loss_fn, DEVICE, NUM_CLASSES)

if __name__ == "__main__":
    train_model()