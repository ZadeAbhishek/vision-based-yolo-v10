import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PEDRoDataset  # Custom dataset
from model.recurrent_yolov10 import RecurrentYOLOv10  # Custom model
import torch.nn as nn

# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/PEDRo/"
BATCH_SIZE = 2  # Smaller batch size for stability
EPOCHS = 2
LEARNING_RATE = 1e-5
H, W, C_in, T = 512, 512, 5, 15  # Input size
NUM_CLASSES = 2
GRAD_CLIP = 1.0  # Gradient clipping threshold

# -----------------------------------------------------------------------------
# Collate Function
# -----------------------------------------------------------------------------
def custom_collate_fn(batch):
    vteis, labels = zip(*batch)
    vteis = torch.stack(vteis)  # (B, T, C_in, H, W)
    return vteis, labels

# -----------------------------------------------------------------------------
# Build Targets
# -----------------------------------------------------------------------------
def build_targets(class_logits, bbox_preds, labels, num_classes):
    B, _, H_out, W_out = class_logits.shape
    obj_target_one = torch.zeros(B, 1, H_out, W_out, device=class_logits.device)
    obj_target_many = torch.zeros(B, 1, H_out, W_out, device=class_logits.device)
    class_target = torch.zeros(B, num_classes, H_out, W_out, device=class_logits.device)
    bbox_target = torch.zeros(B, 4, H_out, W_out, device=class_logits.device)
    return obj_target_one, obj_target_many, class_target, bbox_target

# -----------------------------------------------------------------------------
# Validation Loop
# -----------------------------------------------------------------------------
def validate_model(model, val_loader, obj_loss_fn, class_loss_fn, bbox_loss_fn, device, num_classes):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for vtei, labels in val_loader:
            vtei = vtei.to(device)

            # Forward pass
            one_to_one_output, one_to_many_output = model(vtei)

            # Build targets
            obj_target_one, obj_target_many, class_target, bbox_target = build_targets(
                one_to_many_output, one_to_many_output, labels, num_classes
            )

            # Compute losses
            loss_one_to_one = obj_loss_fn(one_to_one_output, obj_target_one)
            loss_obj = obj_loss_fn(one_to_many_output[:, 0:1, :, :], obj_target_many)

            obj_mask = (obj_target_many.squeeze(1) == 1)
            class_pred_obj = one_to_many_output[:, 1:num_classes + 1, :, :].permute(0, 2, 3, 1)[obj_mask]
            class_gt_obj = class_target.argmax(dim=1)[obj_mask]
            loss_cls = (
                class_loss_fn(class_pred_obj, class_gt_obj)
                if class_pred_obj.numel() > 0
                else torch.tensor(0.0, device=device)
            )

            bbox_pred_obj = one_to_many_output[:, num_classes + 1:, :, :].permute(0, 2, 3, 1)[obj_mask]
            bbox_gt_obj = bbox_target.permute(0, 2, 3, 1)[obj_mask]
            loss_bbox = (
                bbox_loss_fn(bbox_pred_obj, bbox_gt_obj)
                if bbox_pred_obj.numel() > 0
                else torch.tensor(0.0, device=device)
            )

            loss = loss_one_to_one + loss_obj + loss_cls + loss_bbox
            total_loss += loss.item()

    avg_val_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
def train_model():
    train_dataset = PEDRoDataset(DATA_DIR, split="train", H=H, W=W, C_in=C_in, T=T)
    val_dataset = PEDRoDataset(DATA_DIR, split="val", H=H, W=W, C_in=C_in, T=T)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn
    )

    model = RecurrentYOLOv10(input_channels=C_in, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    obj_loss_fn = nn.BCEWithLogitsLoss()
    class_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()

    print(f"Using device: {DEVICE}")
    print("Starting training...")

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch_idx, (vtei, labels) in enumerate(train_loader):
            vtei = vtei.to(DEVICE)

            # Forward pass
            one_to_one_output, one_to_many_output = model(vtei)

            # Build targets
            obj_target_one, obj_target_many, class_target, bbox_target = build_targets(
                one_to_many_output, one_to_many_output, labels, NUM_CLASSES
            )

            # Compute losses
            loss_one_to_one = obj_loss_fn(one_to_one_output, obj_target_one)
            loss_obj = obj_loss_fn(one_to_many_output[:, 0:1, :, :], obj_target_many)

            obj_mask = (obj_target_many.squeeze(1) == 1)
            class_pred_obj = one_to_many_output[:, 1:NUM_CLASSES + 1, :, :].permute(0, 2, 3, 1)[obj_mask]
            class_gt_obj = class_target.argmax(dim=1)[obj_mask]
            loss_cls = (
                class_loss_fn(class_pred_obj, class_gt_obj)
                if obj_mask.any()
                else torch.tensor(0.0, device=DEVICE)
            )

            bbox_pred_obj = one_to_many_output[:, NUM_CLASSES + 1:, :, :].permute(0, 2, 3, 1)[obj_mask]
            bbox_gt_obj = bbox_target.permute(0, 2, 3, 1)[obj_mask]
            loss_bbox = (
                bbox_loss_fn(bbox_pred_obj, bbox_gt_obj)
                if obj_mask.any()
                else torch.tensor(0.0, device=DEVICE)
            )

            loss = loss_one_to_one + loss_obj + loss_cls + loss_bbox

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            print(
                f"Epoch {epoch + 1}/{EPOCHS}, Batch {batch_idx + 1}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}"
            )

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Training Loss: {avg_epoch_loss:.4f}")

        # Validation step
        val_loss = validate_model(model, val_loader, obj_loss_fn, class_loss_fn, bbox_loss_fn, DEVICE, NUM_CLASSES)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "recurrentyolov10_best.pth")
            print(f"Saved best model with Validation Loss: {val_loss:.4f}")

        scheduler.step()

    torch.save(model.state_dict(), "recurrentyolov10_last.pth")
    print("Training complete. Model saved.")

    return model


# -----------------------------------------------------------------------------
# Main / Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    trained_model = train_model()