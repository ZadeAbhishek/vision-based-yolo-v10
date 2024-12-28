import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PEDRoDataset  # <-- your custom dataset
from model.recurrent_yolov10 import RecurrentYOLOv10  # <-- your custom model
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# 1) Enable interactive mode so figures update without blocking
# -----------------------------------------------------------------------------
# plt.ion()

# 2) (Optional) Create one global figure & axis for reuse
# fig, ax = plt.subplots()

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/PEDRo/"
BATCH_SIZE = 16   # Adjust based on GPU memory
EPOCHS = 1
LEARNING_RATE = 0.001
H, W, C_in, T = 256, 256, 5, 15  # Example image specs
NUM_CLASSES = 2

# -----------------------------------------------------------------------------
# Visualization function (optional for debugging)
# -----------------------------------------------------------------------------
# def visualize_labels(...): 
#     ...

# -----------------------------------------------------------------------------
# Decoding function (example)
# -----------------------------------------------------------------------------
def decode_predictions(class_logits, bbox_preds, num_classes, confidence_threshold=0.5):
    B, _, H, W = class_logits.shape
    detections = []

    for b in range(B):
        for h_idx in range(H):
            for w_idx in range(W):
                class_scores = class_logits[b, :, h_idx, w_idx]
                max_score, class_idx = class_scores.max(dim=0)
                if max_score > confidence_threshold:
                    bbox = bbox_preds[b, :, h_idx, w_idx].tolist()
                    detections.append(
                        (b, h_idx, w_idx, class_idx.item(), max_score.item(), bbox)
                    )
    return detections

# -----------------------------------------------------------------------------
# Collate function
# -----------------------------------------------------------------------------
def custom_collate_fn(batch):
    vteis, labels = zip(*batch)
    vteis = torch.stack(vteis)  # (B, T, C_in, H, W)
    return vteis, labels

# -----------------------------------------------------------------------------
# Build targets (stub example)
# -----------------------------------------------------------------------------
def build_targets(class_logits, bbox_preds, labels, num_classes):
    B, _, H_out, W_out = class_logits.shape
    obj_target_one  = torch.zeros(B, 1, H_out, W_out, device=class_logits.device)
    obj_target_many = torch.zeros(B, 1, H_out, W_out, device=class_logits.device)
    class_target    = torch.zeros(B, num_classes, H_out, W_out, device=class_logits.device)
    bbox_target     = torch.zeros(B, 4, H_out, W_out, device=class_logits.device)
    return obj_target_one, obj_target_many, class_target, bbox_target

# -----------------------------------------------------------------------------
# Printing YOLO labels for debugging
# -----------------------------------------------------------------------------
def print_labels(labels, prefix="Labels"):
    print(f"{prefix} Labels:")
    for sample_idx, box_list in enumerate(labels):
        print(f"  Sample {sample_idx}:")
        if not box_list:
            print("    (no bounding boxes)")
            continue
        for box_idx, box in enumerate(box_list):
            class_id  = int(box[0])
            x_center  = box[1]
            y_center  = box[2]
            width     = box[3]
            height    = box[4]
            print(
                f"    Box {box_idx}: "
                f"class={class_id}, "
                f"x={x_center:.4f}, "
                f"y={y_center:.4f}, "
                f"w={width:.4f}, "
                f"h={height:.4f}"
            )

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

            # Decode predictions (for debugging)
            detections = decode_predictions(
                one_to_many_output[:, 1:num_classes + 1, :, :],
                one_to_many_output[:, num_classes + 1:, :, :],
                num_classes
            )
            print(f"[Validation] Detected Objects: {detections}")

            # Compute losses
            loss_one_to_one = obj_loss_fn(one_to_one_output, obj_target_one)
            loss_obj        = obj_loss_fn(one_to_many_output[:, 0:1, :, :], obj_target_many)

            # Class loss (only for locations where obj_target=1)
            obj_mask        = (obj_target_many.squeeze(1) == 1)
            class_pred_obj  = one_to_many_output[:, 1:num_classes + 1, :, :].permute(0, 2, 3, 1)[obj_mask]
            class_gt_obj    = class_target.argmax(dim=1)[obj_mask]
            loss_cls        = class_loss_fn(class_pred_obj, class_gt_obj) if class_pred_obj.numel() > 0 else torch.tensor(0.0, device=device)

            # BBox loss (only for locations where obj_target=1)
            bbox_pred_obj   = one_to_many_output[:, num_classes + 1:, :, :].permute(0, 2, 3, 1)[obj_mask]
            bbox_gt_obj     = bbox_target.permute(0, 2, 3, 1)[obj_mask]
            loss_bbox       = bbox_loss_fn(bbox_pred_obj, bbox_gt_obj) if bbox_pred_obj.numel() > 0 else torch.tensor(0.0, device=device)

            loss = loss_one_to_one + loss_obj + loss_cls + loss_bbox
            total_loss += loss.item()

    avg_val_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    print(f"Validation Loss: {avg_val_loss:.4f}")

# -----------------------------------------------------------------------------
# Testing Loop (newly added)
# -----------------------------------------------------------------------------
def test_model(model, test_loader, obj_loss_fn, class_loss_fn, bbox_loss_fn, device, num_classes):
    """
    Similar to validate_model, but we can print or record final metrics here.
    """
    model.eval()
    total_loss = 0
    all_detections = []

    with torch.no_grad():
        for vtei, labels in test_loader:
            vtei = vtei.to(device)

            # Forward pass
            one_to_one_output, one_to_many_output = model(vtei)

            # Build targets
            obj_target_one, obj_target_many, class_target, bbox_target = build_targets(
                one_to_many_output, one_to_many_output, labels, num_classes
            )

            # Decode predictions
            detections = decode_predictions(
                one_to_many_output[:, 1:num_classes + 1, :, :],
                one_to_many_output[:, num_classes + 1:, :, :],
                num_classes
            )
            all_detections.extend(detections)  # accumulate all detections

            # Compute losses
            loss_one_to_one = obj_loss_fn(one_to_one_output, obj_target_one)
            loss_obj        = obj_loss_fn(one_to_many_output[:, 0:1, :, :], obj_target_many)
            obj_mask        = (obj_target_many.squeeze(1) == 1)
            class_pred_obj  = one_to_many_output[:, 1:num_classes + 1, :, :].permute(0, 2, 3, 1)[obj_mask]
            class_gt_obj    = class_target.argmax(dim=1)[obj_mask]
            loss_cls        = class_loss_fn(class_pred_obj, class_gt_obj) if class_pred_obj.numel() > 0 else torch.tensor(0.0, device=device)
            bbox_pred_obj   = one_to_many_output[:, num_classes + 1:, :, :].permute(0, 2, 3, 1)[obj_mask]
            bbox_gt_obj     = bbox_target.permute(0, 2, 3, 1)[obj_mask]
            loss_bbox       = bbox_loss_fn(bbox_pred_obj, bbox_gt_obj) if bbox_pred_obj.numel() > 0 else torch.tensor(0.0, device=device)

            loss = loss_one_to_one + loss_obj + loss_cls + loss_bbox
            total_loss += loss.item()

    avg_test_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Total Detections on Test Set: {len(all_detections)}")

    # If you want to look at a sample of test detections
    for i, detection in enumerate(all_detections[:5]):  # print first 5
        print(f"Test Detection {i}: {detection}")

    return avg_test_loss, all_detections

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
def train_model():
    # 1) Load datasets
    train_dataset = PEDRoDataset(
        data_dir=DATA_DIR,
        split="train",
        H=H,
        W=W,
        C_in=C_in,
        T=T
    )
    val_dataset   = PEDRoDataset(
        data_dir=DATA_DIR,
        split="val",
        H=H,
        W=W,
        C_in=C_in,
        T=T
    )
    # 2) Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # or 1, see note for macOS
        collate_fn=custom_collate_fn,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=False
    )

    # 3) Initialize Model
    model = RecurrentYOLOv10(input_channels=C_in, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4) Define Loss functions
    obj_loss_fn   = nn.BCEWithLogitsLoss()
    class_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn  = nn.SmoothL1Loss()

    print(f"Using device: {DEVICE}")
    print("Starting training...")

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
            loss_obj        = obj_loss_fn(one_to_many_output[:, 0:1, :, :], obj_target_many)

            obj_mask        = (obj_target_many.squeeze(1) == 1)
            class_pred_obj  = one_to_many_output[:, 1:NUM_CLASSES + 1, :, :].permute(0, 2, 3, 1)[obj_mask]
            class_gt_obj    = class_target.argmax(dim=1)[obj_mask]
            loss_cls        = class_loss_fn(class_pred_obj, class_gt_obj) if class_pred_obj.numel() > 0 else torch.tensor(0.0, device=DEVICE)

            bbox_pred_obj   = one_to_many_output[:, NUM_CLASSES + 1:, :, :].permute(0, 2, 3, 1)[obj_mask]
            bbox_gt_obj     = bbox_target.permute(0, 2, 3, 1)[obj_mask]
            loss_bbox       = bbox_loss_fn(bbox_pred_obj, bbox_gt_obj) if bbox_pred_obj.numel() > 0 else torch.tensor(0.0, device=DEVICE)

            loss = loss_one_to_one + loss_obj + loss_cls + loss_bbox

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch + 1} Average Training Loss: {avg_epoch_loss:.4f}")

        # Validation step
        validate_model(model, val_loader, obj_loss_fn, class_loss_fn, bbox_loss_fn, DEVICE, NUM_CLASSES)

    # 5) Save the model after training
    torch.save(model.state_dict(), "recurrentyolov10_trained.pth")
    print("Model saved to recurrentyolov10_trained.pth")

    return model  # Return the trained model if needed

# -----------------------------------------------------------------------------
# Main / Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Train the model
    trained_model = train_model()

    # 2) Create & evaluate on Test dataset
    test_dataset = PEDRoDataset(
        data_dir=DATA_DIR,
        split="test",
        H=H,
        W=W,
        C_in=C_in,
        T=T
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # or 1
        collate_fn=custom_collate_fn,
        pin_memory=False
    )

    # (Optional) Load the saved model weights again 
    # e.g., if you're running test in a separate script:
    # trained_model = RecurrentYOLOv10(input_channels=C_in, num_classes=NUM_CLASSES).to(DEVICE)
    # trained_model.load_state_dict(torch.load("recurrentyolov10_trained.pth"))
    # trained_model.eval()

    # 3) Run the testing routine
    obj_loss_fn   = nn.BCEWithLogitsLoss()
    class_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn  = nn.SmoothL1Loss()

    print("\nRunning Test Evaluation...")
    test_loss, test_detections = test_model(
        trained_model,
        test_loader,
        obj_loss_fn,
        class_loss_fn,
        bbox_loss_fn,
        DEVICE,
        NUM_CLASSES
    )
    print(f"Finished Test Evaluation. Test Loss = {test_loss:.4f}")
    print(f"Number of detections on test set = {len(test_detections)}")