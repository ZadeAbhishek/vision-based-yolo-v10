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
plt.ion()

# 2) Create one global figure & axis for reuse (avoid multiple windows).
fig, ax = plt.subplots()

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/PEDRo/"
BATCH_SIZE = 8    # Adjust based on GPU memory. Start with 8.
EPOCHS = 20       # Increased from 1 for sufficient training
LEARNING_RATE = 0.001  # Lowered from 0.01 for stable convergence
H, W, B = 256, 256, 5    # Standardized image size to 256x256
NUM_CLASSES = 2
MAX_BOXES = 50

# -----------------------------------------------------------------------------
# Single-window visualization function
# -----------------------------------------------------------------------------
def visualize_labels(vtei, labels, idx=0):
    """
    Visualizes a single input frame using a *single* window in interactive mode.
    - Reuses the global figure & axis so we don't open new windows each time.
    - Non-blocking: the code will continue running.
    Args:
        vtei:   Tensor of shape [batch_size, channels=5, H=260, W=346].
        labels: List of labels for the batch.
        idx:    Which sample in the batch to visualize.
    """
    # Convert the first channel to numpy for grayscale
    frame = vtei[idx].cpu().numpy()  # shape: (5, 260, 346)
    frame = frame[0]                 # shape: (260, 346) for grayscale
    
    # Clear the old image from the axis
    ax.clear()
    
    # Display the new frame
    ax.imshow(frame, cmap='gray')
    ax.set_title(f"Labels: {labels[idx]}")

    # Update the plot without blocking
    fig.canvas.draw()
    fig.canvas.flush_events()  # or plt.pause(0.001)


# -----------------------------------------------------------------------------
# Decoding function (example)
# -----------------------------------------------------------------------------
def decode_predictions(class_logits, bbox_preds, num_classes, confidence_threshold=0.5):
    """
    Decodes the model's predictions for demonstration.
    Args:
        class_logits: Tensor [B, num_classes, H, W]
        bbox_preds:   Tensor [B, 4, H, W]
        num_classes:  Number of classes.
        confidence_threshold: Score threshold for a detection.
    Returns:
        List of (batch_idx, h, w, class_idx, score, bbox).
    """
    B, _, H, W = class_logits.shape
    detections = []

    for b in range(B):
        for h in range(H):
            for w in range(W):
                class_scores = class_logits[b, :, h, w]
                max_score, class_idx = class_scores.max(dim=0)
                if max_score > confidence_threshold:
                    bbox = bbox_preds[b, :, h, w].tolist()
                    detections.append(
                        (b, h, w, class_idx.item(), max_score.item(), bbox)
                    )
    return detections


# -----------------------------------------------------------------------------
# Collate function
# -----------------------------------------------------------------------------
def custom_collate_fn(batch):
    """
    Your custom collate that stacks inputs and keeps labels as a list.
    """
    vteis, labels = zip(*batch)
    vteis = torch.stack(vteis)
    return vteis, labels


# -----------------------------------------------------------------------------
# Build targets (stub example)
# -----------------------------------------------------------------------------
def build_targets(class_logits, bbox_preds, labels, num_classes):
    """
    Dummy target building. Replace with your real logic based on 'labels'.
    Returns zero tensors for demonstration.
    """
    B, _, H_out, W_out = class_logits.shape
    obj_target_one  = torch.zeros(B, 1, H_out, W_out, device=class_logits.device)
    obj_target_many = torch.zeros(B, 1, H_out, W_out, device=class_logits.device)
    class_target    = torch.zeros(B, num_classes, H_out, W_out, device=class_logits.device)
    bbox_target     = torch.zeros(B, 4, H_out, W_out, device=class_logits.device)
    return obj_target_one, obj_target_many, class_target, bbox_target

def print_labels(labels, prefix="Labels"):
    """
    Prints YOLO-style labels (list of lists) in a readable format.
    Each inner list corresponds to a single sample's bounding boxes.
    Each bounding box is [class_id, x_center, y_center, width, height].

    Args:
        labels (List[List[List[float]]]): 
            A batch of labels. For example, 'labels' might be:
              [
                [[0, 0.55, 0.40, 0.10, 0.15], [1, 0.30, 0.20, 0.05, 0.10]],
                [[0, 0.60, 0.50, 0.20, 0.20]],
                []
              ]
            Where each item in 'labels' is a list of bounding boxes for one sample.
        prefix (str): A label to prefix your print statements (e.g. "Validation" or "Training").
    """
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
            print_labels(labels=labels,prefix="Validation:")

            # We can visualize less frequently here as well
            visualize_labels(vtei, labels, idx=0)

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
            print(f"Detected Objects: {detections}")

            # Compute losses
            loss_one_to_one = obj_loss_fn(one_to_one_output, obj_target_one)
            loss_obj        = obj_loss_fn(one_to_many_output[:, 0:1, :, :], obj_target_many)

            # Class loss (only for locations where obj_target=1)
            obj_mask        = (obj_target_many.squeeze(1) == 1)
            class_pred_obj  = one_to_many_output[:, 1:num_classes + 1, :, :].permute(0, 2, 3, 1)[obj_mask]
            class_gt_obj    = class_target.argmax(dim=1)[obj_mask]
            loss_cls        = class_loss_fn(class_pred_obj, class_gt_obj) if class_pred_obj.numel() > 0 else 0.0

            # BBox loss (only for locations where obj_target=1)
            bbox_pred_obj   = one_to_many_output[:, num_classes + 1:, :, :].permute(0, 2, 3, 1)[obj_mask]
            bbox_gt_obj     = bbox_target.permute(0, 2, 3, 1)[obj_mask]
            loss_bbox       = bbox_loss_fn(bbox_pred_obj, bbox_gt_obj) if bbox_pred_obj.numel() > 0 else 0.0

            # Sum up all loss parts
            loss = loss_one_to_one + loss_obj + loss_cls + loss_bbox
            total_loss += loss.item()

    avg_val_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    print(f"Validation Loss: {avg_val_loss:.4f}")


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
def train_model():
    # Load datasets
    train_dataset = PEDRoDataset(data_dir=DATA_DIR, split="train", H=H, W=W, B=B)
    val_dataset   = PEDRoDataset(data_dir=DATA_DIR, split="val",   H=H, W=W, B=B)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )

    # Initialize Model
    model = RecurrentYOLOv10(input_channels=B, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Define Loss functions
    obj_loss_fn   = nn.BCEWithLogitsLoss()
    class_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn  = nn.SmoothL1Loss()

    print(f"Using device: {DEVICE}")
    print("Starting training...")

    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (vtei, labels) in enumerate(train_loader):
            vtei = vtei.to(DEVICE)

            # Print labels for debugging
            print(f"Training Batch {batch_idx}")
            print_labels(labels=labels,prefix="Training:")

            # OPTIONAL: Only visualize every N batches to reduce overhead
            if batch_idx % 10 == 0:  
                visualize_labels(vtei, labels, idx=0)

            # Forward pass
            one_to_one_output, one_to_many_output = model(vtei)

            # Build targets
            obj_target_one, obj_target_many, class_target, bbox_target = build_targets(
                one_to_many_output, one_to_many_output, labels, NUM_CLASSES
            )

            # Compute losses
            loss_one_to_one = obj_loss_fn(one_to_one_output, obj_target_one)
            loss_obj        = obj_loss_fn(one_to_many_output[:, 0:1, :, :], obj_target_many)

            # Only compute class & bbox losses where obj_target is 1
            obj_mask        = (obj_target_many.squeeze(1) == 1)
            class_pred_obj  = one_to_many_output[:, 1:NUM_CLASSES + 1, :, :].permute(0, 2, 3, 1)[obj_mask]
            class_gt_obj    = class_target.argmax(dim=1)[obj_mask]
            loss_cls        = class_loss_fn(class_pred_obj, class_gt_obj) if class_pred_obj.numel() > 0 else 0.0

            bbox_pred_obj   = one_to_many_output[:, NUM_CLASSES + 1:, :, :].permute(0, 2, 3, 1)[obj_mask]
            bbox_gt_obj     = bbox_target.permute(0, 2, 3, 1)[obj_mask]
            loss_bbox       = bbox_loss_fn(bbox_pred_obj, bbox_gt_obj) if bbox_pred_obj.numel() > 0 else 0.0

            # Sum all loss components
            loss = loss_one_to_one + loss_obj + loss_cls + loss_bbox

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Validation step after each epoch
        validate_model(model, val_loader, obj_loss_fn, class_loss_fn, bbox_loss_fn, DEVICE, NUM_CLASSES)

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train_model()