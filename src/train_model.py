# src/train_model.py

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

# ---------------------------
# Device Configuration
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")

# ---------------------------
# Hyperparameters and Configuration
# ---------------------------
DATA_DIR = "data/PEDRo/"  # Update as per your directory structure
BATCH_SIZE = 1  # Reduced to 1 to avoid CUDA OOM
EPOCHS = 1  # Set to desired number of epochs
LEARNING_RATE = 5e-5
H, W, B = 260, 346, 5  # Image height, width, temporal bins
NUM_CLASSES = None  # To be determined based on dataset categories

# ---------------------------
# Custom Collate Function
# ---------------------------
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

# ---------------------------
# Placeholder for YOLO-style Target Assignment
# ---------------------------
def build_targets(class_logits, bbox_preds, labels, num_classes):
    """
    Placeholder for YOLO-style target assignment.
    This function should convert ground truth boxes and classes into target tensors aligned with the output grid.
    """
    B, _, H_out, W_out = class_logits.shape
    obj_target = torch.zeros(B, 1, H_out, W_out, device=class_logits.device)
    class_target = torch.zeros(B, num_classes, H_out, W_out, device=class_logits.device)
    bbox_target = torch.zeros(B, 4, H_out, W_out, device=class_logits.device)

    # Implement YOLO-style target assignment logic here.
    # This typically involves assigning each ground truth box to a grid cell and encoding the box parameters.

    return obj_target, class_target, bbox_target

# ---------------------------
# Evaluation Function
# ---------------------------
def evaluate_model(model, loader, obj_loss_fn, class_loss_fn, bbox_loss_fn, device, num_classes, phase="Validation"):
    """
    Evaluate the model on a given dataset loader.

    Args:
        model (torch.nn.Module): The trained model.
        loader (DataLoader): DataLoader for the dataset.
        obj_loss_fn (nn.Module): Loss function for objectness.
        class_loss_fn (nn.Module): Loss function for classification.
        bbox_loss_fn (nn.Module): Loss function for bounding boxes.
        device (str): Device to perform computations on.
        num_classes (int): Number of object classes.
        phase (str): Phase name for logging ('Validation' or 'Test').

    Returns:
        float: Average loss over the dataset.
    """
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

            objectness_pred = class_logits[:, :1, :, :]  # First channel for objectness
            class_pred = class_logits[:, 1:, :, :]      # Remaining channels for class predictions
            loss_obj = obj_loss_fn(objectness_pred, obj_target)

            obj_mask = (obj_target.squeeze(1) == 1)
            class_indices = class_target.argmax(dim=1)
            if obj_mask.sum() > 0:
                class_pred_obj = class_pred.permute(0, 2, 3, 1)[obj_mask]
                class_gt_obj = class_indices[obj_mask]
                loss_cls = class_loss_fn(class_pred_obj, class_gt_obj)
            else:
                loss_cls = torch.tensor(0.0, device=device)

            if obj_mask.sum() > 0:
                bbox_pred_obj = bbox_preds.permute(0, 2, 3, 1)[obj_mask]
                bbox_gt_obj = bbox_target.permute(0, 2, 3, 1)[obj_mask]
                loss_bbox = bbox_loss_fn(bbox_pred_obj, bbox_gt_obj)
            else:
                loss_bbox = torch.tensor(0.0, device=device)

            loss = loss_obj + loss_cls + loss_bbox
            total_loss += loss.item()
            total_obj_loss += loss_obj.item()
            total_class_loss += loss_cls.item()
            total_bbox_loss += loss_bbox.item()

            # Log ground truth objects
            if 'OBJECT_CATEGORIES' in globals() and OBJECT_CATEGORIES:
                gt_xml = [obj['name'] for obj in labels['xml']]
                gt_yolo = [OBJECT_CATEGORIES[int(obj['category_id'])] for obj in labels['yolo']]
                logging.debug(f"{phase} - Ground Truth XML: {gt_xml}")
                logging.debug(f"{phase} - Ground Truth YOLO: {gt_yolo}")

    average_loss = total_loss / len(loader)
    average_obj_loss = total_obj_loss / len(loader)
    average_class_loss = total_class_loss / len(loader)
    average_bbox_loss = total_bbox_loss / len(loader)
    logging.info(f"{phase} Loss: {average_loss:.4f} | Objectness: {average_obj_loss:.4f} | "
                 f"Classification: {average_class_loss:.4f} | BBox: {average_bbox_loss:.4f}")
    return average_loss

# ---------------------------
# Training Function
# ---------------------------
def train_model():
    """
    Main function to train the ReYOLOv8s model.
    """
    # Initialize dataset with a limit of 2,000 samples
    subset_size = 2000
    try:
        train_dataset = PEDRoDataset(
            data_dir=DATA_DIR,
            split="train",
            H=H,
            W=W,
            B=B,
            transform=None,  # Add your transformation function if needed
            limit=subset_size
        )
        logging.info(f"Loaded {len(train_dataset)} samples for split 'train' with a limit of {subset_size}.")
    except FileNotFoundError as e:
        logging.error(e)
        return

    # Load validation and test datasets without limits
    try:
        val_dataset = PEDRoDataset(data_dir=DATA_DIR, split="val", H=H, W=W, B=B)
        logging.info(f"Loaded {len(val_dataset)} samples for split 'val'.")
    except FileNotFoundError as e:
        logging.error(e)
        return

    try:
        test_dataset = PEDRoDataset(data_dir=DATA_DIR, split="test", H=H, W=W, B=B)
        logging.info(f"Loaded {len(test_dataset)} samples for split 'test'.")
    except FileNotFoundError as e:
        logging.error(e)
        return

    # Define class categories and number of classes
    OBJECT_CATEGORIES = train_dataset.object_categories
    NUM_CLASSES = len(OBJECT_CATEGORIES)
    logging.info(f"Number of classes: {NUM_CLASSES}")
    logging.info(f"Classes: {OBJECT_CATEGORIES}")

    if NUM_CLASSES == 0:
        # Define default classes or load from a file
        OBJECT_CATEGORIES = ['person', 'car', 'bicycle']  # Example classes
        NUM_CLASSES = len(OBJECT_CATEGORIES)
        logging.info("Default OBJECT_CATEGORIES set.")
        logging.info(f"Number of classes: {NUM_CLASSES}")
        logging.info(f"Classes: {OBJECT_CATEGORIES}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing errors
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0
    )

    # Initialize model with num_classes
    try:
        model = ReYOLOv8s(in_channels=B, num_classes=NUM_CLASSES).to(DEVICE)
        logging.info(f"Model initialized with in_channels={B} and num_classes={NUM_CLASSES}.")
    except TypeError as e:
        logging.error(f"Model initialization error: {e}")
        logging.error("Ensure that ReYOLOv8s accepts 'in_channels' and 'num_classes' as arguments.")
        return

    # Calculate and log the number of million parameters
    total_params = sum(p.numel() for p in model.parameters())
    million_params = total_params / 1e6
    logging.info(f"Model has {million_params:.2f} Million parameters.")

    # Initialize optimizer and loss functions
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    obj_loss_fn = nn.BCEWithLogitsLoss()
    class_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()

    scaler = torch.cuda.amp.GradScaler()  # For mixed precision

    logging.info("Starting training...")

    best_val_loss = float('inf')
    best_model_path = os.path.join(DATA_DIR, "best_ReYOLOv8s.pth")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_obj_loss = 0
        epoch_class_loss = 0
        epoch_bbox_loss = 0

        for batch_idx, (vtei, labels) in enumerate(train_loader):
            vtei = vtei.to(DEVICE)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                try:
                    class_logits, bbox_preds = model(vtei)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        logging.error("CUDA out of memory during forward pass. Skipping this batch.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        logging.error(f"Model forward pass error: {e}")
                        torch.cuda.empty_cache()
                        continue
                except Exception as e:
                    logging.error(f"Unexpected error during forward pass: {e}")
                    torch.cuda.empty_cache()
                    continue

                # Build targets
                obj_target, class_target, bbox_target = build_targets(class_logits, bbox_preds, labels, NUM_CLASSES)

                # Compute losses
                objectness_pred = class_logits[:, :1, :, :]  # First channel for objectness
                class_pred = class_logits[:, 1:, :, :]      # Remaining channels for class predictions
                loss_obj = obj_loss_fn(objectness_pred, obj_target)

                obj_mask = (obj_target.squeeze(1) == 1)
                class_indices = class_target.argmax(dim=1)
                if obj_mask.sum() > 0:
                    class_pred_obj = class_pred.permute(0, 2, 3, 1)[obj_mask]
                    class_gt_obj = class_indices[obj_mask]
                    loss_cls = class_loss_fn(class_pred_obj, class_gt_obj)
                else:
                    loss_cls = torch.tensor(0.0, device=DEVICE)

                if obj_mask.sum() > 0:
                    bbox_pred_obj = bbox_preds.permute(0, 2, 3, 1)[obj_mask]
                    bbox_gt_obj = bbox_target.permute(0, 2, 3, 1)[obj_mask]
                    loss_bbox = bbox_loss_fn(bbox_pred_obj, bbox_gt_obj)
                else:
                    loss_bbox = torch.tensor(0.0, device=DEVICE)

                loss = loss_obj + loss_cls + loss_bbox

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_obj_loss += loss_obj.item()
            epoch_class_loss += loss_cls.item()
            epoch_bbox_loss += loss_bbox.item()

            # Log ground truth objects
            gt_xml = [obj['name'] for obj in labels['xml']]
            gt_yolo = [OBJECT_CATEGORIES[int(obj['category_id'])] for obj in labels['yolo']]
            logging.debug(f"Train Epoch {epoch+1} Batch {batch_idx}: Ground Truth XML: {gt_xml}")
            logging.debug(f"Train Epoch {epoch+1} Batch {batch_idx}: Ground Truth YOLO: {gt_yolo}")

            if batch_idx % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Calculate and log average losses for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_obj_loss = epoch_obj_loss / len(train_loader)
        avg_epoch_class_loss = epoch_class_loss / len(train_loader)
        avg_epoch_bbox_loss = epoch_bbox_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{EPOCHS} Summary: Avg Loss: {avg_epoch_loss:.4f} | "
                     f"Objectness Loss: {avg_epoch_obj_loss:.4f} | "
                     f"Classification Loss: {avg_epoch_class_loss:.4f} | "
                     f"BBox Loss: {avg_epoch_bbox_loss:.4f}")

        # Validate the model
        val_loss = evaluate_model(
            model,
            val_loader,
            obj_loss_fn,
            class_loss_fn,
            bbox_loss_fn,
            DEVICE,
            NUM_CLASSES,
            phase="Validation"
        )

        # Save the model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_model_path)
            logging.info(f"Best model saved at epoch {epoch+1} with validation loss {val_loss:.4f}")

    # After training, evaluate on test set
    logging.info("Training completed. Starting testing...")
    test_loss = evaluate_model(
        model,
        test_loader,
        obj_loss_fn,
        class_loss_fn,
        bbox_loss_fn,
        DEVICE,
        NUM_CLASSES,
        phase="Test"
    )
    logging.info(f"Test Loss: {test_loss:.4f}")

    # Optionally, save the final model
    final_model_path = os.path.join(DATA_DIR, "final_ReYOLOv8s.pth")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved at {final_model_path}")

if __name__ == '__main__':
    train_model()