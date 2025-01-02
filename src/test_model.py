import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import PEDRoDataset  # Ensure your dataset is properly implemented
from model.recurrent_yolov10 import RecurrentYOLOv10  # Ensure your model is implemented
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/PEDRo/"
BATCH_SIZE = 2
H, W, C_in, T = 512, 512, 5, 15  # Input size
NUM_CLASSES = 2

# -----------------------------------------------------------------------------
# Utility: Decode Predictions
# -----------------------------------------------------------------------------
def decode_predictions(class_logits, bbox_preds, num_classes, confidence_threshold=0.5):
    """
    Decodes class logits and bbox predictions into a list of detections.
    """
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
# Visualization: Plot VTEI with Bounding Boxes
# -----------------------------------------------------------------------------
def plot_vtei_with_bboxes(vtei, bboxes, class_names, figsize=(10, 10)):
    """
    Plots VTEI frames with bounding boxes overlaid.
    """
    plt.figure(figsize=figsize)
    T, C_in, H, W = vtei.shape

    for t in range(T):
        frame = vtei[t, 0, :, :].cpu().numpy()  # Example: Using channel 0 for visualization
        plt.imshow(frame, cmap="gray")

        # Plot bounding boxes
        for bbox in bboxes:
            class_id, x_center, y_center, width, height = bbox
            x_center_pix = x_center * W
            y_center_pix = y_center * H
            width_pix = width * W
            height_pix = height * H
            x1 = x_center_pix - width_pix / 2
            y1 = y_center_pix - height_pix / 2

            rect = plt.Rectangle(
                (x1, y1), width_pix, height_pix, linewidth=2, edgecolor="red", facecolor="none"
            )
            plt.gca().add_patch(rect)
            plt.text(
                x1,
                y1 - 10,
                f"{class_names[int(class_id)]}",
                color="yellow",
                fontsize=8,
                backgroundcolor="black",
            )
        plt.title(f"Frame {t + 1}")
        plt.pause(0.5)  # Pause to simulate continuous playback
        plt.clf()  # Clear figure for the next frame

# -----------------------------------------------------------------------------
# Testing Loop with Visualization
# -----------------------------------------------------------------------------
def test_model():
    # Load test dataset
    test_dataset = PEDRoDataset(DATA_DIR, split="test", H=H, W=W, C_in=C_in, T=T)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
    )

    # Load model and weights
    model = RecurrentYOLOv10(input_channels=C_in, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load("recurrentyolov10_best.pth"))
    model.eval()

    class_names = ["Class 0", "Class 1"]  # Update with actual class names if available

    with torch.no_grad():
        for batch_idx, (vtei, labels) in enumerate(test_loader):
            vtei = vtei.to(DEVICE)

            # Forward pass
            one_to_one_output, one_to_many_output = model(vtei)

            # Decode predictions
            detections = decode_predictions(
                one_to_many_output[:, 1:NUM_CLASSES + 1, :, :],
                one_to_many_output[:, NUM_CLASSES + 1:, :, :],
                NUM_CLASSES,
            )

            print(f"Batch {batch_idx}: Detections = {detections}")

            # Visualize predictions frame by frame
            plot_vtei_with_bboxes(vtei.cpu(), labels, class_names)

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Test...")
    test_model()