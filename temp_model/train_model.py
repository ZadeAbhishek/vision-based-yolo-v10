import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from ultralytics import YOLO
from pedro_dataset import PEDRoDataset  # Import your dataset class
import matplotlib.pyplot as plt

# Configuration - Make all static values configurable
DATA_DIR = "data/PEDRo/"  # Path to dataset
BATCH_SIZE = 8  # Batch size for validation/testing
EPOCHS = 50  # Number of epochs for training
LEARNING_RATE = 0.01  # Learning rate for optimizer
H, W, C_IN = 256, 256, 5  # Input dimensions
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise CPU
RESIZE_DIM = (320, 320)  # Resize dimensions for input images
SAVE_OUTPUT_DIR = "output/detection_samples"  # Directory to save output samples (detections)
MODEL_SAVE_PATH = "custom_yolov10n_pedro.pt"  # Path to save the model


# Custom collate function
def custom_collate_fn(batch):
    tensors, labels = zip(*batch)

    max_t = max(tensor.shape[0] for tensor in tensors)
    padded_tensors = []
    for tensor in tensors:
        if tensor.dim() == 3:  # [T, H, W]
            tensor = tensor.unsqueeze(1)  # [T, 1, H, W]
        padded_tensor = F.pad(tensor, (0, 0, 0, 0, 0, max_t - tensor.shape[0]))
        if padded_tensor.shape[1] == 1:  # Single channel
            padded_tensor = padded_tensor.repeat(1, 3, 1, 1)  # Duplicate channels
        elif padded_tensor.shape[1] > 3:  # More than 3 channels
            padded_tensor = padded_tensor[:, :3, :, :]  # Trim to 3 channels
        padded_tensors.append(padded_tensor)
    batch_tensors = torch.cat(padded_tensors, dim=0)  # Flatten temporal dimension into batch dimension
    max_label_size = max(label.shape[0] for label in labels)
    padded_labels = []
    for label in labels:
        if label.shape[0] < max_label_size:
            padding = (0, 0, 0, max_label_size - label.shape[0])
            padded_label = F.pad(label, padding, "constant", 0)
        else:
            padded_label = label
        padded_labels.append(padded_label)
    batch_labels = torch.stack(padded_labels)
    return batch_tensors, batch_labels

# Ensure output directory exists
os.makedirs(SAVE_OUTPUT_DIR, exist_ok=True)

# Transformation (resize input to fit YOLO requirements)
transform = Compose([
    Resize(RESIZE_DIM)  # Resize to match YOLO's expected input size
])

# Dataset and DataLoader for validation and testing
val_dataset = PEDRoDataset(
    data_dir=DATA_DIR,
    split="val",
    H=H,
    W=W,
    C_in=C_IN,
    T=5,
    transform=transform
)

test_dataset = PEDRoDataset(
    data_dir=DATA_DIR,
    split="test",
    H=H,
    W=W,
    C_in=C_IN,
    T=5,
    transform=transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=custom_collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=custom_collate_fn
)

# Initialize YOLO model for training
model = YOLO("yolov10n.yaml")  # Ensure you have a valid yolov10n.yaml file
model.model.to(DEVICE)

# Optimizer
optimizer = torch.optim.Adam(model.model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    model.model.train()
    for i, (images, targets) in enumerate(val_loader):
        images = images.view(-1, 3, 640, 640).to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model.model(images)

        if 'one2one' in outputs:
            logits = outputs['one2one']
        else:
            print(f"Missing 'one2one' key in outputs at Epoch {epoch + 1}, Step {i + 1}")
            continue

        try:
            loss = 0
            for logit in logits:
                logit = logit.view(BATCH_SIZE, -1)  # Flatten logits
                target_resized = targets.view(BATCH_SIZE, -1)  # Flatten targets
                target_indices = torch.argmax(target_resized, dim=-1)  # Get class indices
                loss += F.cross_entropy(logit, target_indices)  # Calculate loss
        except Exception as e:
            print(f"Error calculating loss: {e}")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Step {i + 1}/{len(val_loader)}, Loss: {loss.item()}")

# Save the trained model
model.save(MODEL_SAVE_PATH)  # Save the model to the specified path
print(f"Model saved at {MODEL_SAVE_PATH}")

# Load the trained model for validation and testing
model = YOLO(MODEL_SAVE_PATH)  # Load the saved model
model.model.to(DEVICE)

# Validation Function
def validate(loader, model):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Disable gradient computation for validation
        for i, (images, targets) in enumerate(loader):
            images = images.view(-1, 3, 640, 640).to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model.model(images)

            if 'one2one' in outputs:
                logits = outputs['one2one']
            else:
                print(f"Missing 'one2one' key in outputs at Validation Step {i + 1}")
                continue

            try:
                loss = 0
                for logit in logits:
                    logit = logit.view(BATCH_SIZE, -1)  # Flatten logits
                    target_resized = targets.view(BATCH_SIZE, -1)  # Flatten targets
                    target_indices = torch.argmax(target_resized, dim=-1)  # Get class indices
                    loss += F.cross_entropy(logit, target_indices)  # Calculate loss
            except Exception as e:
                print(f"Error calculating loss: {e}")
                continue

            total_loss += loss.item()

        average_loss = total_loss / len(loader)
        print(f"Validation Loss: {average_loss}")
        return average_loss

# Test Function
def test(loader, model):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Disable gradient computation for testing
        for i, (images, targets) in enumerate(loader):
            images = images.view(-1, 3, 640, 640).to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model.model(images)

            if 'one2one' in outputs:
                logits = outputs['one2one']
            else:
                print(f"Missing 'one2one' key in outputs at Test Step {i + 1}")
                continue

            try:
                loss = 0
                for logit in logits:
                    logit = logit.view(BATCH_SIZE, -1)  # Flatten logits
                    target_resized = targets.view(BATCH_SIZE, -1)  # Flatten targets
                    target_indices = torch.argmax(target_resized, dim=-1)  # Get class indices
                    loss += F.cross_entropy(logit, target_indices)  # Calculate loss
            except Exception as e:
                print(f"Error calculating loss: {e}")
                continue

            total_loss += loss.item()

            # Save detection outputs as images
            if i % 10 == 0:  # Save every 10th image for example
                save_sample_output(images[0], logits[0], f"{SAVE_OUTPUT_DIR}/test_output_{i+1}.png")

        average_loss = total_loss / len(loader)
        print(f"Test Loss: {average_loss}")
        return average_loss

# Function to visualize and save sample detections
def save_sample_output(image, logits, save_path):
    """
    Visualize and save a sample output with detections.
    Arguments:
    - image: Input image tensor.
    - logits: Model output (predicted bounding boxes, scores, etc.).
    - save_path: Path to save the output image.
    """
    image = image.permute(1, 2, 0).cpu().numpy()  # Convert image to HWC format (from CHW)
    plt.imshow(image)
    plt.axis('off')

    # If you have bounding boxes in 'logits', you can overlay them here.
    # Example: Let's assume logits contain bounding box info for now:
    # Note: Adjust this according to how your output is structured.
    for box in logits:
        # Just an example, update this with actual bounding box values
        x1, y1, x2, y2 = box[:4]  # Assuming box contains coordinates
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2))

    plt.savefig(save_path)
    plt.close()

# Run validation
print("Starting Validation...")
validate(val_loader, model.model)

# Run testing
print("Starting Testing...")
test(test_loader, model.model)