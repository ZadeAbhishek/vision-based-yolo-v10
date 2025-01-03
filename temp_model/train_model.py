import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from ultralytics import YOLO
from pedro_dataset import PEDRoDataset  # Import your dataset class


# Custom collate function
def custom_collate_fn(batch):
    tensors, labels = zip(*batch)

    # Find the maximum temporal size in the batch
    max_t = max(tensor.shape[0] for tensor in tensors)

    padded_tensors = []
    for tensor in tensors:
        # Ensure tensor has a channel dimension
        if tensor.dim() == 3:  # [T, H, W]
            tensor = tensor.unsqueeze(1)  # [T, 1, H, W]

        # Pad along the temporal dimension
        padded_tensor = F.pad(tensor, (0, 0, 0, 0, 0, max_t - tensor.shape[0]))  # Pad T

        # Ensure the tensor has exactly 3 channels
        if padded_tensor.shape[1] == 1:  # Single channel
            padded_tensor = padded_tensor.repeat(1, 3, 1, 1)  # Duplicate channels
        elif padded_tensor.shape[1] > 3:  # More than 3 channels
            padded_tensor = padded_tensor[:, :3, :, :]  # Trim to 3 channels

        padded_tensors.append(padded_tensor)

    # Stack tensors along batch dimension
    batch_tensors = torch.cat(padded_tensors, dim=0)  # Flatten temporal dimension into batch dimension

    # Process labels (pad to the maximum size within the batch)
    max_label_size = max(label.shape[0] for label in labels)
    padded_labels = []
    for label in labels:
        if label.shape[0] < max_label_size:
            padding = (0, 0, 0, max_label_size - label.shape[0])
            padded_label = F.pad(label, padding, "constant", 0)  # Pad along the first dimension
        else:
            padded_label = label
        padded_labels.append(padded_label)

    batch_labels = torch.stack(padded_labels)

    return batch_tensors, batch_labels


# Configuration
DATA_DIR = "data/PEDRo/"
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.01
H, W, C_IN = 256, 256, 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Transformation (resize input to fit YOLO requirements)
transform = Compose([
    Resize((640, 640))  # Resize to match YOLO's expected input size (640x640)
])

# Dataset and DataLoader
train_dataset = PEDRoDataset(
    data_dir=DATA_DIR,
    split="train",
    H=H,
    W=W,
    C_in=C_IN,
    T=5,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # Set num_workers to 0 for compatibility on macOS/Windows
    collate_fn=custom_collate_fn  # Use the custom collate function
)

# Initialize YOLO model
model = YOLO("yolov10n.yaml")  # Ensure you have a valid yolov10n.yaml file
model.model.to(DEVICE)  # Move model to the device (GPU or CPU)

# Optimizer
optimizer = torch.optim.Adam(model.model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    model.model.train()
    for i, (images, targets) in enumerate(train_loader):
        # Reshape images to match YOLO input
        images = images.view(-1, 3, 640, 640).to(DEVICE)
        targets = [target.to(DEVICE) for target in targets]

        # Forward pass
        outputs = model.model(images, targets)
        if isinstance(outputs, dict):
            loss = outputs.get('loss', None)
        else:
            loss = outputs  # If loss is directly returned

        if loss is None or not isinstance(loss, torch.Tensor):
            print(f"Loss computation failed at Epoch {epoch + 1}, Step {i + 1}")
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Step {i + 1}/{len(train_loader)}, Loss: {loss.item()}")

# Save the Model
model.save("custom_yolov10n_pedro.pt")