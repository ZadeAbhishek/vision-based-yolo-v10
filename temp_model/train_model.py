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

# Configuration
DATA_DIR = "data/PEDRo/"
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.01
H, W, C_IN = 256, 256, 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Transformation (resize input to fit YOLO requirements)
transform = Compose([
    Resize((320, 320))  # Resize to match YOLO's expected input size
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
    num_workers=0,
    collate_fn=custom_collate_fn
)

# Initialize YOLO model
model = YOLO("yolov10n.yaml")  # Ensure you have a valid yolov10n.yaml file
model.model.to(DEVICE)

# Optimizer
optimizer = torch.optim.Adam(model.model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    model.model.train()
    for i, (images, targets) in enumerate(train_loader):
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

        print(f"Epoch {epoch + 1}/{EPOCHS}, Step {i + 1}/{len(train_loader)}, Loss: {loss.item()}")

# Save the Model
model.save("custom_yolov10n_pedro.pt")