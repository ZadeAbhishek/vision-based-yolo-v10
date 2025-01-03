import os
import sys  # Import sys to use sys.stdout in logging
import numpy as np
import torch
from torch.utils.data import Dataset
from vtei import generate_vtei_with_rps  # Ensure this module is available in your PYTHONPATH
import logging

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more details, or INFO to reduce verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Logs go to stdout
    ]
)

class PEDRoDataset(Dataset):
    """
    A PyTorch Dataset to load event data from .npy files and corresponding YOLO labels.

    Note: This code is device-agnostic. For Apple Silicon (MPS), no special changes
    are strictly required here. The 'mps' device usage is typically handled in 
    your main script or DataLoader setup (e.g., num_workers=0, pin_memory=False).

    Assumes the following directory structure:
        data_dir/numpy/<split>/*.npy  (event data)
        data_dir/yolo/<split>/*.txt   (YOLO labels)
        data_dir/<split>.txt          (list of sample names)

    Args:
        data_dir (str): Path to the root PEDRo dataset directory.
        split (str): Which subset to load: 'train', 'val', or 'test'.
        H (int): Height of the event frame (e.g., 256).
        W (int): Width of the event frame (e.g., 256).
        C_in (int): Number of temporal bins (channels) for the VTEI (e.g., 5).
        T (int): Temporal sequence length (number of frames per sample).
        transform (callable, optional): Optional transform to apply to (vtei, labels).
        suppress_prob (float): Probability used by `generate_vtei_with_rps`.
        pos_prob (float): Probability used by `generate_vtei_with_rps`.
    """

    def __init__(
        self,
        data_dir,
        split="train",
        H=256,
        W=256,
        C_in=5,
        T=5,
        transform=None,
        suppress_prob=0.1,
        pos_prob=0.5
    ):
        self.event_dir = os.path.join(data_dir, "numpy", split)
        self.label_dir = os.path.join(data_dir, "yolo", split)
        self.split_file = os.path.join(data_dir, f"{split}.txt")

        self.H, self.W, self.C_in, self.T = H, W, C_in, T
        self.transform = transform
        self.suppress_prob = suppress_prob
        self.pos_prob = pos_prob

        # Verify existence of event/label directories
        if not os.path.isdir(self.event_dir):
            logging.error(f"Event directory {self.event_dir} does not exist.")
            raise FileNotFoundError(f"Event directory {self.event_dir} does not exist.")
        
        if not os.path.isdir(self.label_dir):
            logging.error(f"Label directory {self.label_dir} does not exist.")
            raise FileNotFoundError(f"Label directory {self.label_dir} does not exist.")

        # Read the list of sample names from <split>.txt
        if not os.path.exists(self.split_file):
            logging.error(f"Split file {self.split_file} does not exist.")
            raise FileNotFoundError(f"Split file {self.split_file} does not exist.")
        
        with open(self.split_file, "r") as f:
            self.samples = [line.strip() for line in f if line.strip()]
        
        if not self.samples:
            logging.error(f"No samples found in split file {self.split_file}.")
            raise ValueError(f"No samples found in split file {self.split_file}.")

        logging.info(f"Loaded {len(self.samples)} samples from {self.split_file}.")

        # (Optional) Check label files for a few samples
        sample_to_check = min(len(self.samples), 5)
        for i in range(sample_to_check):
            sample_name = self.samples[i]
            label_path = os.path.join(self.label_dir, f"{sample_name}.txt")
            if os.path.exists(label_path):
                logging.debug(f"Label file exists: {label_path}")
            else:
                logging.warning(f"Label file missing: {label_path}")

    def __len__(self):
        # Adjust length to account for temporal sequences of length T
        length = max(0, len(self.samples) - self.T + 1)
        logging.debug(f"Dataset length (number of sequences): {length}")
        return length

    def __getitem__(self, idx):
        """
        Returns:
            vtei (torch.Tensor): Shape [T, C_in, H, W].
            labels (list of [float]): YOLO labels for the last frame in the sequence.
        """
        if idx < 0 or idx >= len(self):
            logging.error(f"Index {idx} out of range for dataset of length {len(self)}.")
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}.")

        logging.debug(f"Fetching sequence starting at index {idx}.")

        # Gather T consecutive samples
        sequence_samples = self.samples[idx : idx + self.T]

        vteis = []
        labels_seq = []

        for t_idx, sample_name in enumerate(sequence_samples):
            logging.debug(f"Processing sample {idx + t_idx}: {sample_name}")

            # Load event data
            event_path = os.path.join(self.event_dir, f"{sample_name}.npy")
            if not os.path.exists(event_path):
                logging.error(f"Event file {event_path} does not exist.")
                raise FileNotFoundError(f"Event file {event_path} does not exist.")

            try:
                events = np.load(event_path)  # shape (N,4) => [t, x, y, p]
                logging.debug(f"Loaded events from {event_path} with shape {events.shape}.")
            except Exception as e:
                logging.error(f"Failed to load events from {event_path}: {e}")
                raise e

            if events.ndim != 2 or events.shape[1] != 4:
                logging.error(f"Invalid event data shape in {event_path}: {events.shape}")
                raise ValueError(f"Invalid event data shape in {event_path}: {events.shape}")

            # Separate columns
            t = events[:, 0]
            x = events[:, 1].astype(int)
            y = events[:, 2].astype(int)
            p = events[:, 3]
            polarity = p.astype(int)

            # Generate VTEI
            t0, tN = t.min(), t.max()
            logging.debug(f"Timestamps range from {t0} to {tN}.")

            # Convert events to list of tuples for generate_vtei_with_rps
            event_tuples = list(zip(x, y, polarity, t))
            try:
                vtei = generate_vtei_with_rps(
                    events=event_tuples,
                    H=self.H,
                    W=self.W,
                    B=self.C_in,
                    t0=t0,
                    tN=tN,
                    suppress_prob=self.suppress_prob,
                    pos_prob=self.pos_prob
                )
                logging.debug(f"Generated VTEI with shape {vtei.shape}.")
            except Exception as e:
                logging.error(f"Failed to generate VTEI for {sample_name}: {e}")
                raise e

            if vtei.shape != (self.C_in, self.H, self.W):
                logging.error(
                    f"Generated VTEI has incorrect shape: {vtei.shape}, expected ({self.C_in}, {self.H}, {self.W})"
                )
                raise ValueError(
                    f"Generated VTEI has incorrect shape: {vtei.shape}, "
                    f"expected ({self.C_in}, {self.H}, {self.W})"
                )

            # Load YOLO labels
            label_path = os.path.join(self.label_dir, f"{sample_name}.txt")
            label = self._load_yolo_labels(label_path)
            logging.debug(f"Loaded {len(label)} bounding boxes from {label_path}.")

            # Convert VTEI to torch.Tensor if needed
            if not isinstance(vtei, torch.Tensor):
                vtei = torch.tensor(vtei, dtype=torch.float32)

            vteis.append(vtei)
            labels_seq.append(label)

        # Stack into a single tensor: [T, C_in, H, W]
        vteis = torch.stack(vteis)
        logging.debug(f"Stacked VTEI tensor shape: {vteis.shape}")

        # Take labels from the last frame in the sequence
        labels = labels_seq[-1]
        logging.debug(f"Labels for the last frame: {labels}")

        # Optional transform
        if self.transform:
            logging.debug("Applying transformations.")
            try:
                vteis, labels = self.transform(vteis, labels)
                logging.debug("Transformations applied.")
            except Exception as e:
                logging.error(f"Failed to apply transformations: {e}")
                raise e

        return vteis, torch.tensor(labels, dtype=torch.float32)

    def _load_yolo_labels(self, label_path):
        """
        Loads YOLO-format labels from a .txt file, each line:
            class_id x_center y_center width height
        Returns a list of bounding boxes, each box is a list of floats.
        """
        boxes = []
        if os.path.exists(label_path):
            try:
                with open(label_path, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        vals = line.strip().split()
                        if len(vals) == 5:  # Expect exactly 5 values
                            try:
                                box = [float(val) for val in vals]
                                boxes.append(box)
                            except ValueError:
                                logging.warning(
                                    f"Non-float values in label file {label_path} at line {line_num}. "
                                    f"Line: {line.strip()}"
                                )
                        else:
                            logging.warning(
                                f"Label line in {label_path} at line {line_num} has {len(vals)} values, "
                                f"expected 5."
                            )
            except Exception as e:
                logging.error(f"Failed to read label file {label_path}: {e}")
        else:
            logging.warning(f"Label file {label_path} does not exist.")
        return boxes

    def visualize_sample(self, idx):
        """
        Visualizes a single sample from the dataset.
        """
        import matplotlib.pyplot as plt

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}.")

        vtei, labels = self.__getitem__(idx)

        # Plot each channel across time
        T = vtei.shape[0]
        C_in = vtei.shape[1]
        fig, axs = plt.subplots(1, C_in, figsize=(15, 5))
        fig.suptitle(f"Sample {idx} - Temporal Sequence")
        for c in range(C_in):
            # Summation across time for visualization
            img = vtei[:, c, :, :].sum(dim=0).cpu().numpy()
            axs[c].imshow(img, cmap='gray')
            axs[c].set_title(f"Channel {c+1}")
            axs[c].axis('off')
        plt.show()

        # Plot bounding boxes on the last frame (channel 0 as example)
        last_frame = vtei[-1, 0, :, :].cpu().numpy()
        fig, ax = plt.subplots(1)
        ax.imshow(last_frame, cmap='gray')
        for box in labels:
            class_id, x_center, y_center, w, h = box
            # Convert normalized YOLO coords to pixel coords
            x_center_pix = x_center * self.W
            y_center_pix = y_center * self.H
            w_pix = w * self.W
            h_pix = h * self.H
            x1 = x_center_pix - (w_pix / 2)
            y1 = y_center_pix - (h_pix / 2)
            rect = plt.Rectangle((x1, y1), w_pix, h_pix, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"Class {int(class_id)}", color='yellow', fontsize=12, backgroundcolor='black')
        ax.set_title("Bounding Boxes on Last Frame")
        ax.axis('off')
        plt.show()