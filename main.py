import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd


class UnifiedSkeletonDataset(Dataset):
    """
    PyTorch Dataset for loading unified skeleton data from .npz files.

    Supports both 'original' and 'shoulder_centered' coordinate systems.
    Each sample contains:
        - Landmark coordinates: (num_frames, num_landmarks, 4) where 4 = [x, y, z, visibility]
        - Adjacency matrix: (num_landmarks, num_landmarks)
        - Label: Integer class index for the gloss
    """

    def __init__(
        self,
        data_root: str,
        gloss_mapping_file: str,
        coordinate_system: str = "shoulder_centered",
        min_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        split: Optional[str] = None,
        split_file: Optional[str] = None,
    ):
        """
        Initialize the dataset loader.

        Args:
            data_root: Root directory containing gloss subdirectories with .npz files
            gloss_mapping_file: Path to CSV or JSON file mapping glosses to labels
            coordinate_system: Either 'original' or 'shoulder_centered'
            min_frames: Minimum number of frames required (filter out shorter videos)
            max_frames: Maximum number of frames to use (truncate longer videos)
            split: Optional split name ('train', 'val', 'test')
            split_file: Optional path to file containing video IDs for this split
        """
        self.data_root = Path(data_root)
        self.coordinate_system = coordinate_system
        self.min_frames = min_frames
        self.max_frames = max_frames

        # Load gloss to label mapping
        self.gloss_to_label, self.label_to_gloss = self._load_gloss_mapping(gloss_mapping_file)
        self.num_classes = len(self.gloss_to_label)

        # Find all npz files
        suffix = f"_{coordinate_system}" if coordinate_system != "original" else ""
        all_npz_files = list(self.data_root.rglob(f"*{suffix}_landmarks.npz"))

        self.npz_files = []
        glosses_in_mapping = set(self.gloss_to_label.keys())

        for npz_file in all_npz_files:
            # Extract gloss from parent directory name
            gloss = npz_file.parent.name
            if gloss in glosses_in_mapping:
                self.npz_files.append(npz_file)

        # Load and cache metadata
        self._load_metadata()

        print(f"Loaded {len(self.npz_files)} samples from {self.data_root}")
        print(f"Coordinate system: {coordinate_system}")
        print(f"Number of classes: {self.num_classes}")

    def _load_gloss_mapping(self, mapping_file: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Load mapping from gloss names to integer labels."""
        mapping_path = Path(mapping_file)

        df = pd.read_csv(mapping_path)
        if 'label' in df.columns:
            gloss_to_label = dict(zip(df['gloss'], df['label']))
        else:
            # Create labels from unique glosses
            unique_glosses = sorted(df['gloss'].unique())
            gloss_to_label = {gloss: idx for idx, gloss in enumerate(unique_glosses)}

        label_to_gloss = {v: k for k, v in gloss_to_label.items()}
        return gloss_to_label, label_to_gloss

    def _load_metadata(self):
        """Load metadata from all npz files to enable filtering."""
        valid_files = []

        for npz_file in self.npz_files:
            try:
                data = np.load(npz_file)
                num_frames = data['num_frames'][0] if 'num_frames' in data else len(data['unified_graph'])

                # Apply frame filters
                if self.min_frames and num_frames < self.min_frames:
                    continue

                valid_files.append(npz_file)
                data.close()
            except Exception as e:
                print(f"Warning: Could not load {npz_file}: {e}")

        self.npz_files = valid_files

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.npz_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Returns:
            Dictionary containing:
                - 'landmarks': Tensor of shape (num_frames, num_landmarks, 4)
                - 'adjacency': Tensor of shape (num_landmarks, num_landmarks)
                - 'label': Integer label for the gloss
                - 'num_frames': Number of frames in this sample
                - 'gloss': String name of the gloss
                - 'video_id': String video ID
        """
        npz_file = self.npz_files[idx]

        # Load data
        data = np.load(npz_file)

        # Extract gloss from parent directory name
        gloss = npz_file.parent.name
        label = self.gloss_to_label.get(gloss, -1)

        if self.coordinate_system == "shoulder_centered":
            # Unified graph format
            landmarks = data['unified_graph']  # (num_frames, total_landmarks, 4)
            adjacency = data['adjacency_matrix']  # (total_landmarks, total_landmarks)
        else:
            # for now
            raise Exception

        num_frames = landmarks.shape[0]

        # Truncate if needed
        if self.max_frames and num_frames > self.max_frames:
            landmarks = landmarks[:self.max_frames]
            num_frames = self.max_frames

        data.close()

        # Convert to torch tensors
        return {
            'landmarks': torch.from_numpy(landmarks).float(),
            'adjacency': torch.from_numpy(adjacency).float(),
            'label': torch.tensor(label, dtype=torch.long),
            'num_frames': torch.tensor(num_frames, dtype=torch.long),
        }


def collate_fn_pad_sequences(batch: List[Dict], max_frames: int = 150) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader that pads sequences to a fixed length.

    Args:
        batch: List of samples from __getitem__
        max_frames: Fixed maximum number of frames (default 150)

    Returns:
        Batched dictionary with padded sequences and masks
    """
    # Get dimensions
    num_landmarks = batch[0]['landmarks'].shape[1]
    num_features = batch[0]['landmarks'].shape[2]
    batch_size = len(batch)

    # Initialize padded tensors (all sequences padded to max_frames)
    landmarks_padded = torch.zeros(batch_size, max_frames, num_landmarks, num_features)
    masks = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    adjacency_matrices = torch.stack([sample['adjacency'] for sample in batch])
    labels = torch.stack([sample['label'] for sample in batch])
    num_frames = torch.stack([sample['num_frames'] for sample in batch])

    # Fill in actual data
    for i, sample in enumerate(batch):
        seq_len = sample['num_frames'].item()
        seq_len = min(seq_len, max_frames)  # Ensure we don't exceed max_frames
        landmarks_padded[i, :seq_len] = sample['landmarks'][:seq_len]
        masks[i, :seq_len] = True

    return {
        'landmarks': landmarks_padded,
        'adjacency': adjacency_matrices,
        'mask': masks,
        'label': labels,
        'num_frames': num_frames,
    }


def normalize_adjacency_matrix(A: torch.Tensor, num_nodes: int, device):
    A_tilde = A + torch.eye(num_nodes).to(device)
    D_tilde = torch.diag(torch.sum(A_tilde, dim=1))
    D_tilde_inv_sqrt = torch.pow(D_tilde, -0.5)
    D_tilde_inv_sqrt[torch.isinf(D_tilde_inv_sqrt)] = 0. # Handle cases of isolated nodes

    A_hat = torch.mm(torch.mm(D_tilde_inv_sqrt, A_tilde), D_tilde_inv_sqrt)

    return A_hat


def train(model, train_loader, val_loader, num_epochs, learning_rate, device, num_nodes):
    """Train the STGCN model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            landmarks = batch['landmarks'].to(device)
            adjacency = batch['adjacency'].to(device)
            labels = batch['label'].to(device)

            # Normalize adjacency matrix
            adjacency_norm = normalize_adjacency_matrix(adjacency[0], num_nodes, device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(landmarks, adjacency_norm)
            outputs_avg = outputs.mean(dim=1)

            # Loss and backward
            loss = criterion(outputs_avg, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = outputs_avg.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                landmarks = batch['landmarks'].to(device)
                adjacency = batch['adjacency'].to(device)
                labels = batch['label'].to(device)

                adjacency_norm = normalize_adjacency_matrix(adjacency[0], num_nodes, device)

                outputs = model(landmarks, adjacency_norm)
                outputs_avg = outputs.mean(dim=1)

                loss = criterion(outputs_avg, labels)

                val_loss += loss.item()
                _, predicted = outputs_avg.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        # Print stats
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  --> Best model saved! Val Acc: {val_acc:.2f}%")

    print(f"\nTraining complete! Best val acc: {best_val_acc:.2f}%")
    return best_val_acc


if __name__ == "__main__":
    from src.stgcn import STGCN

    # --- Configuration ---
    GLOSS_MAPPING_FILE = "data/gloss_map.csv"
    DATA_ROOT = "/scratch/izar/nowak/gesture/data/wlaslvideos_processed"
    COORDINATE_SYSTEM = "shoulder_centered"
    MIN_FRAMES = 0
    MAX_FRAMES = 300
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")

    # --- Create dataset ---
    print("\nLoading dataset...")
    dataset = UnifiedSkeletonDataset(
        data_root=DATA_ROOT,
        gloss_mapping_file=GLOSS_MAPPING_FILE,
        coordinate_system=COORDINATE_SYSTEM,
        min_frames=MIN_FRAMES,
        max_frames=MAX_FRAMES,
    )

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create dataloaders with fixed max_frames
    from functools import partial
    collate_fn = partial(collate_fn_pad_sequences, max_frames=MAX_FRAMES)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1,
    )

    # Get model parameters from data
    sample_batch = next(iter(train_loader))
    num_nodes = sample_batch['landmarks'].shape[2]
    num_features = sample_batch['landmarks'].shape[3]
    # Use MAX_FRAMES as fixed timesteps (all batches will be padded to this)
    num_timesteps = MAX_FRAMES
    num_classes = dataset.num_classes

    print("\nModel parameters:")
    print(f"  Num nodes: {num_nodes}, Num features: {num_features}")
    print(f"  Num timesteps (fixed): {num_timesteps}, Num classes: {num_classes}")

    # Create model
    model = STGCN(
        num_nodes=num_nodes,
        num_features=num_features,
        num_timesteps=num_timesteps,
        out_features=num_classes,
    ).to(DEVICE)

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Train
    train(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, DEVICE, num_nodes)