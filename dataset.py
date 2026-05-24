import torch
from torch.utils.data import Dataset
from pathlib import Path

class PRSNetDataset(Dataset):
    """
    Dataset wrapper for preprocessed PRS-Net tensors
    """
    def __init__(self, data_root="ModelNet40_processed", split="train"):
        root = Path(data_root)

        # Determine which .pt files to load based on split argument
        if split is None or split == "all":
            pattern = "**/*.pt"
        else:
            pattern = f"**/{split}/*.pt"
        
        self.data_paths = sorted([str(p) for p in root.glob(pattern)])
        # Check if any files were found
        if not self.data_paths:
            raise FileNotFoundError(
                f"No .pt files found in {root} with split={split!r}. "
                "Run preprocess.py first or check the data path."
            )
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        data = torch.load(self.data_paths[idx], weights_only=False)

        # DataLoader batches data into:
        #   voxels:         (B, 1, 32, 32, 32)
        #   surface_points: (B, N, 3)
        #   closest_grid:   (B, 32, 32, 32, 3)
        return data["voxels"].unsqueeze(0), data["surface_points"], data["closest_grid"]
