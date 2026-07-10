import torch
from torch.utils.data import Dataset
import numpy as np

class ShotDataset(Dataset):
    """
    PyTorch Dataset for shot data.
    Handles tensor conversion in one place, so that the DataLoader can be used directly.
    """
    def __init__(self, df, tabular_features=None):
        # Convert heatmaps to tensors
        self.heatmaps = torch.tensor(np.stack(df['heatmap'].values), dtype=torch.float32)
        self.labels = torch.tensor(df['is_goal'].values, dtype=torch.float32)

        if tabular_features is not None:
            self.tabular_data = torch.tensor(tabular_features, dtype=torch.float32)
        else:
            self.tabular_data = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.tabular_data is not None:
            return self.heatmaps[idx], self.tabular_data[idx], self.labels[idx]
        return self.heatmaps[idx], self.labels[idx]