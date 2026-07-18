import torch
import random
import numpy as np
from torch.utils.data import Dataset

class ShotDataset(Dataset):
    """
    PyTorch Dataset for shot data.
    Handles tensor conversion in one place, so that the DataLoader can be used directly.
    """
    def __init__(self, df, tabular_features=None, is_train=False, y_cols_to_flip=None):
        # Convert heatmaps to tensors
        self.heatmaps = torch.tensor(np.stack(df['heatmap'].values), dtype=torch.float32)
        self.labels = torch.tensor(df['is_goal'].values, dtype=torch.float32)

        if tabular_features is not None:
            self.tabular_data = torch.tensor(tabular_features, dtype=torch.float32)
        else:
            self.tabular_data = None

        self.is_train = is_train
        self.y_cols_to_flip = y_cols_to_flip if y_cols_to_flip is not None else []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        heatmap = self.heatmaps[idx]

        if self.tabular_data is not None:
            tab = self.tabular_data[idx]

            if self.is_train and random.random() < 0.5:
                # Flip heatmap horizontally
                heatmap = torch.flip(heatmap, dims=[1])  # flip width dimension

                tab = tab.clone()
                # Flip tabular features if specified
                for col_idx in self.y_cols_to_flip:
                    # Flip y coordinate (statsbomb field width is 80)
                    tab[col_idx] = 80.0 - tab[col_idx]  
                
            return heatmap, tab, self.labels[idx]
        
        else:
            if self.is_train and random.random() < 0.5:
                # Flip heatmap horizontally
                heatmap = torch.flip(heatmap, dims=[1]) 

            return heatmap, self.labels[idx]