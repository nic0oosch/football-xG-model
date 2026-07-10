import numpy as np
import torch 
import torch.nn as nn

# Constants
CNN_OUT_DIM = 32 * 8 * 6    # = 1536

# helpers
def _build_cnn_backbone() -> nn.Sequential:
    return nn.Sequential(
        # Block 1 : Input Layer (3, 68, 52)
        nn.Conv2d(3, 16, kernel_size=3, padding=1), 
        nn.BatchNorm2d(16), nn.ReLU(),  nn.MaxPool2d(2),

        # Block 2
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32), nn.ReLU(),  nn.MaxPool2d(2),  # -> (32, 17, 13)

        # Block 3
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32), nn.ReLU(),  nn.MaxPool2d(2),  # -> (32, 8, 6)
    )


# CNN model for heatmap only
class XGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = _build_cnn_backbone()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CNN_OUT_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.conv(x)).squeeze(1)  

# hybrid model for CNN + XGBoost (tabular features)
class HybridXGNet(nn.Module):
    def __init__(self, num_tabular_features):
        super().__init__()
        
        # first branch: CNN for heatmap (3, 68, 52)
        self.cnn_branch = _build_cnn_backbone()  # output shape: (batch_size, 32, 8, 6)

        # second branch: tabular features
        self.tabular_branch = nn.Sequential(
            nn.Linear(num_tabular_features, 32),
            nn.ReLU(),
            nn.Dropout(0.5) 
        )

        # combined for classification
        self.classifier = nn.Sequential(
            nn.Linear(CNN_OUT_DIM + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
    def forward(self, x_heatmap, x_tabular):
        # 1. extract CNN-Features and flatten
        feat_cnn = self.cnn_branch(x_heatmap)
        feat_cnn = feat_cnn.flatten(start_dim=1)  # (batch_size, 32*8*6) 

        # 2. extract tabular features
        feat_tabular = self.tabular_branch(x_tabular)  # (batch_size, 32)

        # 3. concatenate features
        feat_combined = torch.cat([feat_cnn, feat_tabular], dim=1)  

        # 4. prediction / classification
        return self.classifier(feat_combined).squeeze(1)
    
# embedding model for CNN + XGBoost (tabular features)
class XGNetEmbedder(nn.Module):
    """
    XGNet without final classification layer, used as feature extractor for stacking with XGBoost.
    Output: 64-dimensional embedding per shot.
    """
    def __init__(self, trained_xgnet: XGNet):
        super().__init__()
        self.cnn = trained_xgnet.conv
        # all without linear layer (64, 1)
        self.embedder = nn.Sequential(
            *list(trained_xgnet.classifier.children())[:-1]
        )
    
    def forward(self, x):
        return self.embedder(self.cnn(x)) # -> (batch_size, 64)