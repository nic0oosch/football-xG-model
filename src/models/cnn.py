import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class XGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1 : Input Layer (3, 68, 52)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),        # -> (16, 34, 26)

            # Block 2 
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),        # -> (32, 17, 13)

            # Block 3 
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)         # -> (32, 8, 6)

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 6, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.conv(x)).squeeze(1)  
    

def train_cnn(train, val, epochs=50, batch_size=64, lr=0.001, weight_decay=1e-4):
    X_train = torch.tensor(np.stack(train['heatmap'].values), dtype=torch.float32)
    y_train = torch.tensor(train['is_goal'].values, dtype=torch.float32)

    X_val = torch.tensor(np.stack(val['heatmap'].values), dtype=torch.float32)
    y_val = torch.tensor(val['is_goal'].values, dtype=torch.float32)

    # class imbalance handling
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    # prepare dataset for minibatching
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = XGNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_losses = []
    val_losses = []

    # Early Stopping 
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    patience = 7
    patience_counter = 0

    for epoch in range(epochs):
        # training loop
        model.train()
        epoch_losses = []
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        epoch_loss = np.mean(epoch_losses)
        train_losses.append(epoch_loss)

        # validation loop
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val).item()
        val_losses.append(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"--> Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f}")
            break
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    return model, train_losses, val_losses