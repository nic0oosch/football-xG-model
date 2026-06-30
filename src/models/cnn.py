import numpy as np
import torch
import torch.nn as nn

class XGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1 
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),        # (16, 34, 26)

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),        # (32, 17, 13)

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 17 * 13, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.conv(x)).squeeze(1)  
    

def train_cnn(train, val, epochs=30):
    X_train = torch.tensor(
        np.stack(train['heatmap'].values), dtype=torch.float32
    )
    y_train = torch.tensor(train['is_goal'].values, dtype=torch.float32)

    X_val = torch.tensor(
        np.stack(val['heatmap'].values), dtype=torch.float32
    )
    y_val = torch.tensor(val['is_goal'].values, dtype=torch.float32)

    model = XGNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        # validation loss every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val)
                val_loss = criterion(y_val_pred, y_val)
            print(f"Epoch {epoch+1:3d} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")
    return model