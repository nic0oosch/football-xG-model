import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .cnn_models import XGNet, HybridXGNet
from .cnn_dataset import ShotDataset
from .base import make_preprocessor, make_X_tabular



# helpers
class EarlyStopping:
    def __init__(self, patience=18):
        self.patience = patience
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.best_model_weights = None

    def check(self, val_loss, model):
        """Returns True if training should stop"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return self.patience_counter >= self.patience
            

    
def train_cnn(train, val, epochs=50, batch_size=64, lr=0.001, weight_decay=1e-4):
    train_dataset = ShotDataset(train, is_train=True)
    val_dataset = ShotDataset(val, is_train=False)

    # class imbalance handling
    n_neg = (train_dataset.labels == 0).sum()
    n_pos = (train_dataset.labels == 1).sum()
    pos_weight = torch.tensor([np.sqrt(n_neg / n_pos)], dtype=torch.float32)

    # prepare dataloader and model
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = XGNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping()

    for epoch in range(epochs):
        # training loop
        model.train()
        epoch_losses = []
        for X_batch, y_batch in train_loader:
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
        val_batch_losses = []
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                y_val_pred = model(X_val_batch)
                val_batch_loss = criterion(y_val_pred, y_val_batch)
                val_batch_losses.append(val_batch_loss.item())
        
        val_loss = np.mean(val_batch_losses)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Early Stopping Check
        if early_stopping.check(val_loss, model):
            print(f"--> Early Stopping in Epoche {epoch+1}. Best Val Loss: {early_stopping.best_val_loss:.4f}")
            break
    
    # Load best model weights
    model.load_state_dict(early_stopping.best_model_weights)
    return model, train_losses, val_losses



def train_hybrid_cnn(train, val, epochs=50, batch_size=64, lr=0.001, weight_decay=1e-4, y_cols_to_flip=None):
    # 1. prepare tabular features to prevent data leakage
    preprocessor = make_preprocessor()
    
    X_train_tab = preprocessor.fit_transform(make_X_tabular(train))
    X_val_tab = preprocessor.transform(make_X_tabular(val))

    if hasattr(X_train_tab, "toarray"):
        X_train_tab = X_train_tab.toarray()
        X_val_tab = X_val_tab.toarray()
    
    num_tabular_features = X_train_tab.shape[1]

    train_dataset = ShotDataset(train, tabular_features=X_train_tab, is_train=True, y_cols_to_flip=y_cols_to_flip)
    val_dataset = ShotDataset(val, tabular_features=X_val_tab, is_train=False)

    # class imbalance handling
    n_neg = (train_dataset.labels == 0).sum()
    n_pos = (train_dataset.labels == 1).sum()
    pos_weight = torch.tensor([np.sqrt(n_neg / n_pos)], dtype=torch.float32)

    # dataloader and model optimizer
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # model initialization
    model = HybridXGNet(num_tabular_features=num_tabular_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping()

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        # extract inputs of batches
        for X_img_batch, X_tab_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # feed input batches into model
            y_pred = model(X_img_batch, X_tab_batch)

            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        epoch_loss = np.mean(epoch_losses)
        train_losses.append(epoch_loss)

        # Validation using val_loader
        model.eval()
        val_epoch_losses = []
        with torch.no_grad():
            for X_val_batch, X_val_tab_batch, y_val_batch in val_loader:
                y_val_pred = model(X_val_batch, X_val_tab_batch)
                val_batch_loss = criterion(y_val_pred, y_val_batch)
                val_epoch_losses.append(val_batch_loss.item())
        val_loss = np.mean(val_epoch_losses)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 5 == 0: 
            print(f"Epoch {epoch+1:3d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        # early stopping check
        if early_stopping.check(val_loss, model):
            print(f"--> Early Stopping in Epoche {epoch+1}. Best Val Loss: {early_stopping.best_val_loss:.4f}")
            break
    
    model.load_state_dict(early_stopping.best_model_weights)
    return model, train_losses, val_losses, preprocessor