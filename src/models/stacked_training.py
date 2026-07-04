import numpy as np
import torch
import pandas as pd
from xgboost import XGBClassifier
from .cnn_models import XGNetEmbedder
from .base import make_preprocessor, make_X_tabular

def extract_embeddings(embedder: XGNetEmbedder, df: pd.DataFrame) -> np.ndarray:
    """
    Extract 64-dim CNN embeddings for all shots in df using the provided embedder model.
    """

    X_heatmap = torch.tensor(np.stack(df['heatmap'].values), dtype=torch.float32)
    embedder.eval()

    with torch.no_grad():
        embeddings = embedder(X_heatmap).numpy() # (n, 64)
    return embeddings


def train_stacked_model(train: pd.DataFrame, val: pd.DataFrame, trained_xgnet):
    """
    Stage 1: CNN as feature extractor
    Stage 2: XGBoost on CNN embeddings + tabular features
    """
    # Stage 1 - CNN embedding
    embedder = XGNetEmbedder(trained_xgnet)

    train_embeddings = extract_embeddings(embedder, train)
    val_embeddings = extract_embeddings(embedder, val)

    # Stage 2  - tabular features
    preprocessor = make_preprocessor()
    train_tab = preprocessor.fit_transform(make_X_tabular(train))
    val_tab = preprocessor.transform(make_X_tabular(val))

    if hasattr(train_tab, "toarray"):
        train_tab = train_tab.toarray()
        val_tab = val_tab.toarray()

    # combine embeddings and tabular features
    X_train = np.hstack([train_embeddings, train_tab])
    X_val = np.hstack([val_embeddings, val_tab])

    y_train = train['is_goal'].values
    y_val = val['is_goal'].values

    # XGBoost on combined features
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.01,
        scale_pos_weight=np.sqrt(n_neg / n_pos),
        eval_metric='logloss',
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False, 
    )

    return model, embedder, preprocessor
