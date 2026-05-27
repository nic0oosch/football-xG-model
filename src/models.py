from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def make_X(df: pd.DataFrame) -> np.ndarray:
    heatmap  = np.stack(df['heatmap'].values)   # (n, 3, 68, 52)
    heat_flat = heatmap.reshape(len(df), -1)     # (n, 10608)
    tabular  = np.vstack([df['distance'], df['angle']]).T
    return np.hstack([tabular, heat_flat])

def train_baseline(train: pd.DataFrame) -> Pipeline:
    X = make_X(train)
    y = train['is_goal'].values

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X, y)
    return pipeline

# later improvement: CNN on heatmap and XGBoost
