from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd


def train_baseline(train: pd.DataFrame) -> Pipeline:
    """Logistic Regression on distance + angle only"""
    X = train[['distance', 'angle']].values
    y = train['is_goal'].values

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X, y)
    return pipeline