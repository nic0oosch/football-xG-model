import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from .base import make_preprocessor, make_X_tabular


def train_xgboost(train: pd.DataFrame) -> Pipeline:
    """XGBoost on all tabular features"""
    X = make_X_tabular(train)
    y = train['is_goal'].values

    pipeline = Pipeline([
        ('preprocessor', make_preprocessor()),
        ('clf', XGBClassifier(
            n_estimators=100,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.01,
            eval_metric='logloss',
            random_state=42
        ))
    ])
    pipeline.fit(X, y)
    return pipeline