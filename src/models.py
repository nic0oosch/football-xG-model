import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# Feature columns
NUMERIC_FEATURES = ['distance', 'angle', 'n_defenders_in_cone', 'goalkeeper_distance']
CATEGORICAL_FEATURES = ['body_part', 'shot_type']

# preprocessor (shared for all tabular models) 
def make_preprocessor():
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])
    return preprocessor

# feature engineering functions
def make_X_tabular(df: pd.DataFrame) -> pd.DataFrame:
    """tabular features only - suitable for logistic regression & XGBoost"""
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

def make_X_heatmap(df: pd.DataFrame) -> np.ndarray:
    """flattened heatmap only"""
    return np.stack(df['heatmap'].values).reshape(len(df), -1) # (n, 10608)

def make_X_full(df: pd.DataFrame) -> np.ndarray:
    """combine tabular + heatmap features for CNN+XGBoost"""
    # preproccessor handles scaling and encoding for tabular features
    tabular = make_preprocessor().fit_transform(make_X_tabular(df)) 
    heatmap = make_X_heatmap(df)                                     
    return np.hstack([tabular, heatmap])  


# Models

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

# later improvement: CNN on heatmap and XGBoost
