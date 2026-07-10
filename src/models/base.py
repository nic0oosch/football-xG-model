import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# constants

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