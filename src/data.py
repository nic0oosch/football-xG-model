import os
import pandas as pd
import numpy as np
from statsbombpy import sb
from features import build_feature_df

# Constants
RAW_CACHE = "data/processed/shots_raw.parquet"
FEAT_CACHE = "data/processed/shots_features.parquet"

EURO_2024 = 55
SEASON_2024 = 282

# load raw data
def load_raw_shots(competition_id=EURO_2024, season_id=SEASON_2024) -> pd.DataFrame:
    if os.path.exists(RAW_CACHE):
        print("Loading existing data from cache...")
        return pd.read_parquet(RAW_CACHE)

    print("Loading statsbomb API...")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    all_shots = []
    for match in matches.match_id:
        events = sb.events(match_id=match)
        all_shots.append(events[events['type'] == 'Shot'])

    dataframe = pd.concat(all_shots, ignore_index=True)
    os.makedirs("data/processed", exist_ok=True)
    dataframe.to_parquet(RAW_CACHE)
    return dataframe

# build features
def load_features() -> pd.DataFrame:
    if os.path.exists(FEAT_CACHE):
        print("Loading existing features from cache...")
        return pd.read_parquet(FEAT_CACHE)

    shots_df = load_raw_shots()
    print("Building features...")
    features_df = build_feature_df(shots_df)
    features_df.to_parquet(FEAT_CACHE)
    return features_df

# Split dataset -> Train | Validate | Test
def split_data(features_df: pd.DataFrame, shots_raw: pd.DataFrame):
    """
    Split dataset in Train, Validate and Test set.
    This function splits the dataset by match, not only by index.
    I.e.: complete matches are allocated to one split
    """
    match_ids = shots_raw['match_id'].unique()
    np.random.seed(42)
    np.random.shuffle(match_ids)

    n = len(match_ids)
    train_ids = match_ids[:int(n * 0.7)]
    val_ids = match_ids[int(n * 0.7):int(n * 0.85)]
    test_ids = match_ids[int(n * 0.85):]

    train = features_df[shots_raw['match_id'].isin(train_ids)]
    val = features_df[shots_raw['match_id'].isin(val_ids)]
    test = features_df[shots_raw['match_id'].isin(test_ids)]

    return train, val, test



