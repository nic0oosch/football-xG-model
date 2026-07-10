import os
import warnings
import pandas as pd
import numpy as np
from statsbombpy import sb
from features import build_feature_df
import hashlib, json

# custom typing
type SEASON_ID = int
type COMPETITION_ID = int

# Suppress StatsBomb NoAuthWarning for free open data access
warnings.filterwarnings('ignore', message='credentials were not supplied')

# helpers
def _competition_hash(competitions) -> str:
    """Return a hash of the competitions set for cache validation"""
    key = sorted(competitions)
    return hashlib.md5(json.dumps(key).encode()).hexdigest()[:8]


# Constants
COMPETITIONS = {
    # a competition is a tuple of (competition_id, season_id)
    (55, 282),  # EURO 2024
    (55, 43),   # EURO 2020
    (9, 281),   # 1. Bundesliga 2023/2024
    (9, 27),    # 1. Bundesliga 2015/2016
    (11, 1),    # La Liga 2017/2018
    (11, 4),    # La Liga 2018/2019
    (11, 42),   # La Liga 2019/2020
    (11, 90),   # La Liga 2020/2021
    (1267, 107), # African Cup of Nations 2023
    (16, 2),    # Champions League 2016/2017
    (16, 1),    # Champions League 2017/2018
    (16, 4),    # Champions League 2018/2019
    (43, 3),    # FIFA World Cup 2018
    (43, 106),  # FIFA World Cup 2022
}

RAW_CACHE = f"data/processed/shots_raw_{_competition_hash(COMPETITIONS)}.parquet"
FEAT_CACHE = "data/processed/shots_features.parquet"
HEATMAP_CACHE = "data/processed/heatmaps.npy"


# load raw data
def load_raw_shots(competitions: set[tuple[COMPETITION_ID, SEASON_ID]] = COMPETITIONS) -> pd.DataFrame:
    if os.path.exists(RAW_CACHE):
        print("Loading existing data from cache...")
        return pd.read_parquet(RAW_CACHE)

    print("Loading statsbomb API...")
    all_shots = []
    for competition_id, season_id in competitions:
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        for match in matches.match_id:
            events = sb.events(match_id=match)
            all_shots.append(events[events['type'] == 'Shot'])

    dataframe = pd.concat(all_shots, ignore_index=True)
    os.makedirs("data/processed", exist_ok=True)
    dataframe.to_parquet(RAW_CACHE)
    return dataframe

# build features
def load_features() -> pd.DataFrame:
    if os.path.exists(FEAT_CACHE) and os.path.exists(HEATMAP_CACHE):
        print("Loading existing features from cache...")
        features_df = pd.read_parquet(FEAT_CACHE)
        heatmaps = np.load(HEATMAP_CACHE)           # shape (n, 3, 68, 52)
        features_df['heatmap'] = list(heatmaps)     # reattach as column
        return features_df

    shots_df = load_raw_shots()
    print("Building features...")
    features_df = build_feature_df(shots_df)

    # save heatmaps as npy
    heatmaps = np.stack(features_df['heatmap'].values)  # (n, 3, 68, 52)
    np.save(HEATMAP_CACHE, heatmaps)

    # extract parquet for features without heatmap
    features_df.drop(columns=['heatmap']).to_parquet(FEAT_CACHE)

    return features_df

# Split dataset -> Train | Validate | Test
def split_data(features_df: pd.DataFrame):
    """
    Split dataset in Train, Validate and Test set.
    This function splits the dataset by match, not only by index.
    I.e.: complete matches are allocated to one split
    """
    match_ids = features_df['match_id'].unique()
    np.random.seed(42)
    np.random.shuffle(match_ids)

    n = len(match_ids)
    train_ids = match_ids[:int(n * 0.7)]
    val_ids = match_ids[int(n * 0.7):int(n * 0.85)]
    test_ids = match_ids[int(n * 0.85):]

    train = features_df[features_df['match_id'].isin(train_ids)]
    val = features_df[features_df['match_id'].isin(val_ids)]
    test = features_df[features_df['match_id'].isin(test_ids)]

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test
