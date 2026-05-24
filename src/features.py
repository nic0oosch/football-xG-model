import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

GOAL_X, GOAL_Y = 120, 40

def calc_distance(x, y):
    return np.sqrt((x - GOAL_X)**2 + (y - GOAL_Y)**2)

def calc_angle(x, y):
    return np.arctan2(abs(y - GOAL_Y), abs(x - GOAL_X))

def build_heatmap(freeze_frame, shooter_location, grid_size=(68, 52), blur=True):
    """
    Channel 0: Teammates
    Channel 1: Opponents
    Channel 2: Shooter
    """
    h, w = grid_size
    heatmap = np.zeros((3, h, w))

    def to_grid(x, y):
        gx = int(np.clip((x - 60) / 60 * w, 0, w - 1))
        gy = int(np.clip(y / 80 * h, 0, h - 1))
        return gy, gx

    for player in freeze_frame:
        gy, gx = to_grid(*player['location'])
        channel = 0 if player['teammate'] else 1
        heatmap[channel, gy, gx] = 1.0

    gy, gx = to_grid(*shooter_location)
    heatmap[2, gy, gx] = 1.0

    if blur:
        for c in range(3):
            heatmap[c] = gaussian_filter(heatmap[c], sigma=1)

    return heatmap

def get_label(shot) -> int:
    return 1 if shot['shot_outcome'] == 'Goal' else 0

def extract_features(shot) -> dict:
    x, y = shot['location']
    return {
        'distance':  calc_distance(x, y),
        'angle':     calc_angle(x, y),
        'body_part': shot['shot_body_part'],
        'shot_type': shot['shot_type'],
        'heatmap':   build_heatmap(shot['shot_freeze_frame'], shot['location']),
        'is_goal':   get_label(shot)
    }

def build_feature_df(shots_df: pd.DataFrame) -> pd.DataFrame:
    dataframe = pd.DataFrame(shots_df.apply(extract_features, axis=1).tolist())
    dataframe['match_id'] = shots_df['match_id'].values
    return dataframe
