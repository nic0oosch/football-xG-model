import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

GOAL_X, GOAL_Y = 120, 40

def calc_distance(x, y):
    return np.sqrt((x - GOAL_X)**2 + (y - GOAL_Y)**2)

def calc_angle(x, y):
    return np.arctan2(abs(y - GOAL_Y), abs(x - GOAL_X))

def count_defenders_in_cone(freeze_frame, shooter_location):
    """ Count opponents between shooter and goal with a rough cone"""
    shooter_x, shooter_y = shooter_location
    count = 0
    for player in freeze_frame:
        if player['teammate']:
            continue
        opp_x, opp_y = player['location']
        # only select oppenents nearer to the goel than the shooter
        if opp_x > shooter_x:
            if abs(opp_y - GOAL_Y) < abs(shooter_y - GOAL_Y) + 3:  # rough cone condition
                count += 1
    return count

def get_goalkeeper_distance(freeze_frame):
    """ Get distance from Goalie to goal """
    for player in freeze_frame:
        if not player['teammate'] and player.get('keeper', False):
            goalie_x, goalie_y = player['location']
            return calc_distance(goalie_x, goalie_y)
    return 0.0 # no goalkeeper found

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

    if freeze_frame is None:
        return heatmap

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
    freeze_frame = shot['shot_freeze_frame']
    return {
        'distance':  calc_distance(x, y),
        'angle':     calc_angle(x, y),
        'body_part': shot['shot_body_part'],
        'shot_type': shot['shot_type'],
        'statsbomb_xg': shot['shot_statsbomb_xg'],
        'n_defenders_in_cone': count_defenders_in_cone(freeze_frame, (x, y)),
        'goalkeeper_distance': get_goalkeeper_distance(freeze_frame),
        'heatmap':   build_heatmap(freeze_frame, (x, y)),
        'is_goal':   get_label(shot)
    }

def build_feature_df(shots_df: pd.DataFrame) -> pd.DataFrame:
    clean = shots_df[shots_df['shot_freeze_frame'].notna()].copy()
    print(f"Dropped {len(shots_df) - len(clean)} shots without freeze frame")

    dataframe = pd.DataFrame(clean.apply(extract_features, axis=1).tolist())
    dataframe['match_id'] = clean['match_id'].values
    return dataframe
