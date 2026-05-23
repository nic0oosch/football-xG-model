import pandas as pd
from statsbombpy import sb

def load_shots(competition_id: int, season_id: int) -> pd.DataFrame:
    matches = sb.matches(competition_id=competition_id, season_id=season_id)

    all_shots = []
    for match_id in matches.match_id:
        events = sb.events(match_id=match_id)
        shots = events[events["type"] == "Shot"]
        all_shots.append(shots)

    return pd.concat(all_shots, ignore_index=True)

