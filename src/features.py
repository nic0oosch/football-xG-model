
def extract_feature(shot) -> dict:
    # Basis feature
    x, y = shot["location"]
    distance = calc_distance(x, y)
    angle = calc_angle(x, y)

    # Freeze Frame -> Heatmap
    freeze_frame = shot["shot"]["freeze_frame"]
    heatmap = build_heatmap(freeze_frame)

    return {"distance": distance, "angle": angle, "heatmap": heatmap}

def get_label(shot) -> int:
    return 1 if shot["shot"]["outcome"]["name"] == "Goal" else 0