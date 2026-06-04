from sklearn.metrics import roc_auc_score, brier_score_loss
import numpy as np

def evaluate(model, X, y, df=None, label="Model"):
    y_pred = model.predict_proba(X)[:, 1]  # probability of class 1 (goal)
    auc = roc_auc_score(y, y_pred)
    brier = brier_score_loss(y, y_pred)

    print(f"{'Model':<22} {'AUC':>6} {'Brier':>7}")
    print("-" * 38)
    print(f"{label:<22} {auc:>6.3f} {brier:>7.4f}")

    if df is not None and 'statsbomb_xg' in df.columns:
        sb_xg = df['statsbomb_xg'].values
        mask = ~np.isnan(sb_xg)
        auch_sb = roc_auc_score(y[mask], sb_xg[mask])
        brier_sb = brier_score_loss(y[mask], sb_xg[mask])
        print(f"{'StatsBomb xG':<22} {auch_sb:>6.3f} {brier_sb:>7.4f}")

    # return y_pred


