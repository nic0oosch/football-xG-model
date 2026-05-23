from sklearn.metrics import roc_auc_score, brier_score_loss

def evaluate(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)[:, 1]
    print(f"AUC: {roc_auc_score(y_test, y_pred):.3f}")
    print(f"Brier: {brier_score_loss(y_test, y_pred):.3f}")

