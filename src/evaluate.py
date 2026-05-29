from sklearn.metrics import roc_auc_score, brier_score_loss

def evaluate(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_pred)
    
    print(f"{auc:>6.3f} \n{brier:>7.4f}")



