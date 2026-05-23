from sklearn.linear_model import LogisticRegression

def train_baseline(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

# later improvement: CNN instead of Heatmap
