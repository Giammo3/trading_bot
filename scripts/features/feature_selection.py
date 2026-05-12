from sklearn.ensemble import RandomForestClassifier

def select_features(X, y, top_k=30):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    top_features = X.columns[sorted_idx[:top_k]]
    return list(top_features)
