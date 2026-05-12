import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Lista colonne non da usare come feature
non_trading_features = [
    'hour_of_day', 'day_of_week', 'valid_trade',
    'timestamp'
]

def load_X_test_filtered():
    """
    Ritorna il DataFrame X_test_filtered dal CSV, utile per test grid bot.
    """
    return pd.read_csv('datasets/binary_02/X_test_filtered.csv')


if __name__ == "__main__":
    X_train = pd.read_csv('datasets/binary_02/X_train.csv')
    y_train = pd.read_csv('datasets/binary_02/y_train.csv').squeeze()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    print("\n Feature Importance:")
    print(feature_importances)

    threshold = 0.01
    important_features = [
        f for f in feature_importances[feature_importances > threshold].index
        if f not in non_trading_features
    ]

    print("\n Features tenute:")
    for feature in important_features:
        print("-", feature)

    X_train_filtered = X_train[important_features]
    X_train_filtered.to_csv('datasets/binary_02/X_train_filtered.csv', index=False)
    print("\n File 'X_train_filtered.csv' salvato!")
