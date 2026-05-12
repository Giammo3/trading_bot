import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import joblib
import xgboost as xgb

#carico i dati
X_train = pd.read_csv('datasets/reversal/X_train.csv')

X_train = X_train.drop(columns=['reversal'], errors='ignore')

y_train =pd.read_csv('datasets/reversal/y_train.csv')

# Addestra un XGBoost per calcolare l'importanza delle feature
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# === Calcola importanza ===
feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
feature_importance = feature_importance.sort_values(ascending=False)

# === Stampa risultati ===
print("\n Feature Importance (Reversal):")
print(feature_importance)

# === Salva top feature se vuoi (facoltativo)
top_features = feature_importance.head(20).index.tolist()
pd.Series(top_features).to_csv('datasets/reversal/top_features.csv', index=False)
print("\n Top 20 features salvate in 'datasets/reversal/top_features.csv'")