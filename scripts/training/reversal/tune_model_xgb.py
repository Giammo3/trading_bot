import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

# === Parametri ===
INPUT_FILE = 'datasets/reversal/X_train_filtered.csv'
TARGET_COLUMN = 'reversal'
OUTPUT_MODEL = 'models/reversal/best_model_xgb.pkl'

# === Carica i dati ===
df = pd.read_csv(INPUT_FILE)

# === Split X e y
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# === Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Griglia di parametri
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 5]
}

results = []
best_f1 = 0
best_model = None

print(" Inizio tuning XGBoost...")

for params in ParameterGrid(param_grid):
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42,
        **params
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    results.append({**params, 'precision': precision, 'recall': recall, 'f1': f1})

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        joblib.dump(model, OUTPUT_MODEL)

# === Salva i risultati
results_df = pd.DataFrame(results).sort_values(by='f1', ascending=False)
results_df.to_csv('models/reversal/xgb_tuning_results.csv', index=False)

print("\n Tuning completato! Migliori combinazioni:")
print(results_df.head(10))
print(f"\n Miglior modello salvato in: {OUTPUT_MODEL}")
