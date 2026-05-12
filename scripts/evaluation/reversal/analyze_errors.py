import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import joblib

# Carica i dati
X_test_real = pd.read_csv('datasets/reversal/X_test_filtered_with_flat.csv')
y_test_real = pd.read_csv('datasets/reversal/y_test.csv').squeeze()

# Carica il modello XGBoost
model = joblib.load('models/reversal/best_model_xgb.pkl')

# Carica le top features usate
top_features = pd.read_csv('datasets/reversal/top_features.csv', header=None).squeeze().tolist()
top_features = [f for f in top_features if f != '0']
X_test_real = X_test_real[top_features]

#  Rimuovi la colonna 'reversal' se è ancora presente
if 'reversal' in X_test_real.columns:
    X_test_real = X_test_real.drop(columns=['reversal'])

# Predizioni
y_pred_real = model.predict(X_test_real)

# Trova falsi negativi e falsi positivi
false_negatives = (y_test_real == 1) & (y_pred_real == 0)
false_positives = (y_test_real == 0) & (y_pred_real == 1)

print(f"\n Falsi Negativi (reversal mancati): {false_negatives.sum()}")
print(f" Falsi Positivi (reversal falsi):   {false_positives.sum()}")

# Salva i dati per analisi
X_fn = X_test_real[false_negatives]
X_fp = X_test_real[false_positives]

X_fn.to_csv('datasets/reversal/false_negatives.csv', index=False)
X_fp.to_csv('datasets/reversal/false_positives.csv', index=False)

print("\n Files salvati:")
print("- datasets/reversal/false_negatives.csv")
print("- datasets/reversal/false_positives.csv")
