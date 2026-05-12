import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import joblib

#paremetri
CONFIDENCE_THRESHOLD = 0.6 #può andare bene anche 0.75 o 0.8

model = joblib.load('models/binary_02/model_v2.pkl')

X_test = pd.read_csv('datasets/binary_02/X_test_filtered.csv')
X_train = pd.read_csv('datasets/binary_02/X_train_filtered.csv')  # Per le colonne

#mantieni le colonne usate nel traiding
X_test = X_test[X_train.columns]

# === PREDIZIONE ===
probs = model.predict_proba(X_test)
preds = model.predict(X_test)

# Probabilità per la classe "1"
confidences = probs[:, 1]

# Applica soglia: se sotto → NO TRADE
final_preds = [
    int(pred) if conf >= CONFIDENCE_THRESHOLD else 'NO TRADE'
    for pred, conf in zip(preds, confidences)
]

X_test['prediction'] = final_preds
X_test['confidence'] = confidences

# === SALVATAGGIO ===
X_test.to_csv('datasets/binary_02/X_test_predicted.csv', index=False)

print("Distribuzione predizioni:\n", X_test['prediction'].value_counts(dropna=False))
print(" Predizioni completate con confidence filtering!")