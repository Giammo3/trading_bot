import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib
import os

# === PATH
DATA_DIR = "datasets/combo_reversal_binary03"
MODEL_PATH = "models/combo/binary03_combo_model_xgb.pkl"

# === Caricamento dati
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_binary03.csv")).squeeze()

X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test_binary03.csv")).squeeze()

# Rimuovi eventuali colonne non numeriche come 'timestamp'
if 'timestamp' in X_train.columns:
    X_train = X_train.drop(columns=['timestamp'])
    X_test = X_test.drop(columns=['timestamp'])

assert len(X_train) == len(y_train), f"X_train ({len(X_train)}) e y_train ({len(y_train)}) hanno lunghezza diversa!"

# === Scale_pos_weight per class imbalance
n_negative = (y_train == 0).sum()
n_positive = (y_train == 1).sum()
scale_pos_weight = round(n_negative / n_positive, 2)
print(f"📊 Bilanciamento: {n_negative} negativi / {n_positive} positivi → scale_pos_weight = {scale_pos_weight}")

# === Allenamento modello XGBoost
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=scale_pos_weight)
model.fit(X_train, y_train)

# === Valutazione
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("📊 Risultati modello XGBoost (combo → binary_03):")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("Confusion Matrix:")
print(cm)

# === Salvataggio modello
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"✅ Modello salvato in {MODEL_PATH}")
