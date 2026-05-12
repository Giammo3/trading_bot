import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

# === CONFIG ===
DATASET_PATH = "datasets/binary_03/"
MODEL_PATH = "models/binary_03/binary_03_model_xgb.pkl"
TARGET_NAME = "target_binary_03"

# === CREA FOLDER MODELLI SE NON ESISTE ===
os.makedirs("models/", exist_ok=True)

# === CARICA I DATI ===
X_train = pd.read_csv(DATASET_PATH + "X_train.csv")
X_test = pd.read_csv(DATASET_PATH + "X_test.csv")
y_train = pd.read_csv(DATASET_PATH + "y_train.csv").squeeze()
y_test = pd.read_csv(DATASET_PATH + "y_test.csv").squeeze()

# === CALCOLO WEIGHT POSITIVO ===
pos = sum(y_train == 1)
neg = sum(y_train == 0)
scale_pos_weight = neg / pos
print(f" Bilanciamento: {neg} negativi / {pos} positivi → scale_pos_weight = {scale_pos_weight:.2f}")

# === MODELLO XGBOOST ===
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# === VALUTAZIONE ===
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
conf = confusion_matrix(y_test, y_pred)

print(" Risultati modello XGBoost (binary_03):")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print("Confusion Matrix:")
print(conf)

# === SALVA MODELLO ===
joblib.dump(model, MODEL_PATH)
print(f" Modello salvato in {MODEL_PATH}")

# === SALVA FEATURE NAMES ===
joblib.dump(list(X_train.columns), "models/binary_03/feature_names.pkl")

