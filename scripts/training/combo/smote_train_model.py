import pandas as pd
import xgboost as xgb
import joblib 
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix
from imblearn.over_sampling import SMOTE

#Configurazione
DATASET_PATH = "datasets/combo_reversal_binary03/"
MODEL_PATH = "models/combo/binary03_combo_model_smote_xgb.pkl"
TARGET_TRAIN = "datasets/combo_reversal_binary03/y_train_binary03.csv"
TARGET_TEST = "datasets/combo_reversal_binary03/y_test_binary03.csv"

#carichimao i dati
# === CARICA I DATI ===
X_train = pd.read_csv(DATASET_PATH + "X_train_filtered_combo.csv")
X_test = pd.read_csv(DATASET_PATH + "X_test_filtered_combo.csv")
y_train = pd.read_csv(TARGET_TRAIN).squeeze()
y_test = pd.read_csv(TARGET_TEST).squeeze()

print(f"📏 Dimensioni originali: X_train={X_train.shape}, y_train={y_train.shape}")

#applichiamo lo SMOTE solo sul train
smote = SMOTE(random_state=42, sampling_strategy="auto") #bilanciamento 1:1
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"📊 Dopo SMOTE: X_train={X_train_res.shape}, y_train={y_train_res.shape}")
print(y_train_res.value_counts())

# === MODELLO XGBOOST ===
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train_res, y_train_res)

# 🔑 Allinea le colonne del test set con quelle usate in training
X_test = X_test[X_train.columns]

# === VALUTAZIONE ===
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
conf = confusion_matrix(y_test, y_pred)

print("📊 Risultati modello XGBoost (binary_03 + SMOTE):")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print("Confusion Matrix:")
print(conf)