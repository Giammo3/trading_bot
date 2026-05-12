import pandas as pd
import joblib
import xgboost as xgb

# === Dataset ===
X_train = pd.read_csv("datasets/combo_reversal_binary03/X_train_filtered_combo.csv")
y_train = pd.read_csv("datasets/combo_reversal_binary03/y_train_filtered_combo.csv").squeeze()

X_test = pd.read_csv("datasets/combo_reversal_binary03/X_test_filtered_combo.csv")
y_test = pd.read_csv("datasets/combo_reversal_binary03/y_test_filtered_combo.csv").squeeze()

# === Carica top features originali ===
with open("datasets/binary_03/top_features.csv") as f:
    top_features = [line.strip() for line in f]

# === Rimuovi colonne non disponibili in real-time ===
forbidden = ["future_return", "future_return_pct", "future_close"]
top_features = [f for f in top_features if f not in forbidden]

print("✅ Features usate per il training (dopo il filtro):")
print(top_features)

# === Applica filtro ===
X_train = X_train[top_features].copy()
X_test = X_test[top_features].copy()

# === Addestra modello XGBoost ===
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# === Valutazione rapida ===
acc = model.score(X_test, y_test)
print(f"📊 Accuracy sul test: {acc:.2%}")

# === Salvataggio modello ===
joblib.dump(model, "models/combo/binary03_combo_model_xgb.pkl")

print("✅ Nuovo modello binary_03 combo addestrato e salvato con successo.")
print("📁 Salvato in: models/combo/binary03_combo_model_xgb.pkl")
