import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

# === 1. Carica i dataset ===
X_train = pd.read_csv('datasets/binary_02/X_train.csv')
X_test = pd.read_csv('datasets/binary_02/X_test.csv')
y_train = pd.read_csv('datasets/binary_02/y_train.csv').squeeze()
y_test = pd.read_csv('datasets/binary_02/y_test.csv').squeeze()

#=== 2. allena il modello ===
print("🚀 Addestramento Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 3. Valutazione ===
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n✅ Performance del modello:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\n🧮 Confusion Matrix:")
print(cm)

# === 4. Salva il modello in /models ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, 'models/binary_02/model_v1.pkl')
print("\n💾 Modello salvato in 'models/binary_02/model_v1.pkl'")