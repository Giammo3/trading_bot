import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#parametri
target_col = 'reversal'
model_path = 'models/reversal/model_reversal.pkl'

#carico i dati
X_train = pd.read_csv('datasets/reversal/X_train.csv')
X_test = pd.read_csv('datasets/reversal/X_test.csv')
y_train = pd.read_csv('datasets/reversal/y_train.csv')
y_test = pd.read_csv('datasets/reversal/y_test.csv')

#modello 
print("🚀 Addestramento Random Forest su reversal...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'  # importante visto che il target è sbilanciato
)
model.fit(X_train, y_train.values.ravel())

# === VALUTAZIONE ===
y_pred = model.predict(X_test)

print("\n✅ Performance del modello reversal:")
print("Accuracy: ", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:   ", round(recall_score(y_test, y_pred), 4))
print("F1 Score: ", round(f1_score(y_test, y_pred), 4))

# Confusion Matrix
print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === SALVATAGGIO ===
joblib.dump(model, model_path)
print(f"\n💾 Modello salvato in '{model_path}'")




