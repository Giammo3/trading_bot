import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#caricaco il modello ottimizzato
model = joblib.load('models/model_reversal_v2.pkl')

#carica i dati reali (filtrati)
X_test_real = pd.read_csv('datasets/binary_02/X_test_filtered.csv')
y_test_real = pd.read_csv('datasets/reversal/y_test.csv')

#carico le feature usate dal modello
top_features = pd.read_csv('datasets/reversal/top_features.csv', header=None).squeeze().tolist()

# Se c'è '0' nella lista, lo togliamo
top_features = [f for f in top_features if f != '0']

#teniamo solo le feature selezionate
X_test_real = X_test_real[top_features]

#predizione 
y_pred_real = model.predict(X_test_real)

# ✅ Valutazione
print("\n✅ Performance modello reversal_v2 su dati reali:")
print("Accuracy: ", round(accuracy_score(y_test_real, y_pred_real), 4))
print("Precision:", round(precision_score(y_test_real, y_pred_real), 4))
print("Recall:   ", round(recall_score(y_test_real, y_pred_real), 4))
print("F1 Score: ", round(f1_score(y_test_real, y_pred_real), 4))

# 🧮 Confusion Matrix
print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_test_real, y_pred_real))




