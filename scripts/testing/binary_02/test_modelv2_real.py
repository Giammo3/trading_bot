import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#carico il nuovo modello
model = joblib.load('models/binary_02/model_v2.pkl')

#carico il dataset reale filtrato
X_test_real = pd.read_csv('datasets/binary_02/X_test_filtered.csv')
X_train = pd.read_csv('datasets/binary_02/X_train_filtered.csv')

# Teniamo solo le stesse feature usate nell'allenamento!
X_test_real = X_test_real[X_train.columns]

#carico traget reale
y_test_real = pd.read_csv('datasets/binary_02/y_test.csv')

#elimino la colonna 'flat_market' se esiste
if 'flat_market' in X_test_real.columns:
    X_test_real = X_test_real.drop(columns=['flat_market'])

#predizione sui dati reali
y_pred_real = model.predict(X_test_real)

# ✅ Valutazione
print("\n✅ Performance del modello v2 su dati reali:")
print("Accuracy: ", round(accuracy_score(y_test_real, y_pred_real), 4))
print("Precision:", round(precision_score(y_test_real, y_pred_real), 4))
print("Recall:   ", round(recall_score(y_test_real, y_pred_real), 4))
print("F1 Score: ", round(f1_score(y_test_real, y_pred_real), 4))

# 🧮 Confusion Matrix
print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_test_real, y_pred_real))