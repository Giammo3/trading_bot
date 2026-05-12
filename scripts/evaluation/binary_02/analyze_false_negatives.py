import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

#carico i dati dei test
X_test = pd.read_csv('datasets/binary_02/X_test.csv')
y_test = pd.read_csv('datasets/binary_02/y_test.csv').squeeze() #elimina le dimensioni inutili e trasforma il DataFrame 2D(causato da read_csv()) in una serie(/vettore) 1D(utili per i modelli scikit-learn)

#carico il modello salvato
model = joblib.load('models/binary_02/model_v1.pkl')

#predici sui dati di test
y_pred = model.predict(X_test)

#calcola la confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(" Confusion Matrix:")
print(cm)

#trova i falsi negativi
false_negatives = (y_test == 1) & (y_pred == 0)
X_fn = X_test[false_negatives]  #X_fn contiene solo le feature degli esempi che il mpdeòòp ha sbagliato

print(f"\n Trovati {X_fn.shape[0]} falsi negativi.")

#analisi statistica dei falsi negativi
print("\n Statistiche sui Falsi Negativi:")
print(X_fn.describe())

# (Optional) Se vuoi anche confrontare coi veri positivi:
true_positives = (y_test == 1) & (y_pred == 1)
X_tp = X_test[true_positives]

print("\n Statistiche sui Veri Positivi:")
print(X_tp.describe())