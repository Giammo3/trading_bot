import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

#parametri
RANDOM_STATE = 42
N_ESTIMATORS = 300          #più alberi
MAX_DEPTH = 8               #NOn troppo profondi
MIN_SAMPLES_LEAF = 5        #Foglie più rubuste
MAX_FEATURES = 'sqrt'       #migliora la generalizzazioneù

#carica i dati
X_train = pd.read_csv('datasets/binary_02/X_train_filtered.csv')
y_train = pd.read_csv('datasets/binary_02/y_train.csv').squeeze()

X_test = pd.read_csv('datasets/binary_02/X_test_filtered.csv')

# Carica le feature selezionate automaticamente
selected_features = X_train.columns.tolist()
X_test = X_test[selected_features]

X_train = X_train[selected_features]

y_test = pd.read_csv('datasets/binary_02/y_test.csv').squeeze()

#modellazzaione
print(" Addestramento Random Forest ottimizzata...")

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    max_features=MAX_FEATURES,
    class_weight='balanced',  # AGGIUNTO
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

# === VALUTAZIONE ===
y_pred = model.predict(X_test)

print("\n Performance del modello ottimizzato:")
print("Accuracy: ", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:   ", round(recall_score(y_test, y_pred), 4))
print("F1 Score: ", round(f1_score(y_test, y_pred), 4))

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\n Distribuzione predizioni:")
print(np.unique(y_pred, return_counts=True))

# === SALVA MODELLO ===
joblib.dump(model, 'models/binary_02/model_v2.pkl')
print("\n Modello ottimizzato salvato in 'models/model_v2.pkl'")
