import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Carica i dati
df = pd.read_csv('datasets/reversal/X_test_predicted_filtered.csv')
y_test = pd.read_csv('datasets/reversal/y_test.csv')

# Teniamo solo i dati dove NON è "NO TRADE"
df_trading = df[df['prediction'] != 'NO TRADE'].copy()

# Allinea il target y_test
y_test = y_test.iloc[df_trading.index]

# Predizioni finali
y_pred = df_trading['prediction'].astype(int)

#  Valutazione
print("\n Performance sulle predizioni filtrate:")
print("Accuracy: ", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:   ", round(recall_score(y_test, y_pred), 4))
print("F1 Score: ", round(f1_score(y_test, y_pred), 4))

#  Confusion Matrix
print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



