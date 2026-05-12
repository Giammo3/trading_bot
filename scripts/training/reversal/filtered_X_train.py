import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

# Carica X_train completo
X_train = pd.read_csv('datasets/reversal/X_train.csv')

# Carica le top features
top_features = pd.read_csv('datasets/reversal/top_features.csv', header=None).squeeze().tolist()
top_features = [f for f in top_features if f != '0']  # in caso ci sia uno '0' fantasma

# Aggiungi 'reversal' se non è già tra le top features
if 'reversal' not in top_features:
    top_features.append('reversal')

# Filtra solo le colonne migliori
X_train_filtered = X_train[top_features]

# Salva il file
X_train_filtered.to_csv('datasets/reversal/X_train_filtered.csv', index=False)

print(" File 'X_train_filtered.csv' creato con successo.")
