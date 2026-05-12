import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from utils.feature_engineering import apply_all_features

# 1. Carica il dataset completo con i timestamp
df_full = pd.read_csv("datasets/binary_03/forex_labeled_with_binary03.csv")
df_full = apply_all_features(df_full)

# 2. Conta quante righe ha il vecchio test
old_test = pd.read_csv("datasets/combo_reversal_binary03/X_test_filtered_combo.csv")
test_length = len(old_test)

# 3. Filtra le ultime N righe del dataset completo (come X_test originale)
df_test = df_full.tail(test_length).copy()

# 4. Prendi le colonne corrette usate durante il training
X_train = pd.read_csv("datasets/combo_reversal_binary03/X_train_filtered_combo.csv")
columns_to_keep = X_train.columns.tolist()

# 5. Filtra le colonne
df_test_filtered = df_test[columns_to_keep]

# 6. Salva
df_test_filtered.to_csv("datasets/combo_reversal_binary03/X_test_filtered_combo.csv", index=False)
print(" X_test_filtered_combo.csv rigenerato correttamente.")
