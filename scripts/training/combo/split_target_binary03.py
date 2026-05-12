import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

# === File di input/output ===
X_train_file = "datasets/combo_reversal_binary03/X_train.csv"
X_test_file = "datasets/combo_reversal_binary03/X_test.csv"
labeled_file = "datasets/binary_03/forex_labeled_with_binary03.csv"
target_col = "target_binary_03"

output_train = "datasets/combo_reversal_binary03/y_train_binary03.csv"
output_test = "datasets/combo_reversal_binary03/y_test_binary03.csv"

# === Carica X_train/X_test per capire la lunghezza
X_train = pd.read_csv(X_train_file)
X_test = pd.read_csv(X_test_file)
n_train = len(X_train)
n_test = len(X_test)

# === Carica il file labeled e estrai target finale (prendiamo solo la coda finale)
df_labeled = pd.read_csv(labeled_file)
y_total = df_labeled[target_col].tail(n_train + n_test).reset_index(drop=True)

# === Split basato sulle dimensioni corrette
y_train = y_total[:n_train]
y_test = y_total[n_train:]

# === Salvataggio
y_train.to_csv(output_train, index=False)
y_test.to_csv(output_test, index=False)

print(" y_train_binary03.csv e y_test_binary03.csv salvati!")
print(f" Lunghezza y_train: {len(y_train)}")
print(f" Lunghezza y_test:  {len(y_test)}")
