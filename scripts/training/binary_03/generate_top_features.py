import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import os 
import sys

#Importiamo la funzione per le features
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from scripts.features.feature_selection import select_features

DATASET_PATH = "datasets/binary_03/"
OUTPUT_FILE = "datasets/binary_03/top_features.csv"

#carichiamo i dati
X_train = pd.read_csv(os.path.join(DATASET_PATH, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATASET_PATH, "y_train.csv"))

#selezione top features
X_train_num = X_train.select_dtypes(include=['number']) #seleziona solo le colonne con in numeri.
top_features = select_features(X_train_num, y_train, top_k=30)

#Salviamo 
with open(OUTPUT_FILE, "w") as f:
    for feat in top_features:
        f.write(f"{feat}\n")

print(f" Top features generate e salvate in {OUTPUT_FILE}")
print(" Features selezionate:", top_features)