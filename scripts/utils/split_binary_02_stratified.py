import pandas as pd
from sklearn.model_selection import train_test_split

# Carica il dataset bilanciato completo
X = pd.read_csv('datasets/binary_02/X.csv')
y = pd.read_csv('datasets/binary_02/y.csv').squeeze()

# Dividi stratificando per mantenere la stessa proporzione di classi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Salva i nuovi file
X_train.to_csv('datasets/binary_02/X_train_filtered.csv', index=False)
X_test.to_csv('datasets/binary_02/X_test_filtered.csv', index=False)
y_train.to_csv('datasets/binary_02/y_train.csv', index=False)
y_test.to_csv('datasets/binary_02/y_test.csv', index=False)

print("✅ Dataset diviso con stratificazione e salvato correttamente.")
