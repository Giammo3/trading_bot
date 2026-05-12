import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/binary_02/X_test_predicted.csv')

df_signals = df[df['prediction'] == 1]

# === Statistiche sulla confidenza ===
print(f"\n Numero predizioni con segnale: {len(df_signals)}")
print(f" Statistiche confidence:")
print(df_signals["confidence"].describe())

# === Istogramma ===
plt.figure(figsize=(8, 5))
plt.hist(df_signals["confidence"], bins=20, color="dodgerblue", edgecolor="black")
plt.title("Distribuzione Confidence Score (solo pred = 1)")
plt.xlabel("Confidence")
plt.ylabel("Frequenza")
plt.grid(True)
plt.tight_layout()
plt.savefig("confidence_distribution.png")
plt.close()
print("\n Grafico salvato in 'confidence_distribution.png'")