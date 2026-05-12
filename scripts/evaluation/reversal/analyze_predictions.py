import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

# Carico le predizioni
df = pd.read_csv('datasets/reversal/X_test_predicted_filtered.csv')

# Conto le predizioni
pred_counts = df['prediction'].value_counts()

print("\n Riassunto Predizioni:")
print(pred_counts)

# Calcolo anche le percentuali
pred_percentages = (pred_counts / len(df) * 100).round(2)

print("\n Percentuali Predizioni:")
print(pred_percentages)