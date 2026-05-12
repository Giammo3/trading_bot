import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

# Carica i dati
X_test = pd.read_csv('datasets/binary_02/X_test.csv')

# Stampa statistiche delle colonne usate nel filtro
print("\n Statistiche chiave:")
print("Volatilità (10 periodi):")
print(f"Min: {X_test['volatility_10'].min():.6f} | Max: {X_test['volatility_10'].max():.6f} | Media: {X_test['volatility_10'].mean():.6f}")
print("\nAccelerazione normalizzata:")
print(f"Min: {X_test['acceleration_5_norm'].abs().min():.6f} | Max: {X_test['acceleration_5_norm'].abs().max():.6f} | Media: {X_test['acceleration_5_norm'].abs().mean():.6f}")
print("\nBody size:")
print(f"Min: {X_test['body_size'].min():.6f} | Max: {X_test['body_size'].max():.6f} | Media: {X_test['body_size'].mean():.6f}")

# Soglie dinamiche basate sui quartili (modifica qui)
VOLATILITY_THRESHOLD = X_test['volatility_10'].quantile(0.25)  # Filtra il 10% più piatto
ACCELERATION_THRESHOLD = X_test['acceleration_5_norm'].abs().quantile(0.25)
BODY_SIZE_THRESHOLD = X_test['body_size'].quantile(0.25)

print(f"\n Soglie adattive:")
print(f"Volatilità: {VOLATILITY_THRESHOLD:.6f}")
print(f"Accelerazione: {ACCELERATION_THRESHOLD:.6f}")
print(f"Body size: {BODY_SIZE_THRESHOLD:.6f}")

# Applica il filtro
flat_condition = (
    (X_test['volatility_10'] < VOLATILITY_THRESHOLD) | 
    (X_test['acceleration_5_norm'].abs() < ACCELERATION_THRESHOLD) | 
    (X_test['body_size'] < BODY_SIZE_THRESHOLD)
)

X_test['flat_market'] = flat_condition.astype(int)
X_test.to_csv('datasets/binary_02/X_test_filtered.csv', index=False)

print(f"\n Filtro flat applicato: {flat_condition.sum()} righe su {len(X_test)} totali marcate come 'flat'.")