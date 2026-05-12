import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

# === PARAMETRI ===
initial_balance = 10000
csv_path = 'datasets/reversal/X_test_traded.csv'

# === CARICAMENTO ===
df = pd.read_csv(csv_path)
df = df.dropna(subset=['entry_price', 'exit_price'])

# Calcolo P&L in percentuale
df['pnl_pct'] = (df['exit_price'] - df['entry_price']) / df['entry_price'] * 100
df['pnl_pct'] = df.apply(lambda row: -abs(row['pnl_pct']) if row['result'] == -1 else row['pnl_pct'], axis=1)

# Calcolo saldo cumulativo
balance = initial_balance
pnl_values = []

for pnl in df['pnl_pct']:
    profit = balance * (pnl / 100)
    balance += profit
    pnl_values.append(profit)

df['pnl_value'] = pnl_values

# === RISULTATI ===
wins = df[df['result'] == 1]
losses = df[df['result'] == -1]

print(" Risultati Mini-Bot Reversal:")
print(f"Totale trade eseguiti: {len(df)}")
print(f"Win: {len(wins)} | Loss: {len(losses)}")
print(f"Win rate: {len(wins) / len(df) * 100:.2f}%\n")

print(f" Saldo iniziale: {initial_balance:.2f}€")
print(f" Saldo finale:   {balance:.2f}€\n")

# Profit Factor
gross_profit = wins['pnl_value'].sum()
gross_loss = -losses['pnl_value'].sum()
profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
print(f"Profit Factor: {profit_factor:.2f}\n")

# Dettagli trade
print(" Distribuzione P&L:")
print(df['pnl_pct'].describe())

print("\n Dettaglio Trade:")
print(f"Media Win: {wins['pnl_pct'].mean():.2f}%")
print(f"Media Loss: {losses['pnl_pct'].mean():.2f}%")
print(f"Best Trade: {df['pnl_pct'].max():.2f}%")
print(f"Worst Trade: {df['pnl_pct'].min():.2f}%")
