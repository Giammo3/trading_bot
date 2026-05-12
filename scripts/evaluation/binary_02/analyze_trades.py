import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np

# === Caricamento dataset ===
df = pd.read_csv('datasets/binary_02/X_test_traded.csv')

# === Trade validi (con exit_price definito) ===
df_trades = df[df['exit_price'].notna()].copy()

# Calcolo PnL (%) corretto
pnl_raw = (df_trades['exit_price'] - df_trades['entry_price']) / df_trades['entry_price'] * 100
pnl_raw[df_trades['exit_price'] == df_trades['entry_price']] = 0
pnl_signed = pnl_raw * df_trades['result']
df_trades['pnl_pct'] = pnl_signed

df_trades['is_null_trade'] = (df_trades['exit_price'] == df_trades['entry_price'])
print(f"\nTrade nulli: {df_trades['is_null_trade'].sum()}/{len(df_trades)}")

# === Statistiche trade ===
total_trades = len(df_trades)
wins   = (df_trades['result'] == 1).sum()
losses = (df_trades['result'] == -1).sum()
win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

initial_balance = 10000
stake = 100
df_trades['pnl_eur'] = (df_trades['pnl_pct'] / 100) * stake

print("\nEsempio di trade con entry/exit:")
print(df_trades[['entry_price', 'exit_price', 'result']].head(10))

gross_profit = df_trades[df_trades['pnl_eur'] > 0]['pnl_eur'].sum()
gross_loss   = abs(df_trades[df_trades['pnl_eur'] < 0]['pnl_eur'].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
final_balance = initial_balance + df_trades['pnl_eur'].sum()

print(f"\nRisultati BOT Trader:")
print(f"Totale trade eseguiti: {total_trades}")
print(f"Win: {wins} | Loss: {losses}")
print(f"Win rate: {win_rate:.2f}%")
print(f"\nSaldo iniziale: {initial_balance:.2f} EUR")
print(f"Saldo finale:   {final_balance:.2f} EUR")
print(f"\nProfit Factor: {'{:.2f}'.format(profit_factor) if np.isfinite(profit_factor) else 'inf'}")

print("\nTrade con P&L nullo:")
print(df_trades[df_trades['pnl_pct'] == 0][['entry_price', 'exit_price', 'result']])

print("\nDistribuzione P&L:")
print(df_trades['pnl_pct'].describe())

print("\nDettaglio Trade:")
print(f"Media Win:   {df_trades[df_trades['result'] == 1]['pnl_pct'].mean():.2f}%")
print(f"Media Loss:  {df_trades[df_trades['result'] == -1]['pnl_pct'].mean():.2f}%")
print(f"Best Trade:  {df_trades['pnl_pct'].max():.2f}%")
worst = df_trades['pnl_pct'].min()
print(f"Worst Trade: {0.0 if abs(worst) < 0.005 else worst:.2f}%")

print("\nAnalisi binary_02 completata.")
