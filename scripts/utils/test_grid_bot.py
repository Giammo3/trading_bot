# scripts/utils/test_grid_bot.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from scripts.trading.binary_02.predict_filtered import get_filtered_predictions
from scripts.trading.bot_universal import run_trading_bot
from scripts.training.binary_02.feature_selection import load_X_test_filtered
import joblib

# === CONFIGURAZIONE PARAMETRI ===
confidence_thresholds = [0.50, 0.55, 0.60, 0.65]
adx_thresholds = [20, 25, 30]
take_profits = [0.004, 0.006]
stop_losses = [0.003, 0.004]

model_path = 'models/binary_02/model_v2.pkl'
x_test_path = 'datasets/binary_02/X_test_filtered.csv'

model = joblib.load(model_path)
X_test = load_X_test_filtered()

results = []

# === TEST GRID ===
for conf_thresh, adx_thresh, tp, sl in product(confidence_thresholds, adx_thresholds, take_profits, stop_losses):
    df_filtered = get_filtered_predictions(conf_thresh)
    df_result = run_trading_bot(
        df_filtered,
        take_profit_pct=tp,
        stop_loss_pct=sl,
        prediction_col='prediction',
        entry_signal=1,
        verbose=False
    )
    trades = df_result[df_result['exit_price'].notna()].copy()
    wins = (trades['result'] == 1).sum()
    losses = (trades['result'] == -1).sum()
    final_balance = 10000 + (((trades['exit_price'] - trades['entry_price']) / trades['entry_price']) * 100 * trades['result']).sum()
    avg_pnl = (((trades['exit_price'] - trades['entry_price']) / trades['entry_price']) * 100 * trades['result']).mean()

    results.append({
        'conf_thresh': conf_thresh,
        'adx_thresh': adx_thresh,
        'take_profit': tp,
        'stop_loss': sl,
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(trades) * 100 if len(trades) > 0 else 0,
        'avg_pnl': avg_pnl if avg_pnl else 0,
        'final_balance': round(final_balance, 2)
    })

# === RISULTATI ===
df_results = pd.DataFrame(results)
df_sorted = df_results.sort_values(by='final_balance', ascending=False)
print("\n\U0001F3C6 Migliori combinazioni:")
print(df_sorted.head(10))
best_combo = df_sorted.iloc[0]
best_combo.to_json('scripts/trading/binary_02/best_combo.json')

# === GRAFICO ===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_results, x='trades', y='win_rate', hue='avg_pnl', size='final_balance', palette='viridis', legend=True)
plt.title('Risultati Test Combinazioni')
plt.xlabel('Numero Trade')
plt.ylabel('Win Rate %')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

