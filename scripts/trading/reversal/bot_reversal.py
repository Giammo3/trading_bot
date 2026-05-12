import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import sys
import os

# Aggiungo la root del progetto al path per importare config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(root_dir)

import pandas as pd
import config
from scripts.trading.bot_universal import run_trading_bot
#from scripts.utils.reversal.entry_filters_auto import entry_filters_reversal


#carico i dati con predizioni
df = pd.read_csv('datasets/reversal/X_test_filtered_with_flat.csv')
print("Colonne presenti:")
print(df.columns.tolist())

# Verifica che abbia la colonna prediction (altrimenti uniamo le predizioni)
if 'prediction' not in df.columns:
    pred_df = pd.read_csv('datasets/reversal/X_test_predicted_filtered.csv')
    df['prediction'] = pred_df['prediction']

print(df['prediction'].value_counts())

#lancio il bot universale
df_traded = run_trading_bot(
    df,
    take_profit_pct=config.TAKE_PROFIT_PCT,
    stop_loss_pct=config.STOP_LOSS_PCT,
    look_ahead_steps=config.LOOK_AHEAD_STEPS,
    entry_signal=1,
    prediction_col='prediction'
    #entry_filters=entry_filters_reversal
)

#salvo i trade
df_traded.to_csv('datasets/reversal/X_test_traded.csv', index=False)
print(" BOT trading reversal completato! File salvato in 'datasets/reversal/X_test_traded.csv'")
