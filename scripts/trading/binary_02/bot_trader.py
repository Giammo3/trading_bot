import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import sys
import os

# Aggiungo la root del progetto al path per importare config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(root_dir)

# SOLO DOPO aver sistemato sys.path fai gli import veri
import config 
from scripts.trading.bot_universal import run_trading_bot
import pandas as pd
import json

#PARAMETRICI DINAMICI

with open('scripts/trading/binary_02/best_combo.json', 'r') as f:
    params = json.load(f)

confidence_threshold = params.get("confidence_threshold", 0.5)
adx_threshold = params.get("adx_threshold", 20)
take_profit_pct = params.get("take_profit_pct", 0.004)
stop_loss_pct = params.get("stop_loss_pct", 0.004)
    
#carico i dati con predizioni
df = pd.read_csv(config.X_TEST_PREDICTED_BINARY)

#trading bot universale
df_traded = run_trading_bot(
    df,
    take_profit_pct=take_profit_pct,
    stop_loss_pct=stop_loss_pct,
    look_ahead_steps=30,
    entry_signal=1,
    prediction_col = 'prediction',
    verbose=False,
    confidence_threshold=confidence_threshold,
    adx_threshold=adx_threshold
)

# Salva il risultato
df_traded.to_csv('datasets/binary_02/X_test_traded.csv', index=False)

print("\n BOT trading completato! File salvato in 'datasets/binary_02/X_test_traded.csv'")