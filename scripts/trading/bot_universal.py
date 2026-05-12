import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

def run_trading_bot(
    df,
    take_profit_pct=0.006,
    stop_loss_pct=0.004,
    look_ahead_steps=20,
    entry_signal=1,
    prediction_col='prediction',
    verbose=False,
    confidence_col='confidence',
    confidence_threshold=None,
    entry_filters=None,  # ⬅️ nuovo
    adx_threshold=25

):
    """
    Esegue un mini trading bot su un dataframe con una colonna di segnali.

    Args:
        df (DataFrame): Il dataframe su cui eseguire il trading (con colonne 'close' e prediction_col).
        take_profit_pct (float): Percentuale di take profit.
        stop_loss_pct (float): Percentuale di stop loss.
        look_ahead_steps (int): Numero massimo di candele da osservare dopo l'entry.
        entry_signal (int/str): Il valore della prediction che indica un'entrata.
        prediction_col (str): Il nome della colonna con i segnali di ingresso.
        verbose (bool): Se True stampa tutte le operazioni.
    """
    # Forza il tipo della colonna prediction a int (compatibile con pandas >= 2)
    # Usa pd.to_numeric per convertire senza toccare il dtype della colonna esistente
    if prediction_col in df.columns:
        df[prediction_col] = pd.to_numeric(df[prediction_col], errors='coerce').fillna(-999).astype(int)

    import numpy as np
    df = df.copy()  # evita SettingWithCopyWarning sul dataframe originale
    df['position']       = 0
    df['entry_price']    = np.nan
    df['exit_price']     = np.nan
    df['entry_index']    = np.nan
    df['trade_duration'] = np.nan
    df['result']         = np.nan

    in_position = False
    entry_price = None
    entry_index = None

    for i, row in df.iterrows():
        price = row['close']
        prediction_raw = row.get(prediction_col)
        
        # Prova a convertire il segnale in int (es. '1'  1, 'NO TRADE'  fallisce)
        try:
            prediction = int(prediction_raw)
        except (ValueError, TypeError):
            prediction = -999  # valore impossibile

        entry_condition = (prediction == entry_signal)

        if confidence_threshold is not None and confidence_col in row:
            entry_condition &= row[confidence_col] >= confidence_threshold

        if entry_filters:
            for filter_func in entry_filters:
                entry_condition &= filter_func(row)

        if not in_position and entry_condition:
                # Apertura posizione
                in_position = True
                entry_price = price
                entry_index = i
                df.at[i, 'position'] = 1
                df.at[i, 'entry_price'] = price
                df.at[i, 'entry_index'] = entry_index
                if verbose:
                    print(f"[+] Apertura posizione a {price:.5f} (riga {i})")

        elif in_position:
            # Calcoliamo dinamicamente lo stop loss basato sulla volatilità
            current_volatility = row.get('volatility_10', df['volatility_10'].mean())
            dynamic_sl_pct = stop_loss_pct * (current_volatility / df['volatility_10'].mean())
            take_profit = entry_price * (1 + take_profit_pct)
            stop_loss = entry_price * (1 - dynamic_sl_pct)


            lookahead_df = df.iloc[i:i+look_ahead_steps]

            for j, future_row in lookahead_df.iterrows():
                future_price = future_row['close']

                if future_price >= take_profit:
                    # Take Profit
                    duration = j - entry_index
                    df.at[j, 'exit_price'] = future_price
                    df.at[j, 'result'] = 1
                    df.at[j, 'entry_price'] = entry_price
                    df.at[j, 'entry_index'] = entry_index
                    df.at[j, 'trade_duration'] = duration
                    in_position = False
                    if verbose:
                        print(f"[] Take Profit a {future_price:.5f} (riga {j})")
                    break

                elif future_price <= stop_loss:
                    # Stop Loss
                    duration = j - entry_index
                    df.at[j, 'exit_price'] = future_price
                    df.at[j, 'result'] = -1
                    df.at[j, 'entry_price'] = entry_price
                    df.at[j, 'entry_index'] = entry_index
                    df.at[j, 'trade_duration'] = duration
                    in_position = False
                    if verbose:
                        print(f"[X] Stop Loss a {future_price:.5f} (riga {j})")
                    break

            if in_position and (i - entry_index) >= look_ahead_steps:
                final_price = price
                duration = i - entry_index
                 # Forza una minima variazione (es. 0.5 pip)
                final_price = entry_price - 0.00005 if final_price >= entry_price else entry_price + 0.00005
                df.at[i, 'exit_price'] = final_price
                df.at[i, 'result'] = 1 if final_price > entry_price else -1
                df.at[i, 'entry_price'] = entry_price
                df.at[i, 'entry_index'] = entry_index
                df.at[i, 'trade_duration'] = duration
                in_position = False
                if verbose:
                    print(f"[~] Chiusura forzata LOSS a {final_price:.5f} (riga {i})")
    
    return df

