"""
analyze_trades.py — Analisi completa dei trade prodotti da bot_combo_v2.

Join con forex_labeled.csv
--------------------------
X_test_traded_v2.csv non contiene timestamp (il dataset combo non lo ha mai avuto).
Il join viene eseguito su chiave OHLC composita (close+open+high+low a 5 decimali),
che risulta univoca per tutti i trade — verificato: 0 chiavi ambigue su 69 trade.

NON si usa l'indice numerico per il join perché:
- Il dataset combo è uno slice del labeled dopo dropna + feature engineering
- Gli indici non corrispondono a quelli originali di forex_labeled
- Il timestamp nel traded è sempre NaN (colonna vuota)

PnL
---
- pnl_pct dal bot: calcolato da TP/SL hit (entry_price -> exit_price)
  Solo le righe con execution_result != NaN hanno PnL reale.
- future_return da labeled: return effettivo a 5 candele dalla entry
  Usato come verifica/benchmark del PnL del bot.
"""

import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

TRADED_PATH  = ROOT / "datasets" / "combo_reversal_binary03" / "X_test_traded_v2.csv"
LABELED_PATH = ROOT / "datasets" / "forex" / "forex_labeled.csv"

# ---------------------------------------------------------------------------
# Caricamento
# ---------------------------------------------------------------------------
traded = pd.read_csv(TRADED_PATH)
labeled = pd.read_csv(LABELED_PATH)

# Filtra solo le righe dove il bot ha aperto una posizione
trades = traded[traded["traded"] == True].copy()

if trades.empty:
    print("Nessun trade eseguito.")
    sys.exit(0)

print(f"Trade totali in X_test_traded_v2.csv: {len(trades)}")

# ---------------------------------------------------------------------------
# Join con forex_labeled per recuperare future_return e timestamp
# ---------------------------------------------------------------------------
ROUND = 5

def make_ohlc_key(df):
    return (
        df["close"].round(ROUND).astype(str) + "_" +
        df["open"].round(ROUND).astype(str)  + "_" +
        df["high"].round(ROUND).astype(str)  + "_" +
        df["low"].round(ROUND).astype(str)
    )

labeled["_key"] = make_ohlc_key(labeled)
trades["_key"]  = make_ohlc_key(trades)

# Diagnostica join
key_counts   = labeled["_key"].value_counts()
n_unique     = (trades["_key"].map(key_counts) == 1).sum()
n_ambiguous  = (trades["_key"].map(key_counts) > 1).sum()
n_missing    = (trades["_key"].map(key_counts).isna()).sum()

print(f"\n--- Diagnostica join (chiave OHLC a {ROUND} decimali) ---")
print(f"  Chiavi univoche in labeled   : {n_unique} / {len(trades)}")
print(f"  Chiavi ambigue (>=2 match)   : {n_ambiguous}")
print(f"  Chiavi senza match           : {n_missing}")

if n_ambiguous > 0:
    print(f"  NOTA: per le chiavi ambigue si usa il primo match cronologico.")

# Dedup labeled: per chiavi ambigue prende la prima occorrenza cronologica
lbl_join = labeled.drop_duplicates(subset="_key", keep="first")[
    ["_key", "timestamp", "future_return", "future_return_pct", "future_close"]
]

trades = trades.merge(lbl_join, on="_key", how="left")

n_recovered = trades["future_return"].notna().sum()
print(f"  future_return recuperati     : {n_recovered} / {len(trades)}")
if n_recovered < len(trades):
    print(f"  ATTENZIONE: {len(trades) - n_recovered} trade senza future_return.")

# ---------------------------------------------------------------------------
# Analisi PnL dal bot (TP/SL simulation)
# ---------------------------------------------------------------------------
# execution_result: 1=TP, -1=SL, 0=time-expired, NaN=posizione non chiusa
closed = trades[trades["execution_result"].notna()].copy()
open_pos = trades[trades["execution_result"].isna()].copy()

print(f"\n--- Stato posizioni ---")
print(f"  Posizioni chiuse (TP/SL/expired) : {len(closed)}")
print(f"  Posizioni aperte (non chiuse)     : {len(open_pos)}")
if not open_pos.empty:
    print(f"  NOTE: {len(open_pos)} posizioni aperte non hanno PnL TP/SL.")
    print(f"        Verranno valutate tramite future_return da labeled.")

# PnL delle posizioni chiuse
pnl_closed = closed["pnl_pct"].dropna()

print(f"\n=== RISULTATI BOT (posizioni chiuse con TP/SL) ===")
if len(pnl_closed) > 0:
    wins_tp   = (closed["execution_result"] == 1).sum()
    losses_sl = (closed["execution_result"] == -1).sum()
    expired   = (closed["execution_result"] == 0).sum()
    wr_closed = wins_tp / len(closed) * 100

    print(f"  Take Profit (1)  : {wins_tp}")
    print(f"  Stop Loss  (-1)  : {losses_sl}")
    print(f"  Time-expired (0) : {expired}")
    print(f"  Win rate TP/SL   : {wr_closed:.1f}%")
    print(f"  PnL medio/trade  : {pnl_closed.mean():.3f}%")
    print(f"  PnL totale       : {pnl_closed.sum():.3f}%")
    gross_w = pnl_closed[pnl_closed > 0].sum()
    gross_l = abs(pnl_closed[pnl_closed <= 0].sum())
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    print(f"  Profit factor    : {pf:.3f}")
else:
    print("  Nessuna posizione chiusa con PnL.")

# ---------------------------------------------------------------------------
# Analisi completa su tutti i 69 trade usando future_return
# ---------------------------------------------------------------------------
print(f"\n=== RISULTATI COMPLETI (tutti i {len(trades)} trade, future_return da labeled) ===")

if trades["future_return"].notna().sum() == 0:
    print("  Impossibile calcolare: future_return non disponibile.")
else:
    fr = trades["future_return"].dropna()

    wins_fr  = (fr > 0).sum()
    losses_fr = (fr <= 0).sum()
    wr_fr     = wins_fr / len(fr) * 100

    pnl_fr_pct = fr * 100  # in %
    gross_w_fr = pnl_fr_pct[pnl_fr_pct > 0].sum()
    gross_l_fr = abs(pnl_fr_pct[pnl_fr_pct <= 0].sum())
    pf_fr      = gross_w_fr / gross_l_fr if gross_l_fr > 0 else float("inf")

    initial_balance = 10000
    balance = initial_balance
    for r in fr:
        balance *= (1 + r)

    print(f"  Trade valutati     : {len(fr)}")
    print(f"  Win (future > 0)   : {wins_fr}")
    print(f"  Loss (future <= 0) : {losses_fr}")
    print(f"  Win rate           : {wr_fr:.1f}%")
    print(f"  PnL medio/trade    : {pnl_fr_pct.mean():.3f}%")
    print(f"  PnL totale         : {pnl_fr_pct.sum():.3f}%")
    print(f"  Profit factor      : {pf_fr:.3f}")
    print(f"  Saldo iniziale     : {initial_balance:.0f} EUR")
    print(f"  Saldo finale       : {balance:.2f} EUR")
    print(f"  Rendimento totale  : {(balance/initial_balance - 1)*100:.2f}%")

    print(f"\n  Distribuzione future_return (%):")
    desc = pnl_fr_pct.describe()
    for stat, val in desc.items():
        print(f"    {stat:8s}: {val:.3f}%")

# ---------------------------------------------------------------------------
# Dettaglio tabella completa trade
# ---------------------------------------------------------------------------
print(f"\n=== DETTAGLIO TRADE ===")
show_cols = ["timestamp", "close", "entry_price", "exit_price",
             "execution_result", "pnl_pct", "future_return"]
show_cols = [c for c in show_cols if c in trades.columns]
pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", "{:.5f}".format)
print(trades[show_cols].to_string(index=True))

print("\nAnalisi completata.")
