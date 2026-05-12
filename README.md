# Forex AI Trading Bot

Sistema di trading automatizzato sul mercato Forex basato su Machine Learning.
Tre modelli indipendenti (reversal, binary\_02, binary\_03) vengono combinati
nell'ensemble **combo\_v2**, che integra policy di decisione modulari,
filtri di mercato calibrati senza data leakage, e diagnostica completa.

---

## Struttura del progetto

```
traiding_bot/
│
├── scripts/
│   ├── data_collection/        # Scaricamento dati OHLCV
│   │   └── main.py
│   │
│   ├── training/
│   │   ├── binary_02/          # Preparazione dataset + training Random Forest
│   │   ├── binary_03/          # Preparazione dataset + training XGBoost
│   │   ├── reversal/           # Preparazione dataset + training XGBoost (tuning)
│   │   └── combo/
│   │       ├── starter_combo.py              # Split train/test + feature selection combo
│   │       ├── split_target_binary03.py      # Target corretto (no leakage)
│   │       └── retrain_binary03_fixed.py     # Training combo + calibrazione soglie filtri
│   │
│   ├── trading/
│   │   ├── bot_universal.py            # Motore TP/SL condiviso
│   │   ├── binary_02/bot_trader.py     # Bot binary_02
│   │   ├── reversal/bot_reversal.py    # Bot reversal
│   │   └── combo/
│   │       ├── bot_combo_v2.py         # Bot ensemble v2 (entry point)
│   │       └── ensemble/               # Architettura modulare
│   │           ├── signal.py           # ModelSignal, SignalBundle, BaseModelAdapter
│   │           ├── adapters.py         # ReversalAdapter, Binary03Adapter
│   │           ├── aggregator.py       # SignalAggregator (join timestamp-safe)
│   │           ├── decision.py         # WeightedScorePolicy, ANDGatePolicy, DecisionEngine
│   │           ├── filters.py          # FlatMarketFilter, SessionFilter, LiquidityFilter
│   │           ├── threshold.py        # StaticThreshold, WalkForwardThresholdManager
│   │           └── orchestrator.py     # EnsembleOrchestrator (pipeline completa)
│   │
│   └── evaluation/
│       ├── binary_02/          # Analisi performance binary_02
│       ├── reversal/           # Analisi performance reversal
│       └── combo/
│           └── analyze_trades.py   # Analisi trade combo v2 (join su chiave OHLC)
│
├── utils/
│   ├── feature_engineering.py  # Fonte unica di feature (apply_all_features)
│   ├── generate_features.py    # Script standalone: raw CSV -> forex_features_optimized.csv
│   └── target.py               # Calcolo di tutti i target (reversal, binary_02, binary_03...)
│
├── starter_scripts/
│   ├── prepare_datasets_binary02.py    # Pipeline completa binary_02
│   ├── prepare_datasets_binary_03.py   # Pipeline completa binary_03
│   ├── prepare_datasets_reversal.py    # Pipeline completa reversal
│   └── run_combo_v2.py                 # Pipeline completa combo v2
│
├── datasets/
│   ├── forex/                          # Dati grezzi e preprocessati
│   ├── binary_02/                      # Dataset binary_02
│   ├── binary_03/                      # Dataset binary_03
│   ├── reversal/                       # Dataset reversal
│   └── combo_reversal_binary03/        # Dataset combo (train/test filtrati)
│
├── models/
│   ├── binary_02/model_v2.pkl                  # Random Forest binary_02
│   ├── binary_03/binary_03_model_xgb.pkl       # XGBoost binary_03
│   ├── reversal/best_model_xgb.pkl             # XGBoost reversal (tuned)
│   └── combo/
│       ├── binary03_combo_model_xgb.pkl        # XGBoost combo (ensemble)
│       └── filter_thresholds.json              # Soglie FlatMarketFilter (calibrate su X_train)
│
├── config.py           # Percorsi e parametri globali (TP, SL, look-ahead)
└── README.md
```

---

## Modelli

### reversal
Prevede se il prezzo invertira' direzione nelle prossime N candele.

- **Algoritmo:** XGBoost (tuning automatico via Optuna/grid search)
- **Feature:** 20 selezionate (liquidity\_proxy, volatility\_10, zscore\_ma50\_filtered, ADX, MACD, ...)
- **Target:** inversione binaria con filtro volatilita'
- **Risultati correnti:** 367 trade, Win Rate 57.2%

### binary\_02
Prevede se ci sara' un movimento significativo del prezzo (> 0.05%).

- **Algoritmo:** Random Forest ottimizzata
- **Feature:** 25 selezionate
- **Target:** `future_return > 0.0005` a 5 candele
- **Filtri:** flat market adattivo, confidence threshold, ADX
- **Risultati correnti:** 10 trade, Win Rate 50.0%

### binary\_03 / combo\_v2 (ensemble)
Combina reversal + binary\_03 in un sistema ensemble modulare.

- **Algoritmo:** XGBoost con `scale_pos_weight` per sbilanciamento classi
- **Feature:** 42 (include `market_session_code` per SessionFilter)
- **Target:** reversal AND movimento > 0.15% nelle 40 candele successive
- **Risultati correnti:** 69 trade, Win Rate 52.2%, Profit Factor 1.50

---

## Architettura combo\_v2

```
df OHLCV
    |
    v
SignalAggregator
    |-- ReversalAdapter   --> ModelSignal (prediction, confidence)
    |-- Binary03Adapter   --> ModelSignal (prediction, confidence)
    |
    v
DecisionEngine (WeightedScorePolicy)
    |-- Hard gate: reversal deve superare gate_threshold
    |-- Min confidence: entrambi i modelli >= 0.40
    |-- Composite score: media pesata >= score_threshold
    |
    v
FilterChain
    |-- FlatMarketFilter   (soglie da filter_thresholds.json, calibrate su X_train)
    |-- SessionFilter      (esclude Asia e Off_Hours tramite market_session_code)
    |-- LiquidityFilter    (liquidity_proxy >= 0.5)
    |
    v
EnsembleOrchestrator --> backtest TP/SL + salvataggio CSV
```

### Garanzie anti-leakage
- Le colonne `future_return`, `future_close`, `target_binary_03`, `reversal` sono
  in una lista `FORBIDDEN` e non vengono mai passate ai modelli.
- Le soglie di `FlatMarketFilter` sono calibrate esclusivamente su `X_train`
  durante il training e salvate in `models/combo/filter_thresholds.json`.
  Al momento del test vengono caricate dal file, non ricalcolate sul test set.
- Il `WalkForwardThresholdManager` per binary\_03 seleziona la soglia usando
  solo dati precedenti alla barra corrente (nessun look-ahead).

---

## Avvio rapido

### Prerequisiti
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 1. Raccolta dati
```bash
python scripts/data_collection/main.py
```

### 2. Pipeline completa per ogni modello

```bash
# Genera features + target + training + bot + analisi
python starter_scripts/prepare_datasets_reversal.py
python starter_scripts/prepare_datasets_binary02.py
python starter_scripts/prepare_datasets_binary_03.py

# Combo ensemble v2 (richiede reversal gia' trainato)
python starter_scripts/run_combo_v2.py
```

Ogni starter script esegue internamente, nell'ordine corretto:
```
utils/generate_features.py   ->  utils/target.py   ->  training  ->  trading  ->  evaluation
```

### 3. Solo combo, saltando la rigenerazione dati
```bash
# Usa i dataset esistenti, riaddestra solo binary_03 e lancia il bot
python starter_scripts/run_combo_v2.py --skip-data --skip-reversal
```

---

## Opzioni CLI di bot\_combo\_v2.py

```bash
python scripts/trading/combo/bot_combo_v2.py [OPZIONI]

  --threshold         static | walk_forward     (default: static)
  --static-value      float                     (default: 0.50)
  --policy            weighted | and_gate       (default: weighted)
  --score-threshold   float                     (default: 0.55)
  --reversal-gate     float                     (default: 0.50)
      Soglia minima di probabilita' per il hard gate del reversal.
      Valori piu' bassi (es. 0.10) ammettono piu' segnali.

  --vol-threshold     float  (0 = carica da filter_thresholds.json)
  --wick-threshold    float  (0 = carica da filter_thresholds.json)
      Soglie FlatMarketFilter. Se non specificati, vengono letti
      dal file calibrato su X_train durante il training.

  --no-flat-filter    Disabilita FlatMarketFilter
  --no-session-filter Disabilita SessionFilter
  --no-liquidity-filter Disabilita LiquidityFilter

  --diag-filters      Stampa diagnostica completa per ogni filtro:
                      barre bloccate, motivi, colonne mancanti
  --live              Modalita' live (nessun salvataggio CSV)
  --verbose           Output di debug esteso
```

### Esempi
```bash
# Configurazione consigliata
python scripts/trading/combo/bot_combo_v2.py \
    --reversal-gate 0.10 --threshold static --static-value 0.50

# Diagnostica filtri
python scripts/trading/combo/bot_combo_v2.py --diag-filters

# Walk-forward threshold (no look-ahead bias)
python scripts/trading/combo/bot_combo_v2.py --threshold walk_forward

# ANDGatePolicy (legacy, per confronto)
python scripts/trading/combo/bot_combo_v2.py --policy and_gate
```

---

## Catena dati (dal raw al modello)

```
scripts/data_collection/forex_data.csv          <- dati OHLCV grezzi
         |
         v
utils/generate_features.py                      <- apply_all_features() (fonte unica)
         |
         +--> datasets/forex/forex_features_optimized.csv
         |
         v
utils/target.py                                  <- calcola reversal, binary_02, binary_03
         |
         +--> datasets/forex/forex_labeled.csv
         +--> datasets/forex/forex_labeled_balanced.csv
         |
         +--[reversal]--> scripts/training/reversal/prepare_dataset.py
         |                      -> datasets/reversal/X_train.csv / X_test.csv
         |
         +--[binary_02]--> scripts/training/binary_02/prepare_dataset.py
         |                      -> datasets/binary_02/X_train.csv / X_test.csv
         |
         +--[binary_03]--> scripts/training/binary_03/forex_with_binary_03.py
                                -> datasets/binary_03/forex_labeled_with_binary03.csv
                                -> scripts/training/combo/starter_combo.py
                                -> scripts/training/combo/split_target_binary03.py
                                      -> datasets/combo_reversal_binary03/X_train_filtered_combo.csv
                                      -> datasets/combo_reversal_binary03/y_train_binary03.csv
                                      |
                                      v
                               scripts/training/combo/retrain_binary03_fixed.py
                                      -> models/combo/binary03_combo_model_xgb.pkl
                                      -> models/combo/filter_thresholds.json
```

---

## Note tecniche

**Feature engineering unificata**
`utils/feature_engineering.py` e' la sola sorgente di feature per tutti i modelli.
`utils/features.py` e' legacy e non deve essere usato — produceva scale diverse
(`return_pct * 100`, sessioni con codici diversi) che causavano collasso delle
predizioni quando i modelli venivano valutati su dati nuovi.

**Target binary\_03 corretto**
Il file `y_train_binary03.csv` contiene il target corretto (18.8% positivi).
`y_train_filtered_combo.csv` era stato corrotto da `starter_combo.py` che
ricalcolava il target con uno shift errato producendo solo 0.76% positivi —
causa del bug "0 trade su 3484 barre".

**Join analyze\_trades.py**
`X_test_traded_v2.csv` non ha timestamp (il dataset combo non lo propaga).
Il join con `forex_labeled.csv` per recuperare `future_return` viene eseguito
su chiave OHLC composita `close+open+high+low` a 5 decimali — risulta univoca
per tutti i trade (0 ambiguità su 69 trade verificati).

**Compatibilita' Windows**
Tutti gli script aggiungono `sys.stdout.reconfigure(encoding='utf-8')` all'inizio
per evitare crash con caratteri non-ASCII su terminali Windows (cp1252).
`bot_universal.py` usa `np.nan` invece di `None` per inizializzare le colonne
numeriche, compatibile con pandas >= 2 (StringDtype).

---

## Stato attuale e prossimi passi

**Completato**
- [x] Feature engineering unificata (utils/feature\_engineering.py)
- [x] Pipeline completa per reversal, binary\_02, binary\_03, combo
- [x] Architettura ensemble modulare combo\_v2 (signal, adapters, aggregator, decision, filters, orchestrator)
- [x] Anti-leakage: colonne forbidden, soglie FlatMarketFilter calibrate su X\_train, WalkForward threshold
- [x] Diagnostica filtri (--diag-filters): barre bloccate per filtro, colonne mancanti
- [x] Reversal gate configurabile da CLI (--reversal-gate)
- [x] market\_session\_code aggiunto al dataset combo (SessionFilter ora attivo)
- [x] Fix join analyze\_trades su chiave OHLC (tutti i 69 trade analizzati)
- [x] Fix compatibilita' Windows (UTF-8, pandas >= 2, bot\_universal)

**Prossimi passi**
- [ ] Ottimizzare look\_ahead\_steps nel motore TP/SL (attualmente 64/69 posizioni restano aperte)
- [ ] Aggiungere timestamp al dataset combo per allineamento temporale preciso
- [ ] Deploy bot in modalita' live (collegamento broker API)
- [ ] Integrazione modelli neurali (MLP, LSTM) come adapter aggiuntivi nell'ensemble
- [ ] Reinforcement Learning policy come DecisionPolicy (gia' previsto nell'architettura)

---

Progetto personale — BRO TEAM
