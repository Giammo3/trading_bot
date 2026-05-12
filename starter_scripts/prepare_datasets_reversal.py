import os
import subprocess
import sys
from pathlib import Path

# Usa l'interprete del venv attivo e imposta PYTHONPATH alla root del progetto
venv_python = sys.executable
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT)

scripts = [
    # FASE 0: Feature engineering (fonte unica, sostituisce features.py)
    "utils/generate_features.py",

    # FASE 1: Labeling (produce forex_labeled.csv e forex_labeled_balanced.csv)
    "utils/target.py",

    # FASE 2: Dataset reversal
    "scripts/training/reversal/prepare_dataset.py",
    "scripts/training/reversal/feature_selection.py",
    "scripts/training/reversal/filtered_X_train.py",

    # FASE 3: Flat filter + training
    "scripts/trading/reversal/flat_filter.py",
    "scripts/training/reversal/tune_model_xgb.py",

    # FASE 4: Predizione
    "scripts/trading/reversal/predict_filtered.py",

    # FASE 5: Analisi
    "scripts/evaluation/reversal/analyze_errors.py",
    "scripts/evaluation/reversal/analyze_fn_fp.py",
    "scripts/evaluation/reversal/analyze_predictions.py",
    "scripts/trading/reversal/bot_reversal.py",
    "scripts/evaluation/reversal/analyze_trades.py",
]

for script_path in scripts:
    print(f"\n>>> Lancio: {script_path}")
    result = subprocess.run([venv_python, script_path], env=env)
    if result.returncode != 0:
        print(f"    ERRORE nello script: {script_path} — esecuzione interrotta.")
        sys.exit(1)

print("\nTutti gli script reversal completati con successo.")
