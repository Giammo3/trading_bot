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

    # FASE 2: Dataset binary_02
    "scripts/training/binary_02/prepare_dataset.py",
    "scripts/training/binary_02/feature_selection.py",

    # FASE 3: Flat filter + training
    "scripts/trading/binary_02/flat_filter.py",
    "scripts/training/binary_02/train_model_optimized.py",

    # FASE 4: Predizione
    "scripts/trading/binary_02/predict_filtered.py",

    # FASE 5: Simulazione trading
    "scripts/trading/binary_02/bot_trader.py",

    # FASE 6: Analisi performance
    "scripts/evaluation/binary_02/analyze_trades.py",
]

for script_path in scripts:
    print(f"\n>>> Lancio: {script_path}")
    result = subprocess.run([venv_python, script_path], env=env)
    if result.returncode != 0:
        print(f"    ERRORE nello script: {script_path} — esecuzione interrotta.")
        sys.exit(1)

print("\nTutti gli script binary_02 completati con successo.")