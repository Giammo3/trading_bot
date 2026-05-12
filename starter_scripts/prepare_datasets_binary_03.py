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

    # FASE 2: Aggiunge target_binary_03 a forex_labeled
    "scripts/training/binary_03/forex_with_binary_03.py",

    # FASE 3: Preparazione dataset binary_03 (X/y train+test)
    "scripts/training/binary_03/prepare_dataset.py",

    # FASE 4: Top features binary_03
    "scripts/training/binary_03/generate_top_features.py",

    # FASE 5: Training modello binary_03
    "scripts/training/binary_03/train_model.py",
]

for script_path in scripts:
    print(f"\n>>> Lancio: {script_path}")
    result = subprocess.run([venv_python, script_path], env=env)
    if result.returncode != 0:
        print(f"    ERRORE nello script: {script_path} — esecuzione interrotta.")
        sys.exit(1)

print("\nTutti gli script binary_03 completati con successo.")
