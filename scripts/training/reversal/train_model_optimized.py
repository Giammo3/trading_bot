import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import xgboost as xgb
import joblib

#carico i dati 
X_train = pd.read_csv('datasets/reversal/X_train.csv')
y_train = pd.read_csv('datasets/reversal/y_train.csv').squeeze()

#Addestro il modello XGBoost
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# === Salva modello ===
joblib.dump(model, 'models/reversal/model_reversal_v3.pkl')
print(" Modello XGBoost salvato in 'models/reversal/model_reversal_v3.pkl'")