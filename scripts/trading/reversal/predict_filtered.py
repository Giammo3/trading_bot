import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import joblib

#carico il modello ottimizzato
model = joblib.load('models/reversal/best_model_xgb.pkl')

#carico il file con info 'falt market'
X_test_filtered = pd.read_csv('datasets/reversal/X_test_filtered_with_flat.csv')

#carico le top features usate
top_features = pd.read_csv('datasets/reversal/top_features.csv', header=None).squeeze().tolist()
top_features = [f for f in top_features if f != '0']  # rimuovo eventuale '0'

# Carico il target (serve per analisi FN/TP)
y_test = pd.read_csv('datasets/reversal/y_test.csv').squeeze()

predictions = []

for idx, row in X_test_filtered.iterrows():
    if row['flat_market'] == 1:
        predictions.append('NO TRADE')
    else:
        input_data = row[top_features].to_frame().T
        # Rimuove la colonna 'reversal' se presente tra le feature
        if 'reversal' in input_data.columns:
            input_data = input_data.drop(columns=['reversal'])
        pred = model.predict(input_data)[0]
        predictions.append(pred)

# Aggiungo le predizioni al dataset e il target
X_test_filtered['prediction'] = predictions
X_test_filtered['reversal'] = y_test.values

# Salvo il file finale
X_test_filtered.to_csv('datasets/reversal/X_test_predicted_filtered.csv', index=False)

print(" Predizioni reversal completate con filtro flat!")
print(" Salvato in 'datasets/reversal/X_test_predicted.csv'")