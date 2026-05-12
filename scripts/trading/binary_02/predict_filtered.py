import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import joblib

def get_filtered_predictions(confidence_threshold=0.65):
    """
    Carica modello e dati test, allinea colonne, predice con soglia di confidence
    e restituisce DataFrame con colonne aggiuntive 'prediction' e 'confidence'.
    """
    model = joblib.load('models/binary_02/model_v2.pkl')

    X_test = pd.read_csv('datasets/binary_02/X_test_filtered.csv')
    X_train = pd.read_csv('datasets/binary_02/X_train_filtered.csv')
    X_test = X_test[X_train.columns]  # allineamento colonne

    probs = model.predict_proba(X_test)
    predictions = (probs[:, 1] > confidence_threshold).astype(int)

    X_test['prediction'] = predictions
    X_test['confidence'] = probs[:, 1]
    return X_test

if __name__ == "__main__":
    df = get_filtered_predictions(confidence_threshold=0.65)
    df.to_csv('datasets/binary_02/X_test_predicted.csv', index=False)
    print("Distribuzione predizioni:\n", df['prediction'].value_counts())
    print(" Predizioni completate con filtro corretto!")
