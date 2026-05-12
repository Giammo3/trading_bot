import pandas as pd
import joblib 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#caricare modello salvato
print("📦 Caricamento modello...")
model = joblib.load('models/binary_02/model_v1.pkl')

#carico i dati reali
print("📊 Caricamento dati reali...")
df_real = pd.read_csv('forex_labeled.csv')

#preparo i dati
df_real['timestamp'] = pd.to_datetime(df_real['timestamp'])
df_real.sort_values(by='timestamp', inplace=True)

#elimino colonne inutili
drop_cols = ['timestamp', 'future_close', 'future_return', 'future_return_pct']
df_real.drop(columns=drop_cols, inplace=True)

#separo features e target
X_real = df_real.drop(columns=['target_binary_02'])
y_real = df_real['target_binary_02']

#predizione
print("🚀 Predizione sui dati reali...")
y_pred_real = model.predict(X_real)

#valutazione
acc = accuracy_score(y_real, y_pred_real)
prec = precision_score(y_real, y_pred_real)
rec = recall_score(y_real, y_pred_real)
f1 = f1_score(y_real, y_pred_real)
cm = confusion_matrix(y_real, y_pred_real)

print("\n✅ Performance del modello su dati reali:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\n🧮 Confusion Matrix:")
print(cm)

print("\n📈 Distribuzione reale dei target:")
print(y_real.value_counts(normalize=True).mul(100).round(2).astype(str) + " %")
