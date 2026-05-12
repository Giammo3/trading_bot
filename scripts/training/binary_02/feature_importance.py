import pandas as pd
import joblib
import matplotlib.pyplot as plt

#carico il modello
model = joblib.load('models/binary_02/model_v1.pkl')

#carico le features usate
X_train = pd.read_csv('datasets/binary_02/X_train.csv')
feature_names = X_train.columns

#estraggo l'importanza delle feature
importances = model.feature_importances_

#creao una calssifica ordinata
feature_importance = pd.Series(importances, index=feature_names)
feature_importance = feature_importance.sort_values(ascending=False)

#stampa le top 10 feature
print("🏆 Top 10 Feature più importanti:")
print(feature_importance.head(10))

#visualizzo un grafico a barre
plt.figure(figsize=(10,6))
feature_importance.head(15).plot(kind='bar')
plt.title('Top 15 Feature Importance - Random Forest')
plt.ylabel('Importanza')
plt.tight_layout()
plt.show()