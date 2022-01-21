# Importar librerías
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Lectura del dataset
df = pd.read_csv("parkinsons.csv")
df.head()

# Obtención de etiquetas
ftr = df.loc[:, df.columns != "status"].values[:, 1:]
lbls = df.loc[:, "status"].values

# Transformación de ftr a un rango dado
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(ftr)
y = lbls

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=9)

# Entrenamiento del modelo
modelo = XGBClassifier()
modelo.fit(x_train, y_train)

# Evaluacion de la precisión del modelo
prediccion = modelo.predict(x_test)
print(accuracy_score(y_test, prediccion)*100)
