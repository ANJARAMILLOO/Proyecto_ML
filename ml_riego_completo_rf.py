import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report
import joblib

# Leer dataset
df = pd.read_csv("datos_entrenamiento_iot.csv")

# Agregar columnas derivadas
df["costo_agua"] = df["litros_requeridos"] * df["precio_litro"]
df["agua_desperdiciada"] = (df["litros_requeridos"] - df["litros_ideales"]).clip(lower=0)
df["tiempo_riego"] = df["litros_requeridos"] / df["caudal_litros_min"]

# Variables independientes
X = df[["humedad_suelo", "cultivo_cod", "temp_ambiente", "hum_ambiente"]]

# Targets
targets = {
    "litros": df["litros_requeridos"],
    "campo_seco": df["campo_seco"],
    "costo_agua": df["costo_agua"],
    "desperdicio": df["agua_desperdiciada"],
    "tiempo_riego": df["tiempo_riego"]
}

# Separar entrenamiento/prueba
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Entrenamiento y guardado
modelos = {}

# Litros (Regresión)
modelo_litros = RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=5, random_state=42)
modelo_litros.fit(X_train, targets["litros"].loc[X_train.index])
joblib.dump(modelo_litros, "modelo_litros_rf.joblib")
modelos["litros"] = modelo_litros

# Campo seco (Clasificación)
modelo_seco = RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_leaf=5, random_state=42)
modelo_seco.fit(X_train, targets["campo_seco"].loc[X_train.index])
joblib.dump(modelo_seco, "modelo_campo_seco_rf.joblib")
modelos["campo_seco"] = modelo_seco

# Costo agua
modelo_costo = RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=5, random_state=42)
modelo_costo.fit(X_train, targets["costo_agua"].loc[X_train.index])
joblib.dump(modelo_costo, "modelo_costo_agua_rf.joblib")
modelos["costo"] = modelo_costo

# Desperdicio
modelo_desp = RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=5, random_state=42)
modelo_desp.fit(X_train, targets["desperdicio"].loc[X_train.index])
joblib.dump(modelo_desp, "modelo_agua_desp_rf.joblib")
modelos["desperdicio"] = modelo_desp

# Tiempo de riego
modelo_tiempo = RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=5, random_state=42)
modelo_tiempo.fit(X_train, targets["tiempo_riego"].loc[X_train.index])
joblib.dump(modelo_tiempo, "modelo_tiempo_riego_rf.joblib")
modelos["tiempo"] = modelo_tiempo

print("\n✅ Todos los modelos entrenados y guardados.")
