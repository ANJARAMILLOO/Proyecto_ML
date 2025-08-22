import pandas as pd
import random

# Información base por cultivo
cultivos_info = {
    "CAÑA DE AZUCAR": {"cod": 0, "litros_base": 110, "litros_ideales": 100},
    "ARROZ": {"cod": 1, "litros_base": 90, "litros_ideales": 85},
    "TRIGO": {"cod": 2, "litros_base": 140, "litros_ideales": 130},
    "PAPA": {"cod": 3, "litros_base": 80, "litros_ideales": 75},
    "MAIZ": {"cod": 4, "litros_base": 70, "litros_ideales": 65},
}

# Columnas del dataset
columnas = [
    'humedad_suelo', 'cultivo', 'temp_ambiente', 'hum_ambiente',
    'litros_requeridos', 'campo_seco', 'cultivo_cod',
    'precio_litro', 'litros_ideales', 'caudal_litros_min'
]

csv_path = "datos_entrenamiento_iot.csv"

# Crear CSV vacío con encabezados
pd.DataFrame(columns=columnas).to_csv(csv_path, index=False)

# Generar 150,000 registros
registros = []
for _ in range(150_000):
    cultivo = random.choice(list(cultivos_info.keys()))
    info = cultivos_info[cultivo]

    humedad_suelo = round(random.uniform(10, 85), 1)
    temp_ambiente = round(random.uniform(18, 38), 1)
    hum_ambiente = round(random.uniform(35, 95), 1)

    litros_requeridos = info["litros_base"] + (50 - humedad_suelo) * random.uniform(1.2, 2.2)
    litros_requeridos = round(max(20, min(litros_requeridos, 280)), 1)

    umbral_seco = 50 if cultivo != "ARROZ" else 60
    campo_seco = 1 if humedad_suelo < umbral_seco else 0

    precio_litro = round(random.uniform(0.01, 0.05), 3)  # S/. por litro
    caudal = round(random.uniform(15, 35), 1)  # L/min
    litros_ideales = info["litros_ideales"]

    registros.append([
        humedad_suelo, cultivo, temp_ambiente, hum_ambiente,
        litros_requeridos, campo_seco, info["cod"],
        precio_litro, litros_ideales, caudal
    ])

# Guardar registros al CSV
pd.DataFrame(registros, columns=columnas).to_csv(csv_path, mode='a', header=False, index=False)

print(f"\n✅ Archivo generado exitosamente: {csv_path}")

