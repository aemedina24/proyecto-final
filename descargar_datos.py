
# %%
#carga de datos usando libreria parquet para reducir tamaño del archivo

import pandas as pd
#Cargar el archivo original 

print("Leyendo datos_contratacion.csv...")
df = pd.read_csv('data/datos_contratacion.csv', sep=',', low_memory=False)

# Convertir a Parquet (comprimido)
print("Comprimiendo a datos_secop.parquet...")
df.to_parquet('data/datos_secop.parquet', index=False)

print("¡Proceso terminado con éxito!")

