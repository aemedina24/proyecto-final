
#carga de datos usando libreria parquet para reducir tamaño del archivo

import pandas as pd
#Cargar el archivo original 

print("Leyendo datos_contratacion.csv...")
df = pd.read_csv('datos_contratacion.csv', sep=',', low_memory=False)

# Convertir a Parquet (comprimido)
print("Comprimiendo a datos_secop.parquet...")
df.to_parquet('data/datos_secop.parquet', index=False)

print("¡Proceso terminado con éxito!")

# %%
#comprimir archivo contratos_con_cluster

print("Leyendo contratos_con_cluster.csv...")
df = pd.read_csv('data/contratos_con_cluster.csv', sep=',', low_memory=False)

# Convertir a Parquet (comprimido)
print("Comprimiendo a contratos_con_cluster.parquet...")
df.to_parquet('data/contratos_con_cluster.parquet', index=False)

print("¡Proceso terminado con éxito!")
# %%
