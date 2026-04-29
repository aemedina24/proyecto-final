
#%%
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import unicodedata
import re
from sklearn.preprocessing import normalize



#%%
# 1. CARGA DE GEOGRAFÍA
df_mun = pd.read_csv('data/municipios.csv')
df_dep = pd.read_csv('data/departamentos.csv')
df_mun['Nombre Municipio'] = df_mun['Nombre Municipio'].str.lower()
df_dep['Nombre Departamento'] = df_dep['Nombre Departamento'].str.lower()

def quitar_tildes(texto):
    if not isinstance(texto, str): return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
# Limpieza de diccionarios geográficos
df_mun['Nombre Municipio'] = df_mun['Nombre Municipio'].str.lower().apply(quitar_tildes)
df_dep['Nombre Departamento'] = df_dep['Nombre Departamento'].str.lower().apply(quitar_tildes)

#CREAR SETS
departamentos = set(df_dep['Nombre Departamento'])
ciudades = set(df_mun['Nombre Municipio'])

#%%
# 2. CONFIGURACIÓN DE LIMPIEZA
# Corregido: Faltaba una coma antes de 'nueva basura'
stop_words = [
    # --- Trámite y Legal ---
    'prestacion', 'servicios', 'apoyo', 'gestion', 'contrato', 'objeto', 
    'suscrito', 'celebrado', 'autonomia', 'independiente', 'caracter', 
    'temporal', 'manera', 'vinculo', 'laboral', 'cuenta', 'riesgo',
    'para', 'prestar', 'como', 'contratista', 'mediante', 'dentro', 
    'caso', 'forma', 'parte', 'clase', 'proceso', 'mismo', 'cada', 
    'fin', 'base', 'cuanto', 'entidad', 'vigencia', 'anualidad',
    'realizar', 'ejecutar', 'orientado', 'consistente', 'favor', 
    'marco', 'desarrollo', 'contratar', 'acuerdo', 'asignadas',

    # --- Relleno Administrativo (Las que ensucian tus clusters) ---
    'aunar', 'esfuerzos', 'tecnicos', 'administrativos', 'financieros', 
    'fortalecimiento', 'mejoramiento', 'desarrollar', 'implementar',
    'tecnico', 'profesional', 'profesionales', 'actividades', 'integral', 
    'especializado', 'administrativo', 'operativo', 'asistencial', 
    'implementacion', 'ejecucion', 'seguimiento', 'coordinacion', 
    'relacionadas', 'proyectos', 'proyecto', 'apoyar', 'procesos', 
    'area', 'areas', 'acciones', 'administrativa',
]


# Convertimos a set para que la búsqueda sea ultra rápida
palabras_basura_set = set(palabras_basura)

def limpieza_pro(texto):
    if not texto or pd.isna(texto): 
        return "missing"
    
    texto = str(texto).lower().replace('\n', ' ') #eliminca saltos de linea y pasa a minuscula
    texto = quitar_tildes(texto) #quita tildes
    texto = re.sub(r'\d+', '', texto) #eliminar numeros
    texto = re.sub(r'[^\w\s]', '', texto) #elimina caracteres especiales

    palabras = texto.split() #divide en palabras
    palabras_limpias = []

        
    for p in palabras:
        if p in ciudades:
            palabras_limpias.append("ciudad")
        elif p in departamentos:
            palabras_limpias.append("departamento")
        elif p not in palabras_basura_set and len(p) > 3:
            palabras_limpias.append(p)

    return " ".join(palabras_limpias) if palabras_limpias else "missing"



#%%
# 3. PROCESAMIENTO
df = pd.read_csv('data/objeto_contratos.csv')
df['texto_limpio'] = df['objeto_del_proceso'].apply(limpieza_pro)

#%%
# 4. EMBEDDINGS
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['texto_limpio'].tolist(), show_progress_bar=True)


#%%
# 5. Normalización L2
embeddings_norm = normalize(embeddings)

#%%
# 6 Guardar arachivo
np.save('data/embeddings_norm.npy', embeddings_norm)

#%%








#%%

#%%
n_clusters = 22 
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['cluster_id'] = kmeans.fit_predict(X)

# Guardar el resultado final
df.to_csv('data/contratos_con_clusters.csv', index=False)



#%%

#Metodo del codo
# Calcular la inercia para diferentes valores de K
inercias = []
# Probamos de 5 a 40 clusters dado que la contratación estatal es muy variada
rango_k = range(5, 41, 2) 

print("Calculando inercias... esto puede tardar un poco dependiendo del volumen de datos.")

for k in rango_k:
    # Usamos k-means++ para una mejor convergencia inicial
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(embeddings_norm)
    inercias.append(kmeans.inertia_)

# 4. Graficar el Método del Codo
plt.figure(figsize=(12, 6))
plt.plot(rango_k, inercias, marker='o', linestyle='--', color='b')
plt.title('Método del Codo: Buscando el K óptimo para Contratos')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inercia')
plt.grid(True)
plt.show()


#%%
#AGRUPAR CLUSTERING

# Cargar la base y los embeddings
df = pd.read_csv('data/objeto_contratos.csv')
X = np.load('data/embeddings_norm.npy')

#Ejecutar el clustering
n_clusters = 24 # Puedes probar con más o menos
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['cluster_id'] = kmeans.fit_predict(X)

# Guardar el resultado final
df.to_csv('data/contratos_con_clusters.csv', index=False)


# %%
#8. GRAFICA DE CLUSTERS 

# CARGAR  ARCHIVOS
df = pd.read_csv('data/contratos_con_clusters.csv')
X = np.load('data/embeddings_norm.npy')

# TOMAR UNA MUESTRA 
# Usaremos 2000 filas, suficiente para ver si hay ruido
n_muestras = min(2000, len(df))
df_sample = df.sample(n=n_muestras, random_state=42).reset_index()
X_sample = X[df_sample['index'].values] # Extrae los embeddings correctos de la muestra

# REDUCCIÓN PREVIA CON PCA (Acelera el proceso de t-SNE significativamente)
# Pasamos de 384 dimensiones a 50 antes de entrar a t-SNE
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_sample)

# EJECUTAR T-SNE 
print("Calculando t-SNE para la muestra...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
puntos_2d = tsne.fit_transform(X_pca)

# PREPARAR DATOS PARA EL GRÁFICO
df_viz = pd.DataFrame({
    'x': puntos_2d[:, 0],
    'y': puntos_2d[:, 1],
    'cluster': df_sample['cluster_id'].astype(str)
})

# 6. VISUALIZAR
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_viz, 
    x='x', y='y', 
    hue='cluster', 
    palette='tab10', 
    alpha=0.7, 
    edgecolor=None
)

plt.title("Mapa de Ruido de Clusters (Muestra de 2000 contratos)")
plt.xlabel("Dimensión 1")
plt.ylabel("Dimensión 2")
plt.legend(title='ID Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# %%
# 9.TABLA DE METRICAS

# df_sample y X_sample vienen del paso anterior del t-SNE
n_muestras = min(2000, len(df))
df_sample = df.sample(n=n_muestras, random_state=42).reset_index(drop=True)
X_sample = X[df.sample(n=n_muestras, random_state=42).index] 

# CALCULAR MÉTRICA GLOBAL
# Este número te dice qué tan bien están separados los grupos en general
score_global = silhouette_score(X_sample, df_sample['cluster_id'])

print("==========================================")
print(f"PUNTAJE DE SILUETA GLOBAL: {score_global:.4f}")
print("==========================================")
print("Interpretación: >0.25 aceptable, >0.5 sólido, <0.1 mucho ruido.\n")

# CALCULAR MÉTRICA POR CLUSTER
# Calculamos el score individual para cada contrato de la muestra
sample_scores = silhouette_samples(X_sample, df_sample['cluster_id'])
df_sample['silueta_individual'] = sample_scores

# Agrupamos para ver el promedio de "limpieza" de cada cluster
reporte_clusters = df_sample.groupby('cluster_id')['silueta_individual'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)

print("REPORTE DE CALIDAD POR CLUSTER:")
print(reporte_clusters.rename(columns={'mean': 'Calidad (Silueta)', 'count': 'N° Ejemplos en muestra'}))

# IDENTIFICAR LOS 5 CONTRATOS CON MÁS RUIDO
# Estos son los que tienen score negativo (están en el grupo equivocado)
print("\n--- CONTRATOS POSIBLEMENTE MAL CLASIFICADOS (RUIDO) ---")
ruido = df_sample.sort_values(by='silueta_individual').head(5)
for i, row in ruido.iterrows():
    print(f"Cluster {row['cluster_id']} | Score: {row['silueta_individual']:.3f} | Texto: {row['objeto_del_proceso'][:80]}...")





# %%
#ARCHIVO PARA ENVIAR A LA IA NO CORRER

# Cargar tu archivo de resultados
df = pd.read_csv('data/contratos_con_clusters.csv')

resumen_ia = []

for cluster_id, grupo in df.groupby('cluster_id'):
    # Ordenamos por silueta (si la tienes) para dar los ejemplos más "limpios"
    if 'silueta_individual' in grupo.columns:
        muestras = grupo.sort_values(by='silueta_individual', ascending=False).head(15)
    else:
        muestras = grupo.sample(n=min(15, len(grupo)), random_state=42)
    
    ejemplos_texto = " | ".join(muestras['objeto_del_proceso'].tolist())
    
    resumen_ia.append({
        'ID_Cluster': cluster_id,
        'Tamaño': len(grupo),
        'Muestra_Contratos': ejemplos_texto
    })

# Guardar el archivo de muestra
df_muestra = pd.DataFrame(resumen_ia)
df_muestra.to_csv('data/muestra_para_bautizar.csv', index=False)


