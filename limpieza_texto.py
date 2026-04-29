import pandas as pd
import re
import unicodedata





# -------------------------
# limpieza texto
# -------------------------
import pandas as pd
import re
import unicodedata

def limpieza_total(texto):
    if pd.isna(texto):
        return "missing"
    
    texto = str(texto).lower()
    
    # quitar tildes
    texto = unicodedata.normalize('NFKD', texto)\
            .encode('ascii', 'ignore')\
            .decode('utf-8')
    
    #quitar parentesis y corchetes
    texto = re.sub(r'\(.*?\)|\[.*?\]', '', texto)
    
    # reemplazar separadores
    texto = re.sub(r'[-\t\n\r]', ' ', texto)
         
    # limpiar espacios
    texto = re.sub(r'\s+', ' ', texto).strip()

    # Eliminar URLs
    texto = re.sub(r'http\S+|www\S+', '', texto)

    # Eliminar números
    texto = re.sub(r'\d+', '', texto)
    
    return texto if texto else "missing"


import re


# -------------------------
# limpiar stop word
# -------------------------


custom_stopwords = set([
    'suscrito', 'celebrado', 'contratista', 'vinculo',
    'caracter', 'manera', 'forma', 'caso', 'parte',
    'mismo', 'cada', 'fin', 'base', 'cuanto',
    'favor', 'marco', 'acuerdo',
    'realizar', 'ejecutar', 'desarrollar', 'implementar',
    'contratar', 'prestar', 'apoyar',
    'objeto', 'contrato', 'proceso', 'vigencia',
    'entidad', 'administrativa', 'administrativo', 'administrativos',
    'asignadas'
])

def limpiar_stopwords(texto):
    
    return " ".join([
        p for p in custom_stopwords
        if len(p) >= 3 and p not in custom_stopwords
    ])
