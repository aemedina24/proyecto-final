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
   
    texto = re.sub(r'[\(\)\[\]\*]', '', texto)
    # reemplazar separadores
    texto = re.sub(r'[-\t\n\r]', ' ', texto)     
    # limpiar espacios
    texto = re.sub(r'\s+', ' ', texto).strip()
    # Eliminar URLs
    texto = re.sub(r'http\S+|www\S+', '', texto)
    # Eliminar números
    texto = re.sub(r'\d+', '', texto)
    return texto if texto else "missing"


