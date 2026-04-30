
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

def ejecutar_entrenamiento_completo(df, target_col='valor_contrato'):
    """
    Entrena un modelo CatBoost manejando automáticamente:
    - Categorías
    - Texto crudo (NLP)
    - Números
    - Transformación Logarítmica
    """
    
    # 1. Identificación automática de columnas
    all_cols = df.columns.tolist()
    
    # Listas de referencia (ajustadas a tu dataset)
    cat_features = [c for c in [
        'nivel_entidad', 'departamento_entidad', 'nombre_de_la_entidad',
        'municipio_entidad', 'estado_del_proceso', 'modalidad_de_contratacion',
        'tipo_de_contrato', 'origen', 'anio', 'mes'
    ] if c in all_cols]
    
    text_features = [c for c in ['objeto_del_proceso'] if c in all_cols]
    
    # Las numéricas son las que NO son target, ni cat, ni text
    num_features = [c for c in all_cols if c not in cat_features + text_features + [target_col]]

    print(f"Configuración: {len(cat_features)} categóricas, {len(text_features)} de texto, {len(num_features)} numéricas.")

    # 2. Preparar X e y
    X = df[cat_features + text_features + num_features].copy()
    y_log = np.log1p(df[target_col])
    
    # 3. Limpieza y tipos de datos
    X[cat_features] = X[cat_features].astype(str).fillna('none')
    if text_features:
        X[text_features] = X[text_features].astype(str).fillna('sin objeto')
    
    # 4. División de datos
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    # 5. Crear los Pools
    train_pool = Pool(X_train, y_train_log, cat_features=cat_features, text_features=text_features)
    test_pool = Pool(X_test, y_test_log, cat_features=cat_features, text_features=text_features)
    
    # 6. Configuración del Modelo
    model = CatBoostRegressor(
        iterations=4000,
        learning_rate=0.05,
        depth=8,
        loss_function='RMSE',
        random_seed=42,
        verbose=200,
        early_stopping_rounds=200
    )
    
    # 7. Entrenamiento
    print("Iniciando entrenamiento en CatBoost...")
    model.fit(train_pool, eval_set=test_pool, plot=False)
    
    return model, X_test, y_test_log
