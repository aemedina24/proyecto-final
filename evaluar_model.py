import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from sklearn.metrics import (mean_absolute_error,r2_score,mean_squared_error)

def comparar_modelos3(y_test_log, preds_lgbm_log, preds_cat_log):
    # 1. Transformación a escala real (Pesos)
    y_real = np.expm1(y_test_log)
    p_lgbm = np.expm1(preds_lgbm_log)
    p_cat = np.expm1(preds_cat_log)
    
    # 2. Definir límite dinámico basado en el valor máximo real
    # Esto elimina el espacio vacío innecesario
    limite_ejes = y_real.max() * 1.05 
    
    # 3. Crear la figura con 1 fila y 2 columnas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    
    # Formateador para millones ($M)
    fmt = FuncFormatter(lambda x, pos: f'${x/1e6:.0f}M')
    
    # --- GRÁFICA 1: LightGBM ---
    ax1.scatter(y_real, p_lgbm, alpha=0.25, s=15, color='#1f77b4')
    ax1.plot([0, limite_ejes], [0, limite_ejes], color='red', linestyle='--', linewidth=2)
    ax1.set_title("LightGBM + Embeddings", fontsize=14)
    ax1.set_xlabel("Valor Real (Pesos)")
    ax1.set_ylabel("Valor Predicho (Pesos)")
    ax1.xaxis.set_major_formatter(fmt)
    ax1.yaxis.set_major_formatter(fmt)
    ax1.set_xlim(0, limite_ejes)
    ax1.set_ylim(0, limite_ejes)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- GRÁFICA 2: CatBoost ---
    ax2.scatter(y_real, p_cat, alpha=0.25, s=15, color='#2ca02c') # Color verde para diferenciar
    ax2.plot([0, limite_ejes], [0, limite_ejes], color='red', linestyle='--', linewidth=2)
    ax2.set_title("CatBoost + Text Features", fontsize=14)
    ax2.set_xlabel("Valor Real (Pesos)")
    # El eje Y se comparte con ax1 por 'sharey=True'
    ax2.xaxis.set_major_formatter(fmt)
    ax2.set_xlim(0, limite_ejes)
    ax2.set_ylim(0, limite_ejes)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Comparativa de Desempeño: Valor Real vs Predicho", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()



def comparar_metricas(y_true_log, p_lgbm_log, p_cat_log):
    """
    Calcula y compara métricas de error entre LightGBM y CatBoost
    convirtiendo los resultados de logaritmo a escala real.
    """
    # Transformación a escala real (Pesos)
    y_true = np.expm1(y_true_log)
    p_lgbm = np.expm1(p_lgbm_log)
    p_cat = np.expm1(p_cat_log)
    
    print(f"\n{' METRICA ':=^30}")
    
    # RMSE
    rmse_lgbm = np.sqrt(mean_squared_error(y_true, p_lgbm))
    rmse_cat = np.sqrt(mean_squared_error(y_true, p_cat))
    print(f"RMSE LightGBM: ${rmse_lgbm:,.0f}")
    print(f"RMSE CatBoost: ${rmse_cat:,.0f}")
    
    print(f"{'':-^30}") # Separador visual
    
    # MAE
    mae_lgbm = mean_absolute_error(y_true, p_lgbm)
    mae_cat = mean_absolute_error(y_true, p_cat)
    print(f"MAE LightGBM:  ${mae_lgbm:,.0f}")
    print(f"MAE CatBoost:  ${mae_cat:,.0f}")
    
    print(f"{'':-^30}") # Separador visual
    
    # R2 (Se calcula sobre los logs para evaluar el ajuste del modelo)
    r2_lgbm = r2_score(y_true_log, p_lgbm_log)
    r2_cat = r2_score(y_true_log, p_cat_log)
    print(f"R2 LightGBM:   {r2_lgbm:.4f}")
    print(f"R2 CatBoost:   {r2_cat:.4f}")
    print(f"{'':=^30}")

   



def comparar_importancia(lgbm_model, cat_model, features_lgbm, features_cat):
    """
    Compara las variables más influyentes de ambos modelos lado a lado.
    """
    # 1. Preparar datos de LightGBM
    imp_lgbm = lgbm_model.feature_importances_
    # Ordenar y tomar las top 15
    sorted_idx_lgbm = np.argsort(imp_lgbm)[-15:]
    
    # 2. Preparar datos de CatBoost
    imp_cat = cat_model.get_feature_importance()
    # Ordenar y tomar las top 15
    sorted_idx_cat = np.argsort(imp_cat)[-15:]

    # 3. Crear la figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Gráfica LightGBM
    ax1.barh([features_lgbm[i] for i in sorted_idx_lgbm], 
             imp_lgbm[sorted_idx_lgbm], color='teal')
    ax1.set_title('Top 15 Variables: LightGBM', fontsize=14)
    ax1.set_xlabel('Importancia (Gain/Split)')

    # Gráfica CatBoost
    ax2.barh([features_cat[i] for i in sorted_idx_cat], 
             imp_cat[sorted_idx_cat], color='darkslategrey')
    ax2.set_title('Top 15 Variables: CatBoost', fontsize=14)
    ax2.set_xlabel('Importancia (Feature Importance)')

    plt.suptitle('Comparativa de Importancia de Variables', fontsize=18, y=1.05)
    plt.tight_layout()
    plt.show()

