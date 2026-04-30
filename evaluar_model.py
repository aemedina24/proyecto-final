import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib.ticker import FuncFormatter


def evaluar_modelos(y_test_log, preds_log):
    y_real = np.expm1(y_test_log)
    y_pred = np.expm1(preds_log)
    
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    
    print(f"="*40)
    print(f"MÉTRICAS EN PESOS")
    print(f"MAE: ${mae:,.2f}")
    print(f"R2:  {r2:.4f}")
    print("="*40)

    plt.figure(figsize=(10, 6))
    # Usamos hexbin o scatter con alpha muy bajo si son muchos datos
    plt.scatter(y_real, y_pred, alpha=0.3, color='#2c3e50', s=10)
    
    # Línea de 45 grados (Ideal)
    max_val = max(y_real.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], color='#e74c3c', lw=2, linestyle='--')
    
    # AJUSTE DE ESCALA: Si tienes valores de hasta 1000M, usamos esta lógica
    def format_millones(x, pos):
        if x >= 1e9: return f'${x/1e9:.1f}B' # Para Billones si aplica
        return f'${x/1e6:.0f}M'

    fmt = FuncFormatter(format_millones)
    plt.gca().xaxis.set_major_formatter(fmt)
    plt.gca().yaxis.set_major_formatter(fmt)
    
    # Evitar que la nube de puntos toque los bordes
    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)

    plt.title('Valor Real vs. Predicho (Evaluación en Pesos)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()




def evaluar_modelo(y_test_log, preds_log):
    """Calcula métricas y genera la gráfica de dispersión."""
    y_real = np.expm1(y_test_log)
    y_pred = np.expm1(preds_log)
    
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    
    print(f"="*40)
    print(f"MÉTRICAS EN PESOS")
    print(f"MAE: ${mae:,.2f}")
    print(f"R2:  {r2:.4f}")
    print("="*40)

    # Gráfica de Dispersión
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_real, y=y_pred, alpha=0.4, color='#2c3e50')
    
    max_val = max(y_real.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], color='#e74c3c', linestyle='--')
    
    fmt = FuncFormatter(lambda x, pos: f'${x/1e6:.0f}M')
    plt.gca().xaxis.set_major_formatter(fmt)
    plt.gca().yaxis.set_major_formatter(fmt)
    
    plt.title('Valor Real vs. Predicho')
    plt.xlabel('Real (Millones)')
    plt.ylabel('Predicción (Millones)')
    plt.show()
    
    return {"MAE": mae, "R2": r2}

def graficar_importancia(model, feature_names):
    """Grafica las variables más influyentes."""
    importance = model.get_feature_importance()
    zipped = sorted(zip(feature_names, importance), key=lambda x: x[1])
    features, values = zip(*zipped)
    
    plt.figure(figsize=(10, 8))
    plt.barh(features, values, color='teal')
    plt.title('Importancia de las Variables en el Modelo')
    plt.show()
