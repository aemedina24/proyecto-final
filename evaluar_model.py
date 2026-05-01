import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from matplotlib.ticker import FuncFormatter


def evaluar_modelo(y_test_log, preds_log):

    # Regresar de log a pesos reales
    y_real = np.expm1(y_test_log)
    y_pred = np.expm1(preds_log)

    # Métricas
    mae = mean_absolute_error(y_real, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_real, y_pred)
    )

    r2 = r2_score(y_real, y_pred)

    # Mostrar métricas
    print("=" * 45)
    print("MÉTRICAS DEL MODELO")
    print(f"MAE : ${mae:,.0f}")
    print(f"RMSE: ${rmse:,.0f}")
    print(f"R2  : {r2:.4f}")
    print("=" * 45)

    # Figura
    plt.figure(figsize=(10, 7))

    # Scatter
    plt.scatter(
        y_real,
        y_pred,
        alpha=0.25,
        s=12
    )

    # Línea ideal
    max_val = 200_000_000

    plt.plot(
        [0, max_val],
        [0, max_val],
        linestyle='--',
        linewidth=2
    )

    # Formato millones
    def formato_millones(x, pos):

        return f'${x/1e6:.0f}M'

    fmt = FuncFormatter(formato_millones)

    ax = plt.gca()

    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    # Límites
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)

    # Labels
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Predicho")

    plt.title(
        "Valor Real vs Valor Predicho"
    )

    plt.grid(True, linestyle=':')

    plt.show()



def graficar_importancia(model, feature_names):
    """Grafica las variables más influyentes."""
    importance = model.get_feature_importance()
    zipped = sorted(zip(feature_names, importance), key=lambda x: x[1])
    features, values = zip(*zipped)
    
    plt.figure(figsize=(10, 8))
    plt.barh(features, values, color='teal')
    plt.title('Importancia de las Variables en el Modelo')
    plt.show()
