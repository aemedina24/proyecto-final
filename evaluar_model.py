from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

def evaluar_modelo(y_test, preds_log):
    """
    Convierte los datos de escala logarítmica a real y calcula métricas de desempeño.
    """
    # 1. Revertir la transformación logarítmica (log1p -> expm1)
    y_real = np.expm1(y_test)
    preds_real = np.expm1(preds_log)

    # 2. Calcular métricas de regresión
    r2 = r2_score(y_real, preds_real)
    mae = mean_absolute_error(y_real, preds_real)
    rmse = np.sqrt(mean_squared_error(y_real, preds_real))
    mape = mean_absolute_percentage_error(y_real, preds_real) * 100

    # 3. Imprimir resultados con formato profesional
    print("--- Evaluación del Modelo (Escala Real) ---")
    print(f"R² (Varianza explicada): {r2:.4f}")
    print(f"MAE (Error promedio):    ${mae:,.2f}")
    print(f"RMSE (Penaliza errores): ${rmse:,.2f}")
    print(f"MAPE (Error porcentual): {mape:.2f}%")
    print("-------------------------------------------")
    
    return {"r2": r2, "mae": mae, "rmse": rmse, "mape": mape}

# --- CÓMO USARLA ---
# Una vez entrenes tu modelo y obtengas las predicciones:
# preds_log = model.predict(X_test)
# resultados = evaluar_modelo(y_test, preds_log)
