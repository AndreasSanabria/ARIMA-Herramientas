from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from tabulate import tabulate
import numpy as np

# Datos
def extraerDatos(csv):
    return read_csv(csv, header=0, parse_dates=[0], index_col=0)

# Función
def iniciarARIMA(Real, P, D, Q):
    modelo = ARIMA(Real, order=(P, D, Q))
    modelo_fit = modelo.fit()
    prediccion = modelo_fit.forecast()[0]
    return prediccion

# Tasas de cambio
datos = extraerDatos('FRB_H10_modified.csv')

# Dividir datos de entrenamiento y validación
entrenamiento_inicio = '2008-01-01'
entrenamiento_final = '2022-05-20'
prueba_inicio = '2022-05-21'
prueba_final = '2023-05-21'

datos_entrenamiento = datos.loc[entrenamiento_inicio:entrenamiento_final]['Rate'].values
datos_prueba = datos.loc[prueba_inicio:prueba_final]['Rate'].values
fechas_prueba = datos.loc[prueba_inicio:prueba_final].index

# Guardar datos reales y predichos
Real = [x for x in datos_entrenamiento]
predicciones = []

# ARIMA
for timepoint in range(len(datos_prueba)):
    valor_real = datos_prueba[timepoint]
    prediccion = iniciarARIMA(Real, 3, 1, 1)
    print('Real = %f, Predicha = %f' % (valor_real, prediccion))
    predicciones.append(prediccion)
    Real.append(valor_real)

# Gráfico
pyplot.plot(fechas_prueba, datos_prueba)
pyplot.plot(fechas_prueba, predicciones, color='red')
pyplot.xlabel('Fecha')
pyplot.ylabel('Tasa de cambio')
pyplot.suptitle('Predicción de la tasa de cambio a un año')
pyplot.title('Entrenando el modelo del 2008 al 2022')
pyplot.legend(['Real', 'Predicha'])
pyplot.xticks(rotation=45)
pyplot.show()

# Tabla
tabla = zip(fechas_prueba, datos_prueba, predicciones)
headers = ["Fecha", "Valores reales", "Valores predichos"]
tabla_final = tabulate(tabla, headers=headers, floatfmt=".6f")
print(tabla_final)

# MSE
print(np.square(np.subtract(predicciones, datos_prueba)).mean())
