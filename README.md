# Modelo de Reclamos
Este repositorio contiene un modelo de reclamos de una compañía de seguros utilizando técnicas de data science. El objetivo del modelo es predecir si un cliente realizará un reclamo en el próximo año y utilizar esta información para tomar decisiones estratégicas.

## Estructura del Repositorio
- Data: Esta carpeta contiene la data en crudo y procesada que es utilizada como entrada para el modelamiento.

- Notebooks: En esta carpeta, se encuentran dos archivos. El archivo analisis_resultados.ipynb evalúa la predicción del modelo LightGBM optimizado y elabora una tabla de KS para determinar qué tan bien el modelo discrimina por decil. Es en esta tabla resumen donde se puede encontrar perfiles propensos a hacer un reclamo que podrían tener una tasa de reclamo mayor a la de toda la base. El otro archivo es el de exploratory_data.ipynb, que realiza la ingestación de la data cruda, un EDA y transformaciones de variables.

- entrenamiento.py: Este archivo contiene un pipeline para entrenar un modelo LightGBM y optimizarlo según Optuna. El modelo entrenado se guarda, junto con la importancia de variables. El AUC de este modelo es 0.585.

- outputs: Esta carpeta contiene varias subcarpetas:

    - hyperparameter_tuning: Aquí se guarda la combinación de hiperparámetros que se usó en las iteraciones para optimizar.

    - imp_variables: Aquí se guarda la importancia de variables.

    - ks: Aquí se guarda la tabla donde se saca éxitos por decil.

    - models: Aquí se guarda el modelo en un archivo de texto.

    - preds: Aquí se guardan las predicciones.

- prediction.py: Este archivo se utiliza para correr el modelo en producción.

- eval_resultados.xlsx: Este archivo es utilizado para evaluar los resultados del modelo.

- metrics.py: Este archivo contiene las métricas que se utilizaron para la medición del modelo LightGBM durante su optimización.

- requirements.txt: Este archivo contiene las librerías utilizadas en el proyecto.

## Resultados
Se evaluaron cuatro modelos: Regresión Logística, Decision Tree Classifier, Random Forest y LightGBM. El ranking de los modelos, de mayor a menor AUC, fue el siguiente:

1. Regresión Logística (AUC: 0.596)
2. Decision Tree Classifier (AUC: 0.593)
3. LightGBM (AUC: 0.561)
4. Random Forest (AUC: 0.560)
<br>Se eligió el AUC como métrica porque permite saber si el modelo discrimina los positivos y negativos y, además, se utiliza la probabilidad para identificar propensiones y priorizar perfiles según el modelo.

## Comparativo de Modelos por AUC

| Modelo | AUC |
|--------|-----|
| Regresión Logística | 0.596 |
| Decision Tree | 0.593 |
| Random Forest | 0.561 |
| LightGBM | 0.561 |
| LightGBM Optimizado | 0.585 |

Cabe mencionar que el modelo de LightGBM optimizado por Optuna se entrenó utilizando un pipeline y se guardó en la carpeta `models` dentro de la carpeta `outputs`. Además, la importancia de las variables utilizadas en este modelo se guardó en la subcarpeta `imp_variables` dentro de la misma carpeta.

Si deseas reproducir los resultados, asegúrate de tener instaladas las librerías listadas en `requirements.txt`. Luego, puedes ejecutar los archivos `entrenamiento.py` y `prediction.py` para entrenar el modelo de LightGBM optimizado y realizar predicciones en producción, respectivamente.

## Próximos Pasos
- Cambiar el método de tratamiento de categorías: Se puede utilizar el mean encoding o frequency encoding.
- Agregar más variables: Tal vez se podría agregar variables de interacciones o consultas por el seguro, tipo de venta, canales de contacto, tipo de pago, etc.
- Afinar la definición del target: Tal vez que no sea el próximo año sino el próximo mes.
- Tratamiento de outliers.