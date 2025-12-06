Hola Juan,
Te adjunto el reporte de los avances del proyecto.

- Logramos crear un dataset utilizable para modelos supervisados, generamos dos versiones del dataset para comparar cuál represeta mejor la información de los datos dinámicos:
  - X_metrics: de cada feature se calculan métricas como media, mediana, desviación estándar, máximos, mínimos, entre otras.
  - X_fourier: por cada feature se calcula la Transformada de Fourier y se utilizan los coeficientes resultantes como features.

La gracia de hacer esto es lograr llevar cada csv dinámico a un sample que se pueda sumar a los datos estáticos. Ah y antes de hacer esto se separaron los datos dinámicos por viaje, ya que algunos csv tenían los datos mezclados. Esto resultó en un dataset final con más datos de los que pensábamos que teníamos (32k samples totales).

Ambas versiones se probaron en varios modelos, Regresión lineal (sin regularización), Ridge, Lasso, Random Forest, XGBoost, LightGBM y una MLP.

Para X_metrics y X_fourier se probaron todos estos modelos. Todavía está pendiente hacer un gráfico comparativo entre ambos rendimientos.

Para este fin de semana tenemos planeado:

- Mejorar los gráficos.

- Comparar el rendimiento de los modelos entrenados con X_metrics y X_fourier.

- Separar correctamente los datos. Los datos ahora se están separando en train/test/dev de forma independiente para cada modelo. La solución sería realizar esta separación una única vez al inicio del pipeline, para que todos los modelos se entrenen y evalúen con exactamente los mismos datos.

Gracias por leernos y quedamos atentos a tus comentarios.
