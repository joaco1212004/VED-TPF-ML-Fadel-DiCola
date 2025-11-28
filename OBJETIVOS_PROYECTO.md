# Objetivos del Proyecto Final - Vehicle Energy Dataset

## Estructura General del Proyecto

### Objetivo Principal
Modelar el consumo energético de vehículos (combustión, híbridos y eléctricos) a partir de variables de conducción, características del vehículo y condiciones ambientales utilizando técnicas de Machine Learning y Deep Learning.

---

## FASE 1: Exploración y Análisis Inicial de Datos

### 1.1 Carga y Comprensión del Dataset
- [x] Descargar y cargar el Vehicle Energy Dataset (VED)
- [ ] Identificar y documentar la estructura de datos estáticos vs dinámicos
- [ ] Analizar dimensiones del dataset (número de registros, vehículos, viajes)
- [ ] Documentar diccionario de variables con descripción de cada feature
- [ ] Identificar tipos de datos (numéricos, categóricos, temporales)

### 1.2 Análisis Exploratorio de Datos (EDA) - Estadísticas Básicas
- [ ] Calcular estadísticas descriptivas (media, mediana, desviación estándar, min, max)
- [ ] Analizar distribución de variables continuas con histogramas y boxplots
- [ ] Examinar balance de clases en variables categóricas (tipo de vehículo, tipo de ruta)
- [ ] Identificar rangos razonables para cada variable

### 1.3 EDA - Análisis de Valores Faltantes y Calidad de Datos
- [ ] Generar mapa de calor de valores faltantes por variable
- [ ] Calcular porcentaje de missingness por columna y por vehículo
- [ ] Identificar patrones en datos faltantes (MCAR, MAR, MNAR)
- [ ] Documentar variables con >50% de missingness (candidatas a eliminar)

### 1.4 EDA - Análisis de Valores Atípicos
- [ ] Detectar outliers usando método IQR y Z-score
- [ ] Visualizar outliers con boxplots por variable y tipo de vehículo
- [ ] Analizar si outliers son errores de medición o eventos reales
- [ ] Documentar decisiones sobre tratamiento de outliers

### 1.5 EDA - Análisis de Distribuciones por Tipo de Vehículo
- [ ] Comparar distribución de consumo entre eléctricos, híbridos y combustión
- [ ] Analizar patrones de velocidad por tipo de vehículo
- [ ] Visualizar diferencias en aceleración y desaceleración
- [ ] Examinar uso de potencia auxiliar (AC, calefacción) por tipo

### 1.6 EDA - Análisis de Correlaciones
- [ ] Generar matriz de correlación entre variables numéricas
- [ ] Visualizar heatmap de correlaciones con valores significativos
- [ ] Identificar multicolinealidad entre features (VIF > 10)
- [ ] Analizar correlación de cada feature con variable target
- [ ] Documentar features redundantes candidatas a eliminar

### 1.7 EDA - Análisis Temporal y de Trayectorias
- [ ] Analizar duración promedio de viajes por tipo de vehículo
- [ ] Visualizar 5-10 trayectorias ejemplo (velocidad vs tiempo)
- [ ] Identificar patrones de conducción (agresiva vs conservadora)
- [ ] Analizar estacionalidad en consumo por temperatura ambiente

### 1.8 EDA - Análisis Geográfico y de Rutas
- [ ] Clasificar trayectos por tipo (urbano, suburbano, autopista)
- [ ] Analizar consumo promedio por tipo de ruta
- [ ] Visualizar distribución geográfica de trayectos (si hay coordenadas GPS)
- [ ] Identificar rutas más eficientes energéticamente

---

## FASE 2: Limpieza y Curación de Datos

### 2.1 Tratamiento de Valores Faltantes
- [ ] Implementar estrategia para variables con <5% missingness (imputación)
- [ ] Eliminar variables con >70% de valores faltantes
- [ ] Aplicar imputación por mediana/media para variables numéricas
- [ ] Aplicar imputación por moda para variables categóricas
- [ ] Considerar imputación por KNN o modelo predictivo para casos complejos
- [ ] Documentar todas las decisiones de imputación

### 2.2 Tratamiento de Outliers
- [ ] Aplicar winsorization (clip a percentiles 1-99) para variables sensibles
- [ ] Eliminar registros con valores físicamente imposibles
- [ ] Mantener outliers válidos (ej: consumo alto en aceleraciones bruscas)
- [ ] Documentar impacto de remoción de outliers en distribuciones

### 2.3 Limpieza de Datos Inconsistentes
- [ ] Verificar rangos válidos (velocidad ≥ 0, temperatura en rango razonable)
- [ ] Corregir unidades inconsistentes si existen
- [ ] Eliminar registros duplicados
- [ ] Validar coherencia temporal (timestamps ordenados)

### 2.4 Normalización y Estandarización
- [ ] Aplicar StandardScaler a variables con distribución normal
- [ ] Aplicar MinMaxScaler a variables con rango fijo conocido
- [ ] Aplicar RobustScaler a variables con outliers persistentes
- [ ] Guardar scalers para uso en producción

### 2.5 Codificación de Variables Categóricas
- [ ] One-Hot Encoding para variables con <10 categorías (tipo de ruta)
- [ ] Label Encoding para variables ordinales si existen
- [ ] Target Encoding para variables de alta cardinalidad si aplica
- [ ] Documentar mapeos de codificación

---

## FASE 3: Feature Engineering

### 3.1 Definición de Variable Target
- [ ] Calcular consumo energético por trayecto (kWh/km o L/100km)
- [ ] Crear variable target continua para regresión
- [ ] Crear variable target categórica para clasificación (Alta/Media/Baja eficiencia)
- [ ] Definir umbrales para categorización basados en percentiles (33%, 66%)
- [ ] Analizar distribución de variable target y balance de clases

### 3.2 Agregación de Datos Temporales a Nivel Trayecto
- [ ] Agrupar datos por `trip_id` para crear features agregadas
- [ ] Calcular velocidad promedio, máxima y mínima por trayecto
- [ ] Calcular aceleración promedio, máxima y varianza por trayecto
- [ ] Calcular distancia total del trayecto
- [ ] Calcular duración total del trayecto
- [ ] Calcular temperatura promedio durante el trayecto

### 3.3 Features Derivadas - Comportamiento de Conducción
- [ ] Crear ratio velocidad/aceleración promedio
- [ ] Calcular porcentaje de tiempo en aceleración vs desaceleración
- [ ] Crear indicador de conducción agresiva (aceleraciones bruscas)
- [ ] Calcular número de paradas completas (velocidad = 0)
- [ ] Crear feature de "suavidad de conducción" (varianza de velocidad)
- [ ] Calcular tiempo en diferentes rangos de velocidad (0-30, 30-60, 60+ km/h)

### 3.9 Features en el Dominio de la Frecuencia (Análisis de Fourier)
- [ ] Justificación: muchas señales dinámicas (velocidad, aceleración, consumo instantáneo) contienen componentes periódicos o armónicos relacionados con patrones de conducción (ciclos de aceleración/frenado, comportamiento en autopista vs urbano); el análisis espectral permite extraer información complementaria o alternativa a las agregaciones temporales.
- [ ] Extracción básica por trayecto (`trip_id`): calcular FFT/PSD de series temporales (ej. velocidad, aceleración, fuel_rate, ac_power)
- [ ] Features sugeridas a extraer por serie y por trayecto:
  - Dominant frequency (Hz) y su amplitud
  - Top-K peaks (frecuencias y amplitudes)
  - Total spectral power (band power) en bandas definidas (baja frecuencia, media, alta)
  - Spectral centroid (centro de masa del espectro)
  - Spectral entropy (medida de dispersión de potencia espectral)
  - Ratio de potencia entre bandas (ej. BF/MF/HF)
  - Energy of harmonics (par/ímpar) y relación señal/ruido espectral
- [ ] Procedimiento:
  - Re-muestrear la serie a frecuencia uniforme (p. ej. 1 Hz) si fuese necesario
  - Aplicar ventana (Hann) y calcular PSD con Welch o FFT con zero-padding
  - Extraer features numéricos y guardarlos a nivel `trip_id`
  - Normalizar/estandarizar features espectrales antes de entrenar modelos
- [ ] Usos:
  - Complementar features temporales tradicionales en modelos supervisados
  - Reemplazar algunas agregaciones si la representación espectral resulta más informativa
  - Detección de anomalías usando reconstrucción en dominio espectral o thresholds de bandpower
- [ ] Guardar funciones reutilizables en `src/fourier_features.py` y ejemplos en `notebooks/08_Fourier_Analysis.ipynb`


### 3.4 Features Derivadas - Energía y Eficiencia
- [ ] Calcular energía cinética promedio (0.5 * m * v²)
- [ ] Crear ratio potencia auxiliar / consumo total
- [ ] Calcular eficiencia regenerativa para vehículos eléctricos/híbridos
- [ ] Para híbridos: crear ratio uso motor eléctrico vs combustión
- [ ] Para eléctricos: crear features de estado de batería (SOC promedio, variación)

### 3.5 Features Derivadas - Condiciones Ambientales
- [ ] Binning de temperatura en rangos (frío <10°C, templado 10-25°C, calor >25°C)
- [ ] Crear indicador de uso de AC/calefacción basado en temperatura
- [ ] Calcular variación de temperatura durante el trayecto

### 3.6 Features Derivadas - Características de Ruta
- [ ] Crear ratio distancia/duración (velocidad efectiva)
- [ ] Calcular tortuosidad de ruta (cambios de dirección) si hay datos GPS
- [ ] Crear indicador de ruta urbana vs autopista basado en velocidad
- [ ] Calcular elevación ganada/perdida si hay datos de altitud

### 3.7 Features de Interacción
- [ ] Interacción tipo_vehiculo × velocidad_promedio
- [ ] Interacción temperatura × potencia_auxiliar
- [ ] Interacción tipo_ruta × aceleracion_promedio
- [ ] Interacción peso_vehiculo × aceleracion_promedio

### 3.8 Selección de Features
- [ ] Aplicar análisis de importancia con Random Forest inicial
- [ ] Aplicar Recursive Feature Elimination (RFE)
- [ ] Calcular Variance Inflation Factor (VIF) para eliminar multicolinealidad
- [ ] Seleccionar top 15-25 features más relevantes
- [ ] Documentar justificación de features seleccionadas

---

## FASE 4: Preparación de Datasets

### 4.1 Estrategia de Muestreo para Desarrollo
- [ ] Crear dataset de desarrollo con 5,000-10,000 muestras
- [ ] Asegurar representatividad por tipo de vehículo en muestra
- [ ] Asegurar representatividad por tipo de ruta en muestra
- [ ] Aplicar stratified sampling basado en variable target
- [ ] Guardar índices de muestras seleccionadas

### 4.2 División Train-Validation-Test (Desarrollo)
- [ ] Separar 20% como test set (reservado hasta evaluación final)
- [ ] Del 80% restante (dev set), dividir en:
  - [ ] 80% train (64% del total)
  - [ ] 20% validation (16% del total)
- [ ] Aplicar stratified split para mantener distribuciones
- [ ] Verificar que vehículos no se repitan entre conjuntos (data leakage)
- [ ] Guardar splits en archivos separados

### 4.3 Dataset Completo para Evaluación Final
- [ ] Definir tamaño de dataset completo (50,000-100,000+ muestras)
- [ ] Aplicar misma estrategia de split al dataset completo
- [ ] Reservar test set final sin tocar hasta última fase
- [ ] Documentar tamaños finales de cada conjunto

---

## FASE 5: Modelado - Enfoque Supervisado (alineado con I302)

Esta fase se centra en los modelos y conceptos vistos en el programa I302: regresión lineal y regularizada, regresión no lineal y no paramétrica, clasificación discriminativa y generativa, máquinas de soporte vectorial, vecinos más cercanos, árboles y ensembles.

### 5.0 Baseline y Diagnóstico
- [ ] Implementar predictor dummy (media) y predictor por grupo (media por tipo de vehículo)
- [ ] Calcular métricas baseline: RMSE, MAE, R², MAPE y usar como referencia mínima
- [ ] Analizar bias-variance del baseline y establecer umbrales mínimos de mejora

### 5.1 Regresión (temas vistos)
- [ ] Regresión Lineal ordinaria (OLS) — derivación y evaluación
- [ ] Regresión regularizada: Ridge (L2), Lasso (L1) — interpretación Bayesiana / MAP
- [ ] Polinomios y regresión no lineal (features polinómicas)
- [ ] Modelos no-paramétricos para regresión: KNN regressor, kernel regression (si aplica)
- [ ] Validación: k-fold Cross-Validation, curvas de validación (learning curves), diagnóstico de over/underfitting
- [ ] Métricas: RMSE, MAE, R², MAPE; análisis de residuos y homocedasticidad

### 5.2 Clasificación (si se elige este enfoque)
- [ ] Regresión logística (binaria y multiclase con one-vs-rest / softmax)
- [ ] Modelos generativos: GDA / LDA, Naive Bayes — revisión teórica y aplicación
- [ ] KNN para clasificación y tratamiento de desbalance (stratified sampling, reweighting)
- [ ] SVM para clasificación (margen, kernels, regularización)
- [ ] Métricas: accuracy, precision, recall, F1, ROC-AUC, matriz de confusión

### 5.3 Árboles y Ensembles (visto en curso)
- [ ] Árboles de decisión para regresión y clasificación — interpretación y poda
- [ ] Random Forest: bagging, OOB error, feature importance
- [ ] Boosting y Gradient Boosting (intuición, regularización): XGBoost/LightGBM como implementaciones prácticas
- [ ] Stacking básico (cuando sea relevante)

### 5.4 Evaluación y Buenas Prácticas
- [ ] Fijar semillas y evitar data leakage (mismos vehículos entre splits)
- [ ] Usar validación estratificada cuando aplica (por clase o por percentiles del target)
- [ ] Curvas ROC/PR, análisis por subgrupos (tipo veh. / tipo ruta)
- [ ] Interpretar resultados desde la perspectiva de bias-variance y del dominio físico

---

## FASE 6: Modelado - Aprendizaje Profundo (alineado con I302)

La sección de Aprendizaje Profundo se centrará en MLPs y autoencoders, utilizando los conceptos teóricos vistos en clase (backpropagation, SGD, regularización, normalización y double descent).

### 6.1 Perceptrón multicapa (MLP) para regresión y clasificación
- [ ] Diseñar MLPs adecuados para regresión: arquitecturas simples (ej. 2-4 capas denses)
- [ ] Normalización de inputs (batch normalization / standardization) y su efecto en el entrenamiento
- [ ] Regularización: weight decay (L2), dropout, early stopping
- [ ] Optimización: SGD con momentum, Adam; tuning de learning rate; scheduling
- [ ] Monitoreo: curvas de entrenamiento/validación, detección de overfitting y double descent
- [ ] Evaluación: usar MSE/MAE para regresión, cross-entropy y métricas de clasificación si aplica

### 6.2 Autoencoders y VAE (reducción de dimensionalidad y detección de anomalías)
- [ ] Autoencoder determinista para reducción dimensional y extracción de features
- [ ] Variational Autoencoder (VAE): reparameterization trick y loss = reconstruction + KL
- [ ] Uso de representaciones latentes como inputs para modelos supervisados (pipeline AE → regresión)
- [ ] Uso del reconstruction error para detección de anomalías (thresholding)

### 6.3 Buenas prácticas de entrenamiento
- [ ] Fijar seed, usar batch size apropiado y normalizar features antes de entrenar
- [ ] Early stopping con patience y checkpoints de modelo
- [ ] Registrar experimentos (tensorboard/MLflow/CSV) para reproducibilidad

### 6.4 Ámbitos que no son foco del PF
- [ ] GANs y arquitecturas muy avanzadas quedan fuera salvo que el alumno demuestre motivación y tiempo adicional

---

## FASE 7: Definición de Loss Functions y Métricas

### 7.1 Loss Functions para Entrenamiento

#### Para Regresión:
- [ ] **MSE (Mean Squared Error)**: Loss principal para NN
- [ ] **MAE (Mean Absolute Error)**: Loss alternativo más robusto a outliers
- [ ] **Huber Loss**: Combina MSE y MAE, robusto a outliers
- [ ] **MAPE Loss**: Para penalizar errores relativos

#### Para Autoencoders:
- [ ] **Reconstruction MSE**: Para AE estándar
- [ ] **VAE Loss**: Reconstruction + β×KL_divergence
- [ ] Experimentar con diferentes valores de β (0.5, 1.0, 2.0)

### 7.2 Métricas de Evaluación para Regresión
- [ ] **RMSE (Root Mean Squared Error)**: Métrica principal
- [ ] **MAE (Mean Absolute Error)**: Interpretable en unidades originales
- [ ] **R² Score**: Proporción de varianza explicada
- [ ] **MAPE (Mean Absolute Percentage Error)**: Error porcentual
- [ ] **Max Error**: Peor predicción del modelo
- [ ] Calcular métricas en train, validation y test

### 7.3 Métricas de Evaluación para Clasificación (si aplica)
- [ ] **Accuracy**: Proporción de predicciones correctas
- [ ] **Precision, Recall, F1-Score**: Por clase
- [ ] **Confusion Matrix**: Visualización de errores
- [ ] **ROC-AUC**: Curva ROC multi-clase

### 7.4 Métricas Específicas del Dominio
- [ ] Error absoluto en L/100km o kWh/km
- [ ] Porcentaje de predicciones dentro de ±10% del valor real
- [ ] Error promedio por tipo de vehículo
- [ ] Error promedio por tipo de ruta

---

## FASE 8: Comparación de Modelos

### 8.1 Tabla Comparativa de Performance
- [ ] Crear tabla con RMSE, MAE, R² para todos los modelos
- [ ] Incluir tiempo de entrenamiento de cada modelo
- [ ] Incluir tiempo de inferencia (predicción)
- [ ] Destacar mejor modelo por métrica

### 8.2 Análisis Cualitativo de Predicciones
- [ ] Graficar predicciones vs valores reales (scatter) por modelo
- [ ] Analizar en qué rangos de consumo cada modelo falla más
- [ ] Identificar patrones en errores (¿subestima o sobreestima?)
- [ ] Comparar distribución de errores entre modelos (boxplot)

### 8.3 Comparación AE vs VAE vs Features Originales
- [ ] Tabla comparativa de performance downstream con cada representación
- [ ] Visualizar espacio latente de AE vs VAE (t-SNE o PCA)
- [ ] Analizar interpretabilidad de features latentes
- [ ] Evaluar si la reducción de dimensionalidad ayuda o perjudica

### 8.4 Análisis de Complejidad vs Performance
- [ ] Graficar trade-off entre complejidad (# parámetros) y performance
- [ ] Evaluar si modelos más complejos justifican ganancia marginal
- [ ] Considerar trade-off interpretabilidad vs precisión

---

## FASE 9: Visualizaciones y Gráficos para el Informe

### 9.1 Gráficos de Análisis Exploratorio
- [ ] **Histogramas**: Distribución de consumo por tipo de vehículo
- [ ] **Boxplots**: Consumo por tipo de ruta y tipo de vehículo
- [ ] **Heatmap**: Matriz de correlación entre features principales
- [ ] **Scatter matrix**: Relaciones entre top 4-5 features y target
- [ ] **Series temporales**: 3-5 trayectorias ejemplo mostrando velocidad, aceleración, consumo

### 9.2 Gráficos de Feature Engineering
- [ ] **Bar plot**: Feature importance de Random Forest
- [ ] **Bar plot**: Coeficientes de Lasso/Ridge (top 10 features)
- [ ] **Violin plot**: Distribución de features clave por categoría de eficiencia

### 9.3 Gráficos de Performance de Modelos
- [ ] **Bar chart**: Comparación de RMSE entre todos los modelos (incluir baseline dummy)
- [ ] **Bar chart**: Comparación de R² entre todos los modelos (baseline tendrá R²≈0)
- [ ] **Line chart**: Mejora relativa (%) vs baseline para cada modelo
- [ ] **Scatter plot**: Predicciones vs Valores Reales (para mejor modelo)
- [ ] **Scatter plot**: Comparación baseline vs mejor modelo (lado a lado)
- [ ] **Residual plot**: Análisis de residuos del mejor modelo
- [ ] **Error distribution**: Histograma de errores por modelo

### 9.4 Gráficos de Deep Learning
- [ ] **Learning curves**: Loss train vs validation por época (para NN, AE, VAE)
- [ ] **Scatter plot 2D**: Espacio latente de VAE coloreado por tipo de vehículo
- [ ] **Reconstruction examples**: Original vs Reconstruido para AE/VAE (3-5 ejemplos)
- [ ] **Bar chart**: Comparación AE vs VAE vs Original features

### 9.5 Gráficos de Análisis de Resultados
- [ ] **Box plot**: Error por tipo de vehículo del mejor modelo
- [ ] **Box plot**: Error por tipo de ruta del mejor modelo
- [ ] **Heatmap**: Confusion matrix (si se hace clasificación)
- [ ] **Curva ROC**: Multi-clase (si se hace clasificación)
- [ ] **Mapas de calor**: Consumo vs velocidad vs temperatura (3D surface o heatmap 2D)

### 9.6 Gráficos de Feature Importance y SHAP
- [ ] **SHAP summary plot**: Importancia global de features
- [ ] **SHAP dependence plot**: Para top 3 features más importantes
- [ ] **SHAP force plot**: Explicación de 2-3 predicciones individuales

---

## FASE 10: Hyperparameter Tuning

### 10.1 Tuning de Random Forest
- [ ] Grid/Random Search sobre:
  - `n_estimators`: [100, 200, 500]
  - `max_depth`: [10, 20, 30, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
- [ ] Usar 5-fold Cross-Validation
- [ ] Documentar mejores hiperparámetros encontrados

### 10.2 Tuning de XGBoost/LightGBM
- [ ] Grid/Random Search sobre:
  - `n_estimators`: [100, 200, 500]
  - `learning_rate`: [0.01, 0.05, 0.1]
  - `max_depth`: [3, 5, 7, 10]
  - `subsample`: [0.7, 0.8, 0.9, 1.0]
  - `colsample_bytree`: [0.7, 0.8, 0.9, 1.0]
- [ ] Implementar Early Stopping
- [ ] Documentar mejores hiperparámetros

### 10.3 Tuning de Neural Networks
- [ ] Búsqueda sobre:
  - Arquitectura: número de capas [3, 4, 5]
  - Neuronas por capa: [32, 64, 128, 256]
  - Dropout rate: [0.1, 0.2, 0.3, 0.4]
  - Learning rate: [0.0001, 0.001, 0.01]
  - Batch size: [32, 64, 128]
- [ ] Usar Optuna, Keras Tuner o similar
- [ ] Documentar mejor arquitectura

### 10.4 Tuning de Autoencoders
- [ ] Búsqueda sobre:
  - Dimensión del bottleneck: [8, 16, 32]
  - Arquitectura del encoder: diferentes profundidades
  - Learning rate: [0.0001, 0.001, 0.01]
  - β para VAE: [0.5, 1.0, 2.0, 5.0]
- [ ] Evaluar basado en reconstruction loss
- [ ] Documentar mejor configuración

---

## FASE 11: Evaluación Final

### 11.1 Entrenamiento Final con Dataset Completo
- [ ] Entrenar mejor modelo en dataset completo (train + validation)
- [ ] Usar hiperparámetros óptimos encontrados
- [ ] Monitorear tiempo de entrenamiento y recursos utilizados
- [ ] Guardar modelo entrenado final

### 11.2 Evaluación en Test Set
- [ ] Cargar test set reservado (nunca usado hasta ahora)
- [ ] Generar predicciones con modelo final
- [ ] Calcular todas las métricas (RMSE, MAE, R², MAPE)
- [ ] Comparar con resultados en validation set

### 11.3 Análisis de Errores Detallado
- [ ] Identificar top 10% peores predicciones
- [ ] Analizar características comunes de casos mal predichos
- [ ] Investigar posibles causas de errores sistemáticos
- [ ] Proponer mejoras futuras basadas en análisis

### 11.4 Evaluación por Subgrupos
- [ ] Calcular métricas separadas por tipo de vehículo
- [ ] Calcular métricas separadas por tipo de ruta
- [ ] Identificar subgrupos donde el modelo funciona mejor/peor
- [ ] Analizar si hay sesgo en predicciones

---

## FASE 12: Interpretabilidad y Explicación

### 12.1 Feature Importance Global
- [ ] Extraer feature importance del mejor modelo
- [ ] Generar ranking de top 15 features más importantes
- [ ] Interpretar desde perspectiva física/ingenieril cada feature importante
- [ ] Validar si features importantes tienen sentido con conocimiento del dominio

### 12.2 SHAP Values (SHapley Additive exPlanations)
- [ ] Calcular SHAP values para modelo final
- [ ] Generar SHAP summary plot (importancia global)
- [ ] Generar SHAP dependence plots para top 3 features
- [ ] Analizar interacciones entre features reveladas por SHAP
- [ ] Explicar 3-5 predicciones individuales con SHAP force plots

### 12.3 Partial Dependence Plots
- [ ] Generar PDP para top 5 features más importantes
- [ ] Analizar relación marginal entre cada feature y predicción
- [ ] Identificar umbrales o rangos críticos en las variables

### 12.4 Interpretación de Resultados
- [ ] Explicar qué factores más influyen en el consumo energético
- [ ] Comparar diferencias entre vehículos eléctricos, híbridos y combustión
- [ ] Analizar impacto de condiciones ambientales (temperatura)
- [ ] Interpretar efecto del estilo de conducción (agresivo vs suave)

---

## FASE 13: Extensiones y Análisis Avanzado

### 13.1 Análisis Comparativo Eléctricos vs Híbridos vs Combustión
- [ ] Entrenar modelos separados por tipo de vehículo
- [ ] Comparar features importantes en cada tipo
- [ ] Analizar si factores de eficiencia difieren entre tipos
- [ ] Cuantificar diferencias promedio en consumo

### 13.2 Mapas de Calor de Eficiencia
- [ ] Crear heatmap 2D: Consumo vs Velocidad vs Temperatura
- [ ] Identificar "zona óptima" de operación para cada tipo de vehículo
- [ ] Visualizar cómo temperatura afecta eficiencia energética
- [ ] Crear curvas de eficiencia por rango de velocidad

### 13.3 Análisis de Impacto de Variables Externas
- [ ] Cuantificar impacto de temperatura en consumo (regresión parcial)
- [ ] Analizar efecto de potencia auxiliar (AC/calefacción)
- [ ] Evaluar diferencias entre tipos de ruta (urbano vs autopista)
- [ ] Estimar potencial de ahorro energético bajo condiciones óptimas

### 13.4 Detección de Anomalías con Autoencoders
- [ ] Usar reconstruction error de AE/VAE como score de anomalía
- [ ] Definir threshold para anomalías (percentil 95 de reconstruction error)
- [ ] Identificar trayectos anómalos en test set
- [ ] Analizar características de trayectos anómalos
- [ ] Visualizar ejemplos de trayectos normales vs anómalos

### 13.5 Escenarios de Optimización
- [ ] Simular escenarios: ¿Qué pasa si reducimos velocidad promedio en 10%?
- [ ] Estimar ahorro energético de conducción más suave
- [ ] Calcular impacto de eliminar uso de AC en días templados
- [ ] Proponer recomendaciones concretas para mejorar eficiencia

---

## FASE 14: Documentación del Informe

### 14.1 Estructura del Informe
- [ ] **Resumen ejecutivo** (1 página): Problema, enfoque, resultados clave
- [ ] **Introducción**: Contexto, motivación, objetivos
- [ ] **Dataset**: Descripción, fuente, características
- [ ] **Análisis exploratorio**: Insights principales con visualizaciones
- [ ] **Preprocesamiento**: Limpieza, curación, decisiones tomadas
- [ ] **Feature Engineering**: Features creadas, selección, justificación
- [ ] **Metodología**: Modelos probados, arquitecturas, hiperparámetros
- [ ] **Resultados**: Comparación de modelos, métricas, visualizaciones
- [ ] **Análisis de resultados**: Interpretación, SHAP, feature importance
- [ ] **Discusión**: Limitaciones, sesgos, mejoras futuras
- [ ] **Conclusiones**: Hallazgos clave, respuesta a objetivos
- [ ] **Referencias**: Papers, documentación, recursos utilizados

### 14.2 Secciones Críticas del Informe

#### Tabla de Comparación de Modelos
| Modelo | RMSE | MAE | R² | Tiempo Train | Complejidad |
|--------|------|-----|----|--------------| ------------|
| **Baseline (Media)** | ... | ... | 0.00 | <1s | Mínima |
| **Baseline (Por Grupo)** | ... | ... | ... | <1s | Mínima |
| Linear Regression | ... | ... | ... | ... | Baja |
| Random Forest | ... | ... | ... | ... | Media |
| XGBoost | ... | ... | ... | ... | Media-Alta |
| Neural Network | ... | ... | ... | ... | Alta |
| NN + AE features | ... | ... | ... | ... | Alta |
| NN + VAE features | ... | ... | ... | ... | Alta |

#### Resultados Clave a Reportar
- [ ] **Métricas del baseline** (media y por grupo) como referencia mínima
- [ ] Modelo con mejor performance (RMSE en test)
- [ ] Top 5 features más importantes
- [ ] Error promedio por tipo de vehículo
- [ ] **Mejora porcentual respecto a baseline** (ej: 45% reducción en RMSE)
- [ ] Tiempo de inferencia del modelo final

### 14.3 Checklist de Calidad del Informe
- [ ] Todas las figuras tienen título, ejes etiquetados y leyenda
- [ ] Todas las tablas tienen caption descriptivo
- [ ] Código está documentado y reproducible
- [ ] Decisiones metodológicas están justificadas
- [ ] Resultados numéricos tienen precisión apropiada (2-3 decimales)
- [ ] Se discuten limitaciones y sesgos del estudio
- [ ] Se proponen trabajos futuros
- [ ] Referencias están formateadas correctamente
- [ ] Informe tiene narrativa coherente (no solo listado de gráficos)

---

## FASE 15: Aspectos Técnicos y Reproducibilidad

### 15.1 Organización del Código
- [ ] Estructura de carpetas clara:
  ```
  TP_Final/
  ├── data/
  │   ├── raw/
  │   ├── processed/
  │   └── splits/
  ├── notebooks/
  │   ├── 01_EDA.ipynb
  │   ├── 02_Preprocessing.ipynb
  │   ├── 03_Feature_Engineering.ipynb
  │   ├── 04_Modeling_Classical.ipynb
  │   ├── 05_Modeling_DL.ipynb
  │   ├── 06_Evaluation.ipynb
  │   └── 07_Analysis.ipynb
  ├── src/
  │   ├── data_processing.py
  │   ├── feature_engineering.py
  │   ├── models.py
  │   ├── evaluation.py
  │   └── visualization.py
  ├── models/
  │   └── saved_models/
  ├── results/
  │   ├── figures/
  │   └── metrics/
  ├── requirements.txt
  └── README.md
  ```

### 15.2 Control de Versiones y Reproducibilidad
- [ ] Usar Git para control de versiones
- [ ] Fijar random seeds (42) en todos los experimentos
- [ ] Documentar versiones de librerías en `requirements.txt`
- [ ] Guardar configuraciones de modelos en archivos JSON
- [ ] Documentar hardware utilizado (CPU/GPU)

### 15.3 Guardado de Artefactos
- [ ] Guardar scalers entrenados (pickle/joblib)
- [ ] Guardar modelos finales (pickle/h5/pt)
- [ ] Guardar splits de datos (índices)
- [ ] Guardar métricas en CSV para referencia
- [ ] Guardar todas las figuras en alta resolución (300 DPI)

---

## Checklist Final de Entrega

### Entregables Requeridos
- [ ] **Informe en PDF** (15-25 páginas)
- [ ] **Código fuente** (notebooks + scripts .py)
- [ ] **README.md** con instrucciones de reproducción
- [ ] **requirements.txt** o environment.yml
- [ ] **Presentación** (slides, 10-15 minutos)
- [ ] **Modelos entrenados** (si el tamaño lo permite)

### Criterios de Evaluación a Cubrir
- [ ] Calidad del análisis exploratorio ✓
- [ ] Preprocesamiento y limpieza adecuados ✓
- [ ] Feature engineering creativo y justificado ✓
- [ ] Variedad de modelos comparados (clásicos + DL) ✓
- [ ] Evaluación rigurosa con métricas apropiadas ✓
- [ ] Interpretabilidad y explicación de resultados ✓
- [ ] Visualizaciones claras y profesionales ✓
- [ ] Calidad de escritura del informe ✓
- [ ] Reproducibilidad del trabajo ✓
- [ ] Insights de ingeniería valiosos ✓

---

## 📅 Cronograma Sugerido (3 semanas)

Nota: El cronograma se comprime a 3 semanas priorizando iteraciones rápidas y muestras de desarrollo (5k-10k) para validar decisiones antes de escalar al dataset completo.

### Semana 1 — Exploración, Curación y Features Iniciales
- Días 1-2: Carga de datos, EDA inicial (estadísticas, missingness, outliers) y documentación
- Días 3-4: Limpieza y curación (imputación, tratamiento de outliers, coherencia de unidades)
- Días 5-7: Feature engineering inicial y agregación por `trip_id` (features clave para modelos rápidos)

### Semana 2 — Modelado Rápido y Tuning (desarrollo con muestra)
- Días 1-2: Baselines y modelos simples (dummy mean/median, linear, ridge/lasso). Evaluación sobre dev sample
- Días 3-4: Modelos tree-based (Random Forest, XGBoost/LightGBM) y búsqueda de hiperparámetros básica (Random Search)
- Días 5-7: Primeras NN y/o AE/VAE en muestra pequeña; comparar representaciones y rendimiento downstream

### Semana 3 — Entrenamiento Final, Evaluación y Documentación
- Días 1-2: Escalado a dataset grande (train+val) para el mejor modelo; entrenamiento final con hiperparámetros óptimos
- Días 3: Evaluación final en test reservado, métricas y análisis por subgrupos (tipo vehículo, tipo ruta)
- Días 4-5: Interpretabilidad (SHAP, PDP) y análisis de errores; generar visualizaciones clave
- Días 6-7: Redacción del informe ejecutivo, preparar slides y empaquetar artefactos reproducibles

---

## Métricas de Éxito del Proyecto

### Métricas Cuantitativas
- [ ] **Superar baseline dummy** (predicción por media/grupo) significativamente
- [ ] R² > 0.80 en test set (excelente)
- [ ] RMSE < 15% del promedio de consumo
- [ ] MAPE < 10%
- [ ] **Mejora de al menos 40-50% en RMSE vs baseline dummy**
- [ ] Mejora de al menos 20-30% vs baseline de regresión lineal

### Métricas Cualitativas
- [ ] Insights accionables para mejorar eficiencia energética
- [ ] Interpretación física coherente de features importantes
- [ ] Visualizaciones que comunican claramente los hallazgos
- [ ] Informe bien estructurado y profesional

---

## Tips para un Informe Excelente

1. **Narrativa coherente**: El informe debe contar una historia, no ser solo una colección de gráficos
2. **Justificar decisiones**: Explicar el "por qué" de cada decisión metodológica
3. **Balance técnico**: Suficiente detalle técnico pero accesible
4. **Visualizaciones**: Menos es más - cada gráfico debe aportar información valiosa
5. **Interpretación**: No solo reportar números, interpretarlos en contexto
6. **Limitaciones**: Discutir honestamente limitaciones y sesgos
7. **Reproducibilidad**: Código limpio, documentado y reproducible
8. **Originalidad**: Más allá de lo pedido, aportar análisis creativos

---

## Recursos Recomendados

### Librerías Python
- **Data**: pandas, numpy
- **Visualización**: matplotlib, seaborn, plotly
- **ML Clásico**: scikit-learn, xgboost, lightgbm, catboost
- **DL**: tensorflow/keras o pytorch
- **Interpretabilidad**: shap, lime, eli5
- **Tuning**: optuna, scikit-optimize
- **Utils**: joblib, pickle, tqdm

### Papers y Referencias
- Vehicle Energy Dataset original paper (DOE)
- Papers sobre predicción de consumo energético en vehículos
- Documentación de SHAP y técnicas de interpretabilidad
- Tutoriales de autoencoders y VAEs