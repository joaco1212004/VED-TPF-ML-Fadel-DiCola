# Plan de informe

## Objetivo

Predecir el consumo de un veículo dado un X

## Creación del Dataset

### Problema 1

Crear un Dataset X y separarlo del Y para más tarde probar modelos supervisados.

### Rama 1: Fourier

Cada csv dinámico lo convertimos en un feature para luego agregarlo a el X e Y.

### Rama 2: Métricas

Calculamos métricas básicas como máximo, mínimo, promedio, desviación estándar, etc; con el fin de agregarlas al X e Y.

### Probaoms que dataset es más útil para predecir el consumo.

Testeamos con modelos supervisados (Regresión Lineal, Random Forest, XGBoost, y otros) y vemos cual dataset nos da mejores resultados.

- **De esto sacamos un set de gráficos para el informe**
- Con estos gráficos damos el análisis de los mismos para decirdir el uso de un dataset y el costo-beneficio (computacional) de ambos.

