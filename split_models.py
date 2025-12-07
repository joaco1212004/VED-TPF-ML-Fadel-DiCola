import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Cargar datos
X_fourier = pd.read_csv("DATA/X/X_fourier.csv")
X_metrics = pd.read_csv("DATA/X/X_metrics.csv")
Y = pd.read_csv("DATA/Y/Y.csv")

# Split 1: Train + Temp
Xf_train, Xf_temp, Xm_train, Xm_temp, y_train, y_temp = train_test_split(
    X_fourier, X_metrics, Y, test_size=0.30, random_state=42, shuffle=True
)

# Split 2: Dev + Test (50/50 de ese 30%)
Xf_dev, Xf_test, Xm_dev, Xm_test, y_dev, y_test = train_test_split(
    Xf_temp, Xm_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
)

# Crear carpeta de splits
Path("splits").mkdir(exist_ok=True)

# Guardar
Xf_train.to_csv("splits/X_fourier_train.csv", index=False)
Xf_dev.to_csv("splits/X_fourier_dev.csv", index=False)
Xf_test.to_csv("splits/X_fourier_test.csv", index=False)

Xm_train.to_csv("splits/X_metrics_train.csv", index=False)
Xm_dev.to_csv("splits/X_metrics_dev.csv", index=False)
Xm_test.to_csv("splits/X_metrics_test.csv", index=False)

y_train.to_csv("splits/Y_train.csv", index=False)
y_dev.to_csv("splits/Y_dev.csv", index=False)
y_test.to_csv("splits/Y_test.csv", index=False)

print("Splits generados correctamente.")