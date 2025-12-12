"""
VED - Modelo Final de Predicción de Consumo Energético

Modelo ejecutable que realiza predicciones sobre un conjunto de datos
con el mismo formato usado para el desarrollo del modelo.

Uso:
    python main.py                                # Demo con sample aleatorio
    python main.py _                              # Predice todo el dataset y guarda en predicciones.csv
    python main.py _ -o resultado.csv             # Predice todo el dataset y guarda en archivo
    python main.py datos.csv                      # Predice CSV externo y guarda en predicciones.csv
    python main.py datos.csv -o resultados.csv   # Especifica archivo de salida
    python main.py datos.csv --show              # Muestra predicciones en consola
    python main.py datos.csv -m modelo.pkl        # Usa modelo personalizado

Formato de entrada (CSV):
    El archivo debe tener las mismas columnas que X_metrics.csv:
    - filename, VehId, DayNum, Trip, Vehicle Type, ...
    - Métricas estadísticas de las señales del vehículo

Formato de salida (CSV):
    - filename: identificador del viaje
    - VehId: identificador del vehículo
    - Vehicle Type: tipo de vehículo (ICE, HEV, PHEV, EV)
    - pred_combustion_L_per_100km: predicción de consumo de combustible
    - pred_electric_kWh_per_km: predicción de consumo eléctrico

Modelos:
    - Combustión (XGBoost): R²=0.80, RMSE=2.26 L/100km
    - Eléctrico (XGBoost): R²=0.63, RMSE=0.069 kWh/km
"""

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = SCRIPT_DIR / 'models' / 'saved_models' / 'final'


def load_models(custom_model_path=None):
    """Carga los modelos entrenados y scalers."""
    
    # Si se especifica modelo personalizado
    if custom_model_path:
        model_path = Path(custom_model_path)
        if not model_path.exists():
            print(f"Error: No se encontró el modelo: {custom_model_path}")
            sys.exit(1)
        
        # Cargar modelo personalizado
        custom_model = joblib.load(model_path)
        
        # Buscar scaler en el mismo directorio
        model_dir = model_path.parent
        model_name = model_path.stem  # ej: Random_Forest_combustion
        
        # Determinar tipo de target del nombre del modelo
        scaler_path = None
        if '_combustion' in model_name or 'combustion' in model_name.lower():
            scaler_path = model_dir / 'scaler_combustion.pkl'
        elif '_electric' in model_name or 'electric' in model_name.lower():
            scaler_path = model_dir / 'scaler_electric.pkl'
        
        # Si no se encontró, buscar scaler genérico
        if scaler_path is None or not scaler_path.exists():
            scaler_path = model_dir / 'scaler.pkl'
        
        # Para modelos fourier, buscar en saved_models
        if not scaler_path.exists() and 'fourier' in str(model_path).lower():
            if '_combustion' in model_name or 'combustion' in model_name.lower():
                scaler_path = SCRIPT_DIR / 'models' / 'saved_models' / 'fourier_scaler_combustion.pkl'
            elif '_electric' in model_name or 'electric' in model_name.lower():
                scaler_path = SCRIPT_DIR / 'models' / 'saved_models' / 'fourier_scaler_electric.pkl'
        
        custom_scaler = None
        if scaler_path and scaler_path.exists():
            custom_scaler = joblib.load(scaler_path)
        else:
            print(f"Advertencia: No se encontró scaler para el modelo")
        
        return {
            'custom': {
                'model': custom_model,
                'scaler': custom_scaler,
                'path': model_path,
            },
            'metrics': {'custom': {'r2': 'N/A', 'rmse': 'N/A'}}
        }
    
    # Cargar modelos por defecto
    if not MODEL_DIR.exists():
        print(f"Error: No se encontró el directorio de modelos: {MODEL_DIR}")
        print("Ejecutá primero el notebook 5final_model_train_test.ipynb")
        sys.exit(1)
    
    models = {}
    
    # Cargar modelo y scaler de combustión
    models['combustion'] = {
        'model': joblib.load(MODEL_DIR / 'xgboost_combustion.pkl'),
        'scaler': joblib.load(MODEL_DIR / 'scaler_combustion.pkl'),
    }
    
    # Cargar modelo y scaler eléctrico
    models['electric'] = {
        'model': joblib.load(MODEL_DIR / 'xgboost_electric.pkl'),
        'scaler': joblib.load(MODEL_DIR / 'scaler_electric.pkl'),
    }
    
    # Cargar métricas
    models['metrics'] = joblib.load(MODEL_DIR / 'final_metrics.pkl')
    
    return models


def prepare_features(df, scaler):
    """Prepara las features en el orden correcto para el modelo."""
    feature_names = list(scaler.feature_names_in_)
    
    # Crear matriz de features
    X = pd.DataFrame(index=df.index, columns=feature_names)
    
    for feat in feature_names:
        if feat in df.columns:
            X[feat] = df[feat].values
        else:
            # Buscar con nombre sanitizado
            sanitized = feat.replace('[', '').replace(']', '').replace('<', '').replace('>', '')
            if sanitized in df.columns:
                X[feat] = df[sanitized].values
            else:
                X[feat] = 0.0
    
    # Convertir a numérico
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return X


def demo_mode(models, output_file=None):
    """Modo demo: sample aleatorio o predicción de todo el dataset."""
    data_path = SCRIPT_DIR / 'Data' / 'X' / 'X_metrics.csv'
    y_path = SCRIPT_DIR / 'Data' / 'Y' / 'Y.csv'
    
    if not data_path.exists():
        print(f"Error: No se encontró el dataset: {data_path}")
        return 1
    
    # Cargar datos
    df = pd.read_csv(data_path, low_memory=False)
    
    # Cargar valores reales si existen
    y_real = None
    if y_path.exists():
        y_real = pd.read_csv(y_path)
    
    # Si hay archivo de salida, predecir todo el dataset
    if output_file:
        print(f"\nDatos: {data_path.name}")
        print(f"  {len(df)} registros")
        
        # Realizar predicciones
        results = predict(df, models)
        
        # Agregar valores reales si existen (solo modelos estándar)
        if y_real is not None and 'custom' not in models:
            # Usar índice para evitar duplicados
            y_subset = y_real[['filename', 'Y_consumption_combustion_L_per_100km', 'Y_consumption_electric_kWh_per_km']].drop_duplicates(subset='filename')
            results = results.merge(
                y_subset,
                on='filename',
                how='left'
            )
            results.rename(columns={
                'Y_consumption_combustion_L_per_100km': 'real_combustion_L_per_100km',
                'Y_consumption_electric_kWh_per_km': 'real_electric_kWh_per_km'
            }, inplace=True)
        
        # Guardar
        results.to_csv(output_file, index=False)
        print(f"\nPredicciones guardadas en: {output_file}")
        
        # Resumen (solo si no es modelo custom)
        if 'custom' in models:
            if 'prediction' in results.columns:
                mean_pred = results['prediction'].mean()
                print(f"  {len(results)} predicciones, promedio {mean_pred:.4f}")
        else:
            valid_comb = results['pred_combustion_L_per_100km'].notna()
            if valid_comb.any():
                mean_comb = results.loc[valid_comb, 'pred_combustion_L_per_100km'].mean()
                print(f"  Combustión: {valid_comb.sum()} predicciones, promedio {mean_comb:.2f} L/100km")
            
            valid_elec = results['pred_electric_kWh_per_km'].notna()
            if valid_elec.any():
                mean_elec = results.loc[valid_elec, 'pred_electric_kWh_per_km'].mean()
                print(f"  Eléctrico:  {valid_elec.sum()} predicciones, promedio {mean_elec:.4f} kWh/km")
        
        return 0
    
    # Modo sample aleatorio (sin archivo de salida)
    idx = np.random.randint(0, len(df))
    sample = df.iloc[[idx]]
    
    # Obtener info del sample
    filename = sample['filename'].values[0]
    veh_id = sample['VehId'].values[0]
    veh_type = sample['Vehicle Type'].values[0]
    
    # Realizar predicción
    results = predict(sample, models)
    pred_comb = results['pred_combustion_L_per_100km'].values[0]
    pred_elec = results['pred_electric_kWh_per_km'].values[0]
    
    # Buscar valores reales
    real_comb = None
    real_elec = None
    if y_real is not None:
        match = y_real[y_real['filename'] == filename]
        if len(match) > 0:
            real_comb = match['Y_consumption_combustion_L_per_100km'].values[0]
            real_elec = match['Y_consumption_electric_kWh_per_km'].values[0]
    
    # Mostrar resultado
    print(f"\nSample #{idx} de {len(df)}")
    print(f"Viaje: {filename}")
    print(f"VehId: {veh_id} | Tipo: {veh_type}")
    
    if not np.isnan(pred_comb):
        if real_comb is not None and not np.isnan(real_comb) and real_comb > 0:
            error = ((pred_comb - real_comb) / real_comb) * 100
            print(f"\nCombustión: {pred_comb:.2f} L/100km (real: {real_comb:.2f}, error: {error:+.1f}%)")
        else:
            print(f"\nCombustión: {pred_comb:.2f} L/100km")
    
    if not np.isnan(pred_elec):
        if real_elec is not None and not np.isnan(real_elec) and real_elec > 0:
            error = ((pred_elec - real_elec) / real_elec) * 100
            print(f"Eléctrico: {pred_elec:.4f} kWh/km (real: {real_elec:.4f}, error: {error:+.1f}%)")
        else:
            print(f"Eléctrico: {pred_elec:.4f} kWh/km")
    
    print()
    return 0


def predict(df, models):
    """Realiza predicciones para todo el DataFrame."""
    results = pd.DataFrame()
    
    # Copiar columnas identificadoras
    for col in ['filename', 'VehId', 'Vehicle Type']:
        if col in df.columns:
            results[col] = df[col]
    
    # Si es modelo personalizado
    if 'custom' in models:
        model_info = models['custom']
        model = model_info['model']
        scaler = model_info['scaler']
        
        if scaler:
            X = prepare_features(df, scaler)
            X_scaled = scaler.transform(X)
            results['prediction'] = model.predict(X_scaled)
        else:
            # Sin scaler, usar todas las columnas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            results['prediction'] = model.predict(df[numeric_cols].fillna(0))
        
        return results
    
    # Predicción de combustión
    scaler_comb = models['combustion']['scaler']
    model_comb = models['combustion']['model']
    X_comb = prepare_features(df, scaler_comb)
    X_comb_scaled = scaler_comb.transform(X_comb)
    results['pred_combustion_L_per_100km'] = model_comb.predict(X_comb_scaled)
    
    # Predicción eléctrica
    scaler_elec = models['electric']['scaler']
    model_elec = models['electric']['model']
    X_elec = prepare_features(df, scaler_elec)
    X_elec_scaled = scaler_elec.transform(X_elec)
    results['pred_electric_kWh_per_km'] = model_elec.predict(X_elec_scaled)
    
    # Aplicar lógica por tipo de vehículo
    if 'Vehicle Type' in results.columns:
        # ICE solo tiene combustión
        ice_mask = results['Vehicle Type'] == 'ICE'
        results.loc[ice_mask, 'pred_electric_kWh_per_km'] = np.nan
        
        # EV solo tiene eléctrico
        ev_mask = results['Vehicle Type'] == 'EV'
        results.loc[ev_mask, 'pred_combustion_L_per_100km'] = np.nan
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Modelo final de predicción de consumo energético vehicular',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'input_file',
        type=str,
        nargs='?',
        default=None,
        help='Archivo CSV de entrada (si no se especifica, usa modo demo)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='predicciones.csv',
        help='Archivo CSV de salida (default: predicciones.csv)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Mostrar predicciones en consola'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default=None,
        help='Path a modelo .pkl personalizado'
    )
    
    args = parser.parse_args()
    
    # Cargar modelos
    print("Cargando modelos...")
    models = load_models(args.model)
    
    if args.model:
        print(f"Modelo personalizado: {args.model}")
    else:
        metrics = models['metrics']
        print(f"Combustión R²={metrics['combustion']['r2']:.2f} | Eléctrico R²={metrics['electric']['r2']:.2f}")
    
    # Modo demo si no se especifica archivo o se usa "_"
    if args.input_file is None or args.input_file == '_':
        output = args.output if args.input_file == '_' else None
        return demo_mode(models, output)
    
    # Verificar archivo de entrada
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: No se encontró el archivo: {args.input_file}")
        sys.exit(1)
    
    # Cargar datos
    print(f"\nDatos: {args.input_file}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  {len(df)} registros")
    
    # Realizar predicciones
    results = predict(df, models)
    
    # Guardar resultados
    output_path = Path(args.output)
    results.to_csv(output_path, index=False)
    print(f"\nPredicciones guardadas en: {output_path}")
    
    # Mostrar resumen
    print(f"\nResumen:")
    
    valid_comb = results['pred_combustion_L_per_100km'].notna()
    if valid_comb.any():
        mean_comb = results.loc[valid_comb, 'pred_combustion_L_per_100km'].mean()
        print(f"  Combustión: {valid_comb.sum()} predicciones, promedio {mean_comb:.2f} L/100km")
    
    valid_elec = results['pred_electric_kWh_per_km'].notna()
    if valid_elec.any():
        mean_elec = results.loc[valid_elec, 'pred_electric_kWh_per_km'].mean()
        print(f"  Eléctrico:  {valid_elec.sum()} predicciones, promedio {mean_elec:.4f} kWh/km")
    
    # Mostrar predicciones si se solicita
    if args.show:
        print("\nPredicciones:")
        print(results.to_string(index=False))
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
