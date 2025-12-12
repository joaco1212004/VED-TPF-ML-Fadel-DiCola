"""
VED Energy Consumption Prediction - Demo
=========================================

Este script demuestra el funcionamiento de los modelos entrenados para predecir
el consumo energético de vehículos usando el dataset VED.

Modelos disponibles:
- Combustión (L/100km): Para vehículos ICE, HEV, PHEV
- Eléctrico (kWh/km): Para vehículos PHEV, EV

Uso:
    python main.py                    # Muestra un sample aleatorio
    python main.py --index 5          # Muestra el sample #5
    python main.py --type combustion  # Solo samples de combustión
    python main.py --type electric    # Solo samples de eléctrico
    python main.py --type phev        # Samples PHEV con ambos targets
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# Columnas usadas para calcular los targets (data leakage prevention)
TARGET_CALCULATION_COLUMNS = {
    'Fuel Rate[L/hr]',
    'MAF[g/sec]',
    'HV Battery Current[A]',
    'HV Battery Voltage[V]',
    'HV Battery SOC[%]',
    'Short Term Fuel Trim Bank 1[%]',
    'Short Term Fuel Trim Bank 2[%]',
    'Long Term Fuel Trim Bank 1[%]',
    'Long Term Fuel Trim Bank 2[%]',
}


def load_models():
    """Carga los modelos, scalers y metadatos."""
    model_dir = SCRIPT_DIR / 'models/saved_models/final'
    
    if not model_dir.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio de modelos: {model_dir}\n"
            "Ejecuta primero el notebook 5final_model_train_test.ipynb"
        )
    
    # Cargar scaler 
    if (model_dir / 'scaler_combustion.pkl').exists():
        scaler_combustion = joblib.load(model_dir / 'scaler_combustion.pkl')
        scaler_electric = joblib.load(model_dir / 'scaler_electric.pkl')
    else:
        # Fallback: usar scaler único para ambos
        scaler_final = joblib.load(model_dir / 'scaler_final.pkl')
        scaler_combustion = scaler_final
        scaler_electric = scaler_final
    
    models = {
        'combustion': {
            'model': joblib.load(model_dir / 'xgboost_combustion.pkl'),
            'scaler': scaler_combustion,
        },
        'electric': {
            'model': joblib.load(model_dir / 'xgboost_electric.pkl'),
            'scaler': scaler_electric,
        },
        'feature_names': joblib.load(model_dir / 'feature_names.pkl'),
        'metrics': joblib.load(model_dir / 'final_metrics.pkl'),
    }
    
    # Cargar feature_cols originales para mapeo
    demo_dir = model_dir / 'demo_samples'
    if (demo_dir / 'original_feature_cols.pkl').exists():
        models['original_feature_cols'] = joblib.load(demo_dir / 'original_feature_cols.pkl')
    
    return models


def load_demo_samples(sample_type='random'):
    """Carga samples de demostración del test set."""
    demo_dir = SCRIPT_DIR / 'models/saved_models/final/demo_samples'
    
    if not demo_dir.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio de demos: {demo_dir}\n"
            "Ejecuta primero el notebook 5final_model_train_test.ipynb"
        )
    
    samples = {}
    
    if sample_type in ['combustion', 'random', 'all']:
        comb_file = demo_dir / 'demo_combustion.csv'
        if comb_file.exists():
            samples['combustion'] = pd.read_csv(comb_file)
    
    if sample_type in ['electric', 'random', 'all']:
        elec_file = demo_dir / 'demo_electric.csv'
        if elec_file.exists():
            samples['electric'] = pd.read_csv(elec_file)
    
    if sample_type in ['phev', 'random', 'all']:
        phev_file = demo_dir / 'demo_phev_both.csv'
        if phev_file.exists():
            samples['phev'] = pd.read_csv(phev_file)
    
    return samples


def predict_sample(sample_row, models, model_type):
    """Realiza predicción para un sample.
    
    El scaler usa feature names originales (con corchetes).
    Los CSVs de demo tienen columnas con nombres originales (con corchetes).
    Necesitamos extraer las features en el mismo orden que el scaler.
    """
    scaler = models[model_type]['scaler']
    model = models[model_type]['model']
    
    scaler_feature_names = list(scaler.feature_names_in_)
    
    # Extraer features en el orden correcto
    features = []
    for feat_name in scaler_feature_names:
        value = None
        
        if feat_name in sample_row:
            value = sample_row[feat_name]
        else:
            sanitized_name = feat_name.replace('[', '').replace(']', '').replace('<', '').replace('>', '')
            if sanitized_name in sample_row:
                value = sample_row[sanitized_name]
        
        # Convertir a float
        if value is None:
            features.append(0.0)
        else:
            try:
                features.append(float(value))
            except (ValueError, TypeError):
                features.append(0.0)
    
    # Crear DataFrame con los feature names del scaler
    X = pd.DataFrame([features], columns=scaler_feature_names)
    
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    
    return prediction
    return prediction


def display_prediction(sample_row, models, sample_type):
    """Muestra la predicción para un sample con formato bonito."""
    
    # Info del vehículo
    veh_type = sample_row.get('Vehicle Type', 'Unknown')
    veh_id = sample_row.get('VehId', 'N/A')
    filename = sample_row.get('filename', 'N/A')
    
    print("\n" + "="*70)
    print("PREDICCIÓN DE CONSUMO ENERGÉTICO")
    print("="*70)
    
    print(f"\nINFORMACIÓN DEL VIAJE:")
    print(f"Archivo: {filename}")
    print(f"VehId: {veh_id}")
    print(f"Tipo de Vehículo: {veh_type}")
    
    # Verificar si tiene columnas de cálculo de target
    has_target_cols = any(col in sample_row.index for col in TARGET_CALCULATION_COLUMNS)
    
    if has_target_cols:
        print(f"\nEste sample contiene las columnas usadas para calcular")
        print(f"los targets (Fuel Rate, MAF, HV Battery, etc.)")
        print(f"Los valores reales están disponibles para comparación.")
    
    print("\n" + "-"*70)
    
    # Predicción de combustión
    if veh_type in ['ICE', 'HEV', 'PHEV'] or sample_type == 'combustion':
        print("\nCONSUMO DE COMBUSTIÓN (L/100km):")
        
        pred_comb = predict_sample(sample_row, models, 'combustion')
        print(f"Predicción: {pred_comb:.2f} L/100km")
        
        # Valor real si existe
        real_comb = sample_row.get('Y_consumption_combustion_L_per_100km', None)
        if real_comb is not None and not pd.isna(real_comb) and real_comb > 0:
            error = pred_comb - real_comb
            pct_error = (error / real_comb) * 100
            print(f"Valor Real: {real_comb:.2f} L/100km")
            print(f"Diferencia: {error:+.2f} L/100km ({pct_error:+.1f}%)")
            
            if abs(pct_error) < 10:
                print(f"Excelente predicción (error < 10%)")
            elif abs(pct_error) < 25:
                print(f"Buena predicción (error < 25%)")
            else:
                print(f"Predicción con margen de error significativo")
    
    # Predicción de eléctrico
    if veh_type in ['PHEV', 'EV', 'HEV'] or sample_type == 'electric':
        print("\nCONSUMO ELÉCTRICO (kWh/km):")
        
        pred_elec = predict_sample(sample_row, models, 'electric')
        print(f"Predicción: {pred_elec:.4f} kWh/km ({pred_elec*100:.2f} kWh/100km)")
        
        # Valor real si existe
        real_elec = sample_row.get('Y_consumption_electric_kWh_per_km', None)
        if real_elec is not None and not pd.isna(real_elec) and real_elec > 0:
            error = pred_elec - real_elec
            pct_error = (error / real_elec) * 100
            print(f"Valor Real: {real_elec:.4f} kWh/km ({real_elec*100:.2f} kWh/100km)")
            print(f"Diferencia: {error:+.4f} kWh/km ({pct_error:+.1f}%)")
            
            if abs(pct_error) < 10:
                print(f"Excelente predicción (error < 10%)")
            elif abs(pct_error) < 25:
                print(f"Buena predicción (error < 25%)")
            else:
                print(f"Predicción con margen de error significativo")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Demo de predicción de consumo energético vehicular',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--index', '-i', type=int, default=None,
        help='Índice del sample a mostrar (default: aleatorio)'
    )
    parser.add_argument(
        '--type', '-t', type=str, default='phev',
        choices=['combustion', 'electric', 'phev', 'random'],
        help='Tipo de sample a mostrar (default: phev para ver ambos targets)'
    )
    parser.add_argument(
        '--list', '-l', action='store_true',
        help='Lista los samples disponibles sin hacer predicción'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("VED - Vehicle Energy Dataset - Predicción de Consumo")
    print("="*70)
    
    print("\nCargando modelos...")
    try:
        models = load_models()
        print("Modelos cargados correctamente")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    
    metrics = models['metrics']
    print(f"\nModelo Combustión: R²={metrics['combustion']['r2']:.4f}, RMSE={metrics['combustion']['rmse']:.4f}")
    print(f"Modelo Eléctrico: R²={metrics['electric']['r2']:.4f}, RMSE={metrics['electric']['rmse']:.4f}")
    
    print("\nCargando samples de demostración...")
    try:
        samples = load_demo_samples(args.type)
        if not samples:
            print("No se encontraron samples de demostración")
            return 1
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    
    # Determinar qué dataset usar
    if args.type == 'phev' and 'phev' in samples:
        df = samples['phev']
        sample_type = 'phev'
    elif args.type == 'combustion' and 'combustion' in samples:
        df = samples['combustion']
        sample_type = 'combustion'
    elif args.type == 'electric' and 'electric' in samples:
        df = samples['electric']
        sample_type = 'electric'
    elif args.type == 'random':
        # Elegir aleatoriamente un dataset
        available = list(samples.keys())
        sample_type = np.random.choice(available)
        df = samples[sample_type]
    else:
        # Fallback a cualquier disponible
        sample_type = list(samples.keys())[0]
        df = samples[sample_type]
    
    print(f"Usando dataset: {sample_type} ({len(df)} samples disponibles)")
    
    # Listar samples
    if args.list:
        print(f"\nSamples disponibles en '{sample_type}':")
        for i, row in df.iterrows():
            veh_type = row.get('Vehicle Type', 'N/A')
            filename = row.get('filename', 'N/A')
            print(f"[{i:3d}] {veh_type:5s} - {filename}")
        return 0
    
    # Seleccionar sample
    if args.index is not None:
        if args.index >= len(df):
            print(f"Índice {args.index} fuera de rango (max: {len(df)-1})")
            return 1
        idx = args.index
    else:
        idx = np.random.randint(0, len(df))
    
    sample_row = df.iloc[idx]
    
    print(f"\nMostrando sample #{idx}:")
    
    display_prediction(sample_row, models, sample_type)
    
    print("\nUso: python main.py --help para ver más opciones\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
