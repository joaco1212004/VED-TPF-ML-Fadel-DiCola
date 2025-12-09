import pandas as pd
from pathlib import Path

# Verificar un HEV específico (VehId 115)
hev_folder = Path('Data/VED_DynamicData/VehId_115')
files = list(hev_folder.glob('Trip_*.csv'))[:3]

print('=== Verificando VehId 115 (HEV) ===')
for f in files:
    df = pd.read_csv(f)
    print(f'\n{f.name}:')
    hv_volt = df['HV Battery Voltage[V]']
    hv_curr = df['HV Battery Current[A]']
    maf = df['MAF[g/sec]']
    print(f'  HV Battery Voltage: non-null={hv_volt.notna().sum()}, mean={hv_volt.mean():.2f}')
    print(f'  HV Battery Current: non-null={hv_curr.notna().sum()}, mean={hv_curr.mean():.2f}')
    print(f'  MAF: non-null={maf.notna().sum()}, mean={maf.mean():.2f}')

# Verificar un ICE sin combustión (VehId 110)
print('\n=== Verificando VehId 110 (ICE sin combustión) ===')
ice_folder = Path('Data/VED_DynamicData/VehId_110')
files = list(ice_folder.glob('Trip_*.csv'))[:3]
for f in files:
    df = pd.read_csv(f)
    print(f'\n{f.name}:')
    fuel = df['Fuel Rate[L/hr]']
    maf = df['MAF[g/sec]']
    print(f'  Fuel Rate: non-null={fuel.notna().sum()}')
    print(f'  MAF: non-null={maf.notna().sum()}, >0: {(maf.fillna(0) > 0).sum()}')
