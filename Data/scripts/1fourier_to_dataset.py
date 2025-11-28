import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')


def detect_timestamp_col(cols):
    for c in cols:
        if 'timestamp' in c.lower() or 'time' == c.lower() or 'date' in c.lower():
            return c
    return None


def infer_numeric_columns(file_list, n_sample=5):
    cols = set()
    for f in file_list[:n_sample]:
        try:
            df = pd.read_csv(f, nrows=5)
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    cols.add(c)
        except Exception:
            continue
    return sorted(cols)


def compute_welch_features(arr, fs=1.0):
    # arr is 1D numeric numpy array
    if arr.size < 4 or np.all(np.isnan(arr)):
        return None
    # remove NaNs by interpolation
    s = pd.Series(arr).interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill').values
    if s.size < 4:
        return None
    nperseg = min(256, s.size)
    try:
        f, Pxx = welch(s, fs=fs, nperseg=nperseg)
    except Exception:
        return None
    Psum = Pxx.sum()
    if Psum <= 0:
        return {
            'spec_centroid': np.nan,
            'spec_entropy': np.nan,
            'band_low': 0.0,
            'band_med': 0.0,
            'band_high': 0.0,
            'dominant_freq': np.nan,
            'total_power': 0.0,
        }
    centroid = (f * Pxx).sum() / Psum
    pnorm = Pxx / Psum
    entropy = -np.sum(np.where(pnorm>0, pnorm * np.log(pnorm), 0.0))
    # bands (absolute Hz): low <0.05, med 0.05-0.2, high >0.2
    band_low = float(Pxx[(f>=0) & (f<0.05)].sum())
    band_med = float(Pxx[(f>=0.05) & (f<0.2)].sum())
    band_high = float(Pxx[f>=0.2].sum())
    dom = float(f[np.nanargmax(Pxx)])
    return {
        'spec_centroid': float(centroid),
        'spec_entropy': float(entropy),
        'band_low': band_low,
        'band_med': band_med,
        'band_high': band_high,
        'dominant_freq': dom,
        'total_power': float(Psum)
    }


def process_file(path, numeric_cols):
    # returns dict of features for this file
    row = {'file': path.name}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print('Failed read', path, e)
        for c in numeric_cols:
            for k in ('spec_centroid','spec_entropy','band_low','band_med','band_high','dominant_freq','total_power'):
                row[f'{c}_{k}'] = np.nan
        return row

    # detect timestamp and infer sampling frequency
    ts_col = detect_timestamp_col(df.columns)
    fs = 1.0
    if ts_col is not None:
        try:
            ts = pd.to_numeric(df[ts_col], errors='coerce')
            if ts.dropna().empty:
                raise ValueError
            median = ts.dropna().diff().median()
            if median is None or np.isnan(median) or median == 0:
                fs = 1.0
            else:
                # determine units: if values large (>1e9) probably epoch ms; but differences are in ms
                # If typical timestamps small integers (like 0,200,1100) it's ms
                if median > 1e3:
                    # diff in ms
                    dt_sec = median/1000.0
                else:
                    dt_sec = median
                if dt_sec > 0:
                    fs = 1.0 / dt_sec
        except Exception:
            fs = 1.0

    for c in numeric_cols:
        if c in df.columns:
            # if column exists but is fully NaN, omit Fourier computation and leave NaNs
            col_series = pd.to_numeric(df[c], errors='coerce')
            if col_series.dropna().empty:
                # explicitly record that this column had no valid data in this file
                row[f'{c}_was_all_nan'] = True
                for k in ('spec_centroid','spec_entropy','band_low','band_med','band_high','dominant_freq','total_power'):
                    # ensure the output features are explicit NaN
                    row[f'{c}_{k}'] = np.nan
                # skip further processing for this column
                continue
            # otherwise compute features
            row[f'{c}_was_all_nan'] = False
            arr = col_series.values
            feats = compute_welch_features(arr, fs=fs)
            if feats is None:
                for k in ('spec_centroid','spec_entropy','band_low','band_med','band_high','dominant_freq','total_power'):
                    row[f'{c}_{k}'] = np.nan
            else:
                for k,v in feats.items():
                    row[f'{c}_{k}'] = v
        else:
            # column missing entirely: mark features NaN and flag as True (no data)
            for k in ('spec_centroid','spec_entropy','band_low','band_med','band_high','dominant_freq','total_power'):
                row[f'{c}_{k}'] = np.nan
            row[f'{c}_was_all_nan'] = True
    return row


def main(input_path, output_csv):
    p = Path(input_path)
    files = sorted([x for x in p.rglob('*.csv')])
    if not files:
        print('No csv files found in', p)
        return

    # build union of numeric columns by sampling header/first rows across files
    numeric_cols = infer_numeric_columns(files, n_sample=min(10,len(files)))
    # remove obvious id/time/lat/lon columns
    blacklist = [c for c in numeric_cols if any(s in c.lower() for s in ('daynum','vehid','trip','timestamp','latitude','longitude'))]
    numeric_cols = [c for c in numeric_cols if c not in blacklist]

    print('Files found:', len(files))
    print('Numeric columns candidates:', numeric_cols)

    rows = []
    for f in files:
        r = process_file(f, numeric_cols)
        rows.append(r)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print('Wrote', output_csv, 'rows:', len(df_out), 'cols:', len(df_out.columns))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='Data/VED_DynamicData', help='Input folder to scan for CSV files')
    parser.add_argument('--output', '-o', default='results/fourier_dataset.csv', help='Output CSV path')
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    main(args.input, args.output)
