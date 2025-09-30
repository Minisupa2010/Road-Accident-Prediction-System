
import pandas as pd
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

def basic_preprocess(df):
    """
    Basic preprocessing for Stats19-like CSVs. Accepts a pandas DataFrame.
    Produces standardized columns: timestamp (datetime), latitude, longitude, severity (int)
    and some time features.
    """
    df = df.copy()
    # Normalize column names to lower-case for robustness
    df.columns = [c.lower() for c in df.columns]

    # Common column names in the provided CSV: 'date','time','latitude','longitude','collision_severity'
    if 'date' in df.columns and 'time' in df.columns:
        # Some rows may have missing time; fill with 00:00
        df['time'] = df['time'].fillna('00:00')
        # Combine date and time to datetime
        df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce', dayfirst=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        # If no date/time, try to parse a single datetime column
        df['timestamp'] = pd.to_datetime(df.get('datetime', None), errors='coerce')

    # Latitude/longitude
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    else:
        # Try alternative names
        for lat_name in ['lat','y']:
            for lon_name in ['lon','x']:
                if lat_name in df.columns and lon_name in df.columns:
                    df['latitude'] = pd.to_numeric(df[lat_name], errors='coerce')
                    df['longitude'] = pd.to_numeric(df[lon_name], errors='coerce')

    # Severity - many datasets use numeric codes; try to map or cast
    sev_candidates = ['collision_severity','accident_severity','severity','severity_code']
    severity_col = None
    for c in sev_candidates:
        if c in df.columns:
            severity_col = c
            break
    if severity_col is not None:
        df['severity'] = pd.to_numeric(df[severity_col], errors='coerce')
    else:
        # Fallback: if there are adjusted severity columns, use them
        if 'collision_adjusted_severity_serious' in df.columns:
            df['severity'] = (pd.to_numeric(df['collision_adjusted_severity_serious'], errors='coerce') > 0.5).astype(int)
        else:
            df['severity'] = np.nan

    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)

    # Drop rows without geometry
    df = df.dropna(subset=['latitude','longitude'])

    return df

def prepare_features(df_row):
    """
    Prepare a single-row feature vector (pandas Series or dict expected) for prediction.
    Returns a pandas DataFrame with one row suitable for model.predict.
    """
    import pandas as pd, numpy as np
    if isinstance(df_row, dict):
        s = pd.Series(df_row)
    elif hasattr(df_row, 'to_dict'):
        s = pd.Series(df_row.to_dict())
    else:
        s = pd.Series(df_row)

    # Ensure timestamp is datetime
    ts = pd.to_datetime(s.get('timestamp', pd.NaT), errors='coerce')
    hour = ts.hour if not pd.isna(ts) else int(s.get('hour', 0))
    day = ts.dayofweek if not pd.isna(ts) else int(s.get('dayofweek', 0))

    latitude = float(s.get('latitude', 0.0))
    longitude = float(s.get('longitude', 0.0))

    # Basic numeric features - keep simple for portability
    features = {
        'hour': hour,
        'dayofweek': day,
        'latitude': latitude,
        'longitude': longitude,
        'is_weekend': int(day in [5,6])
    }
    return pd.DataFrame([features])

def enrich_weather_open_meteo(lat, lon, ts):
    """
    Minimal placeholder for weather enrichment. If OPEN_METEO_BASE is set and network is allowed,
    you can implement fetching here. For now return empty dict or None safely.
    """
    return {}
