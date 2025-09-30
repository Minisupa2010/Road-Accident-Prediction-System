
import argparse, os
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from utils import basic_preprocess, prepare_features

def main(data_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print("Loading data from", data_path)
    df = pd.read_csv(data_path)
    print("Raw rows:", len(df))
    df = basic_preprocess(df)
    print("After preprocess rows:", len(df))
    # For regression target use 'collision_adjusted_severity_serious' if present, else severity normalized
    if 'collision_adjusted_severity_serious' in df.columns:
        y_reg = pd.to_numeric(df['collision_adjusted_severity_serious'], errors='coerce').fillna(0.0)
    else:
        # Normalize severity to [0,1]
        y_reg = pd.to_numeric(df['severity'], errors='coerce').fillna(0)
        ymax = max(1, y_reg.max())
        y_reg = y_reg / ymax

    # For classifier target, create binary label: serious (severity==1 or severity<=2 depending on coding)
    y_clf = (pd.to_numeric(df['severity'], errors='coerce') <= 2).astype(int).fillna(0)

    # Prepare simple features
    X = df[['hour','dayofweek','latitude','longitude','is_weekend']].fillna(0)

    print("Training RandomForestRegressor...")
    reg = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    reg.fit(X, y_reg)
    dump(reg, os.path.join(out_dir, 'rf_reg.joblib'))

    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X, y_clf)
    dump(clf, os.path.join(out_dir, 'rf_clf.joblib'))

    print("Models saved to", out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='models')
    args = parser.parse_args()
    main(args.data, args.out)
