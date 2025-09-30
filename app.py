
import streamlit as st
import pandas as pd, os, subprocess, time
from utils import basic_preprocess, prepare_features
from joblib import load
from streamlit_folium import st_folium
import folium

st.set_page_config(layout='wide', page_title='Road Accident Prediction')

st.title("Road Accident Prediction")

DATA_PATH = os.path.join('data', 'accidents.csv')
MODELS_DIR = os.path.join('models')

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        st.error(f"Data file not found: {path}. Please place the CSV in the 'data' folder.")
        return None
    try:
        df = pd.read_csv(path)
        df = basic_preprocess(df)
        return df
    except Exception as e:
        st.exception(e)
        return None

df = load_data()
if df is None:
    st.stop()

st.sidebar.header("Controls")
if st.sidebar.button("Train models (may take a while)"):
    with st.spinner("Training models..."):
        # Run train_model.py as subprocess so user can see output in terminal too
        subprocess.run(["python", "train_model.py", "--data", DATA_PATH, "--out", MODELS_DIR])
        st.rerun()


st.sidebar.markdown("Models will be saved to `models/` directory.")

st.header("Data preview")
st.write(df.head(5))
st.write("Rows:", len(df))

st.header("Heatmap of accidents (first 500 points)")
m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=10)
from folium.plugins import HeatMap, MarkerCluster
heat_data = df[['latitude','longitude']].dropna().values.tolist()[:5000]
HeatMap(heat_data, radius=8).add_to(m)
st_folium(m, width=900, height=500)

st.header("Prediction panel")
st.write("Enter a location and time to get model predictions (models must be trained)")

col1, col2, col3 = st.columns(3)
with col1:
    lat = st.number_input("Latitude", value=float(df['latitude'].mean()))
    lon = st.number_input("Longitude", value=float(df['longitude'].mean()))
with col2:
    date = st.date_input("Date", value=pd.to_datetime(df['timestamp'].iloc[0]).date())
    t = st.time_input("Time", value=pd.to_datetime(df['timestamp'].iloc[0]).time())
with col3:
    run_pred = st.button("Predict")

if run_pred:
    ts = pd.to_datetime(str(date) + ' ' + str(t))
    feat = prepare_features({'timestamp': ts, 'latitude': lat, 'longitude': lon})
    st.write("Features:", feat.to_dict(orient='records')[0])

    # Check models
    reg_path = os.path.join(MODELS_DIR, 'rf_reg.joblib')
    clf_path = os.path.join(MODELS_DIR, 'rf_clf.joblib')
    if os.path.exists(reg_path) and os.path.exists(clf_path):
        reg = load(reg_path)
        clf = load(clf_path)
        pred_reg = float(reg.predict(feat)[0])
        pred_prob = float(clf.predict_proba(feat)[0][1]) if hasattr(clf, 'predict_proba') else float(clf.predict(feat)[0])
        st.success(f"Predicted adjusted-severity (regression): {pred_reg:.4f}")
        st.success(f"Predicted probability (classifier): {pred_prob:.4f}")
    else:
        st.warning("Models not found. Please train models using the sidebar button.")

st.header("Top intersections (simple aggregation)")
agg = df.groupby(['latitude','longitude']).size().reset_index(name='obs').sort_values('obs', ascending=False).head(50)
st.dataframe(agg)

st.info("This system standardizes preprocessing and model training. For production, add exposure, better features, and caching for weather APIs.")
