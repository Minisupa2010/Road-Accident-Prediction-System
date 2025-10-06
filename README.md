# Road Accident Prediction System ( streamlit based )

A production-ready, self-hosted system for analyzing and predicting road accidents using historical incident data, weather enrichment, and machine-learning models. This package includes a Streamlit UI, background worker (Celery), weather enrichment with caching (Open-Meteo), OSGB → WGS84 coordinate conversion for UK STATS19 datasets, SHAP explanations, and Docker Compose for local orchestration.

---

## Key Features

- **Data ingestion & preprocessing** (robust handling of common CSV schemas)
- **API-based weather enrichment** (Open-Meteo archive) with local caching to avoid rate limits
- **OSGB to WGS84 conversion** for Easting/Northing (auto-detected)
- **Model training**: RandomForest regression + classifier (saved as `models/rf_reg.joblib` and `models/rf_clf.joblib`)
- **Streamlit dashboard**:
  - Data preview & analytics (hourly trends, severity distribution)
  - Heatmap of incidents (Folium)
  - Prediction panel (single-point predict)
  - SHAP-based model explanations (after training)
- **Background tasks** using Celery (enqueue training / heavy enrichment)
- **Docker Compose** for Redis + worker + Streamlit orchestration
- **Safe defaults** and error-handling for production stability

---

## Quickstart (local machine)

**Prereqs**
- Python 3.9+  
- Git (optional)  
- Docker (optional, for Redis via Docker)  

1. **Unzip the project**
   ```bash
   unzip road_accident_prediction_system_robust.zip -d road_accident_robust
   cd road_accident_robust
