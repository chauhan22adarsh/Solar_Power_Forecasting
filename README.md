# ☀️ Solar Power Forecasting — Deployment Guide

## Local Setup (2 commands)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at: http://localhost:8501

## Free Cloud Deployment (Streamlit Community Cloud)

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Click "New app" → select your repo → set `app.py` as entrypoint
4. Click Deploy — live URL in ~2 minutes ✓

## Files needed
- app.py            ← main web app
- requirements.txt  ← dependencies

## Usage
1. Upload Plant_X_Generation_Data.csv
2. Upload Plant_X_Weather_Sensor_Data.csv  
3. Click "Run Full Analysis"
4. Explore 4 tabs: EDA | ML Models | SARIMA | Comparison
