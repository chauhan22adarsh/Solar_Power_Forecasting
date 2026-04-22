# ☀️ Solar Power Forecasting — Deployment Guide

## Local Setup (2 commands)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at: http://localhost:8501

## Free Cloud Deployment (Streamlit Community Cloud)

Deployed Site Link:-
https://solarpowerforecasting-7fmabm2xo5dkudkycnpdwu.streamlit.app/

## Files needed
- app.py            ← main web app
- requirements.txt  ← dependencies
- data/Plant_<X>_Generation_Data.csv
- data/Plant_<X>_Weather_Sensor_Data.csv

## Usage
1. Upload Plant_<X>_Generation_Data.csv
2. Upload Plant_<X>_Weather_Sensor_Data.csv  
3. Click "Run Full Analysis"
4. Explore 4 tabs: EDA | ML Models | SARIMA | Comparison


X = 1, 2
