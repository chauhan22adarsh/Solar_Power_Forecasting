# Solar Power Generation Forecasting System
## Project Documentation & Release Notes

**Project:** Solar Power Output Forecasting Using Machine Learning Techniques  
**Author:** Adarsh Chauhan | Roll No: 230107008  
**Course:** CL653 — Applications of AI and ML for Chemical Engineering  
**Institution:** IIT Guwahati  
**Date:** April 2025  

---

## Overview

This project implements a production-ready solar power generation forecasting system for Plant 1 and Plant 2 of an Indian solar installation. The system forecasts AC power output using a multi-model approach: classical machine learning for 15-minute inverter-level predictions, and SARIMA for hourly plant-level seasonal time-series forecasting.

A Streamlit web application provides an accessible interface for plant engineers without requiring ML expertise or SCADA integration.

---

## What the System Does

### Data Pipeline
- Loads Plant Generation CSV and Weather Sensor CSV for any plant
- Merges on `DATE_TIME`; handles both Plant 1 and Plant 2 column schemas automatically
- Cleans invalid sensor readings (negative power values, nulls)
- Engineers 14 predictive features from raw sensor data

### Feature Engineering

Features are engineered to be strictly prediction-honest — no feature contains information about the value being predicted:

| Feature | Type | Description |
|---------|------|-------------|
| `TOTAL_MINUTES_PASSED` | Temporal | Continuous intra-day position (HOUR×60+MIN) |
| `HOUR`, `MONTH` | Temporal | Time of day and seasonal markers |
| `DELTA_TEMPERATURE` | Physical | Module−Ambient temperature difference |
| `IRRAD_X_MODULE` | Physical | Irradiation × Module temperature interaction |
| `AC_POWER_ROLL3_MEAN` | Autoregressive | Rolling mean of past 3 readings (shifted; t-1 to t-3) |
| `AC_POWER_LAG1`, `LAG3` | Autoregressive | Previous readings at t-1 and t-3 |
| `SOURCE_KEY_NUM` | Categorical | Label-encoded inverter identifier |

The following columns are intentionally excluded from ML training features:
- `DC_POWER` — near-perfect correlation with AC_POWER (Spearman ρ = 0.99+); including it makes the problem trivially solvable rather than genuinely learned
- `EFFICIENCY` — defined as AC_POWER / (IRRADIATION × 1000); contains the prediction target in its numerator
- `DC_AC_RATIO` — defined as DC_POWER / (AC_POWER + ε); contains the prediction target in its denominator
- `DAILY_YIELD` — cumulative within-day energy counter; its value at time t contains energy information from timestamps later in the same day

### Machine Learning Models

Three models trained on 15-minute per-inverter data (80/20 temporal split, shuffle=False):

| Model | Details |
|-------|---------|
| Linear Regression | StandardScaler preprocessing; interpretable baseline |
| Decision Tree | max_depth=15, min_samples_split=10 |
| Random Forest | GridSearchCV with TimeSeriesSplit(n_splits=5); recommended model |

Typical R² on held-out test data: Linear Regression ~72–85%, Decision Tree ~85–92%, Random Forest ~89–96%.

### SARIMA Time-Series Model

`SARIMA(1,1,1)(1,0,0,24)` fitted on hourly plant-level AC_POWER totals (~816 data points):

- Seasonal period s=24 captures the daily bell-curve generation pattern
- `D=0` (seasonal AR only; no seasonal differencing) appropriate for solar series with recurring nighttime zeros
- 95% confidence intervals for forecast uncertainty quantification
- Full four-panel residual diagnostics valid at ~650 training points

A daily ARIMA(1,1,1) on 34 daily totals is also provided as a weekly-scale companion view.

### Visualisations

Every model includes a **Forecast vs Actual time-series plot** over the test period, in addition to scatter plots, residual plots, and feature importance charts.

### Fault Detection

Coefficient of Variation (CoV) analysis classifies each of the 34 days as:
- ✅ Very Stable (CoV < 0.20)
- 🟢 Low Fluctuation (0.20–0.40)
- 🟡 High Fluctuation (0.40–0.70)
- 🔴 Very High Fluctuation (> 0.70)

---

## Deliverables

| File | Description |
|------|-------------|
| `Solar_Power_Forecasting_FINAL.ipynb` | 53-cell Jupyter notebook; supports Plant 1 & 2 via PLANT_ID toggle |
| `app.py` | Streamlit web application; 4-tab dashboard |
| `requirements.txt` | 7 Python package dependencies |
| `README_DEPLOY.md` | 2-command deployment guide |
| `RELEASE_NOTES.md` | This document |

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at: `http://localhost:8501`

**Free cloud deployment (2 steps):**
1. Push project folder to a GitHub repository
2. Go to `https://share.streamlit.io` → connect repository → Deploy (~2 minutes, no cost)

---

## Web Application — Tab Overview

| Tab | Content |
|-----|---------|
| 📊 EDA & Overview | Spearman correlation heatmap, feature distributions (daytime only), hourly AC profile, daily totals with best/worst day highlighted, CoV fault severity table |
| 🌳 ML Models | LR / DT / RF training; **Forecast vs Actual time-series overlay for all three models**; Gini feature importance; scatter + residual plots; cross-check prediction table |
| 📈 SARIMA | Hourly/Daily toggle; **Forecast vs Actual with 95% CI band**; full four-panel diagnostics (hourly) or simplified residual analysis (daily); SARIMA model summary |
| 🏆 Comparison | ML-only bar charts (same scale); complete metrics table; scale note for SARIMA; JSON report download |

---

## Dataset

**Source:** Kaggle — Solar Power Generation Data  
**URL:** https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

Required files (place in a `data/` folder):
- `Plant_1_Generation_Data.csv` + `Plant_1_Weather_Sensor_Data.csv`
- `Plant_2_Generation_Data.csv` + `Plant_2_Weather_Sensor_Data.csv`

34 days · 15-minute intervals · 22 inverters per plant

---

## Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
joblib>=1.3.0
jinja2>=3.1.6
tensorflow>=2.20.0
scipy>=1.10.0
```

**Minimum Python:** 3.10 | **GPU required:** No | **SCADA required:** No

TensorFlow (for the LSTM section in the notebook) is an optional dependency not required for the Streamlit application.

---

## Design Decisions

**Why SARIMA and not plain ARIMA:**  
Solar generation has a strong daily seasonal pattern — the bell-curve peaking at solar noon. SARIMA captures this periodic structure explicitly via the seasonal AR component `(P=1, s=24)`, rather than treating it as random variation.

**Why hourly aggregation for SARIMA:**  
Hourly plant-level aggregation (~816 points) provides sufficient data for statistically reliable diagnostics and avoids the computational overhead of fitting on 68,000 high-frequency rows. The seasonal period s=24 maps directly to one full day in hourly data.

**Why Forecast vs Actual plots for all models:**  
A scatter plot shows overall accuracy but hides temporal structure. The time-series overlay reveals whether the model correctly tracks ramp-up at dawn, plateau at midday, and ramp-down at dusk — the operationally important part of the generation curve.

---

*Solar Power Generation Forecasting System | CL653 Final Project | IIT Guwahati | 2025*
