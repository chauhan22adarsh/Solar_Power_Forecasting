"""
Solar Power Generation Forecasting System
Streamlit Web Application
Author: Adarsh Chauhan | Roll: 230107008 | CL653

Model strategy (consistent with notebook and report):
  - ML models (LR, DT, RF): 15-min per-inverter data
  - SARIMA: user chooses hourly (s=24, ~816 pts, full diagnostics)
            or daily (s=7, 34 pts, residual plot only)
  - Hourly SARIMA is the primary recommended approach
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # ✅ REQUIRED for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, mean_absolute_percentage_error)
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Power Forecasting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main { background:#0d1117; color:#c9d1d9; }
  section[data-testid="stSidebar"] { background:#161b22; }
  .block-container { padding-top:1.5rem; }
  h1,h2,h3 { color:#58a6ff; }
  .stMetric label { color:#8b949e !important; font-size:0.8em !important; }
  .stMetric [data-testid="metric-container"] {
    background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px; }
  .badge { display:inline-block; padding:3px 10px; border-radius:20px;
           font-size:0.78em; font-weight:600; margin:2px; }
  .badge-blue  { background:#1f6feb; color:#fff; }
  .badge-green { background:#238636; color:#fff; }
  .badge-gold  { background:#9e6a03; color:#fff; }
  .callout { background:#161b22; border-left:4px solid #58a6ff;
             border-radius:6px; padding:10px 16px; margin:8px 0; font-size:0.9em; }
  .callout-warn { border-left-color:#f78166 !important; }
  .callout-ok   { border-left-color:#3fb950 !important; }
</style>
""", unsafe_allow_html=True)

# ── Plot theme ────────────────────────────────────────────────────────────────
DARK_BG  = '#0d1117'
PANEL_BG = '#161b22'
C1, C2, C3, C4 = '#58a6ff', '#f78166', '#3fb950', '#d2a8ff'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,   'axes.facecolor':  PANEL_BG,
    'axes.edgecolor':   '#30363d', 'axes.labelcolor': '#c9d1d9',
    'xtick.color':      '#8b949e', 'ytick.color':     '#8b949e',
    'text.color':       '#c9d1d9', 'grid.color':      '#21262d',
    'grid.linewidth':   0.7,       'font.family':     'monospace',
    'axes.titlesize':   11,        'axes.labelsize':  10,
    'legend.facecolor': PANEL_BG,  'legend.edgecolor':'#30363d',
})

SEED = 42
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true > 0   # exclude zero-power nighttime rows from MAPE
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100 if mask.sum() > 0 else np.nan
    return {
        'MAE' : round(float(mean_absolute_error(y_true, y_pred)), 3),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        'R²'  : round(float(r2_score(y_true, y_pred) * 100), 2),
        'MAPE': round(float(mape), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_process(gen_bytes, weather_bytes):
    import io
    gen     = pd.read_csv(io.BytesIO(gen_bytes))
    weather = pd.read_csv(io.BytesIO(weather_bytes))

    gen['DATE_TIME']     = pd.to_datetime(gen['DATE_TIME'],     infer_datetime_format=True, dayfirst=True)
    weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'], infer_datetime_format=True)
    gen     = gen.drop(columns=[c for c in ['PLANT_ID'] if c in gen.columns])
    weather = weather.drop(columns=[c for c in ['PLANT_ID', 'SOURCE_KEY'] if c in weather.columns])
    weather = weather.drop_duplicates(subset=['DATE_TIME'])

    df = pd.merge(gen, weather, on='DATE_TIME', how='inner')
    df = df[(df['AC_POWER'] >= 0) & (df['DC_POWER'] >= 0)]
    df = df.dropna(subset=['AC_POWER', 'DC_POWER', 'IRRADIATION',
                            'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']).reset_index(drop=True)

    df['HOUR']                 = df['DATE_TIME'].dt.hour
    df['TOTAL_MINUTES_PASSED'] = df['HOUR'] * 60 + df['DATE_TIME'].dt.minute
    df['MONTH']                = df['DATE_TIME'].dt.month
    df['DATE_STR']             = df['DATE_TIME'].dt.date.astype(str)

    df['DELTA_TEMPERATURE'] = df['MODULE_TEMPERATURE'] - df['AMBIENT_TEMPERATURE']
    df['DC_AC_RATIO']       = df['DC_POWER']   / (df['AC_POWER']   + 1e-6)
    df['IRRAD_X_MODULE']    = df['IRRADIATION'] * df['MODULE_TEMPERATURE']

    if 'SOURCE_KEY' in df.columns:
        df['SOURCE_KEY_NUM'] = LabelEncoder().fit_transform(df['SOURCE_KEY'])
    else:
        df['SOURCE_KEY_NUM'] = 0

    grp = 'SOURCE_KEY' if 'SOURCE_KEY' in df.columns else None
    df  = df.sort_values(([grp, 'DATE_TIME'] if grp else ['DATE_TIME'])).reset_index(drop=True)

    for col in ['AC_POWER', 'IRRADIATION', 'MODULE_TEMPERATURE']:
        if grp:
            df[f'{col}_ROLL3_MEAN'] = df.groupby(grp)[col].transform(
                lambda s: s.shift(1).rolling(3, min_periods=1).mean()
            )
        else:
            df[f'{col}_ROLL3_MEAN'] = (
                df[col].shift(1).rolling(3, min_periods=1).mean()
            )

    # Lag features — strictly past values, no leakage
    # Note: DAILY_YIELD excluded — cumulative within-day counter causes look-ahead bias
    for lag in [1, 3]:
        col_name = f'AC_POWER_LAG{lag}'
        if grp:
            df[col_name] = df.groupby(grp)['AC_POWER'].shift(lag).fillna(0)
        else:
            df[col_name] = df['AC_POWER'].shift(lag).fillna(0)

    return df.dropna().reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ☀️ Solar Forecasting")
    st.markdown("**Adarsh Chauhan | Roll 230107008**")
    st.markdown("---")

    st.markdown("### 📂 Upload Data")
    gen_file     = st.file_uploader("Generation Data CSV",     type=['csv'], key='gen')
    weather_file = st.file_uploader("Weather Sensor Data CSV", type=['csv'], key='weather')


    st.markdown("---")
    st.markdown("### ⚙️ ML Settings")
    rf_trees = st.slider("RF — Number of Trees", 50, 300, 150, 50)
    rf_depth = st.selectbox("RF — Max Depth", [10, 15, 20, None], index=1)

    st.markdown("---")
    st.markdown("### 📈 SARIMA Settings")
    sarima_granularity = st.radio(
        "Time aggregation for SARIMA",
        options=["Hourly  (recommended — ~816 pts, s=24, full diagnostics)",
                 "Daily   (34 pts only, s=7, residual plot only)"],
        index=0,
    )
    use_hourly = sarima_granularity.startswith("Hourly")

    if use_hourly:
        st.info("✅ Hourly: ~816 points · season s=24 · full SARIMA diagnostics valid")
    else:
        st.warning("⚠️ Daily: only 34 points · season s=7 · diagnostics unreliable")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **ML:** LR · Decision Tree · Random Forest  
    **Time-Series:** SARIMA (seasonal ARIMA)  
    **Stack:** scikit-learn · statsmodels · Streamlit  
    **Course:** CL653 — AI/ML for Chemical Eng.
    """)
    run_button = st.button("🚀 Run Full Analysis", use_container_width=True, type='primary')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# ☀️ Solar Power Generation Forecasting System")
st.markdown(
    '<span class="badge badge-blue">Plant Analysis</span> '
    '<span class="badge badge-green">Multi-Model ML</span> '
    '<span class="badge badge-gold">SARIMA Time-Series</span>',
    unsafe_allow_html=True
)
st.markdown("---")

import os

DEFAULT_GEN_PATH = "data/Plant_1_Generation_Data.csv"
DEFAULT_WEATHER_PATH = "data/Plant_1_Weather_Sensor_Data.csv"

if gen_file is not None and weather_file is not None:
    gen_bytes = gen_file.read()
    weather_bytes = weather_file.read()
    st.success("✅ Using uploaded files")

else:
    if os.path.exists(DEFAULT_GEN_PATH) and os.path.exists(DEFAULT_WEATHER_PATH):
        st.info("📁 Using default dataset")
        with open(DEFAULT_GEN_PATH, "rb") as f:
            gen_bytes = f.read()
        with open(DEFAULT_WEATHER_PATH, "rb") as f:
            weather_bytes = f.read()
    else:
        st.error("❌ Default files not found. Please upload.")
        st.stop()

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading and processing data..."):
    try:
        df = load_and_process(gen_bytes, weather_bytes)
    except Exception as e:
        st.error(f"Data error: {e}")
        st.stop()

st.success(f"✅ **{len(df):,} rows** · **{len(df.columns)} features** · **{df['DATE_STR'].nunique()} days**")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA & Overview",
    "🌳 ML Models",
    "📈 SARIMA Time-Series",
    "🏆 Model Comparison",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## 📊 Exploratory Data Analysis")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Rows",       f"{len(df):,}")
    c2.metric("Unique Days",       df['DATE_STR'].nunique())
    c3.metric("Avg AC Power",     f"{df['AC_POWER'].mean():.1f} kW")
    c4.metric("Peak AC Power",    f"{df['AC_POWER'].max():.1f} kW")
    c5.metric("Avg Irradiation",  f"{df['IRRADIATION'].mean():.3f}")

    st.markdown("---")

    # Correlation
    st.markdown("### 🔗 Spearman Correlation Matrix")
    corr_cols = [c for c in ['AC_POWER','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE',
                               'IRRADIATION','DELTA_TEMPERATURE','IRRAD_X_MODULE',
                               'TOTAL_MINUTES_PASSED'] if c in df.columns]
    corr = df[corr_cols].corr(method='spearman')
    fig, ax = plt.subplots(figsize=(9,5))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr,dtype=bool)),
                ax=ax, cmap='coolwarm', center=0, annot=True, fmt='.2f',
                linewidths=0.4, annot_kws={'size':9}, cbar_kws={'shrink':0.7})
    ax.tick_params(labelsize=9)
    st.pyplot(fig, use_container_width=True); plt.close()

    # Distributions
    st.markdown("### 📉 Distributions (Daytime — AC_POWER > 0)")
    day_df = df[df['AC_POWER'] > 0]
    fig, axes = plt.subplots(1, 3, figsize=(16,4))
    for ax,(col,color,xlabel) in zip(axes, [
            ('AC_POWER',C1,'AC Power (kW)'),
            ('IRRADIATION',C2,'Irradiation (W/m²)'),
            ('DELTA_TEMPERATURE',C3,'ΔTemp (°C)')]):
        v = day_df[col]; v = v[v.between(*np.percentile(v,[1,99]))]
        ax.hist(v, bins=40, color=color, alpha=0.85, edgecolor='none')
        ax.axvline(v.mean(), color=C2, lw=1.5, ls='--', label=f'μ={v.mean():.2f}')
        ax.set_xlabel(xlabel); ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # Hourly profile
    st.markdown("### ⏰ Average AC Power by Hour")
    hourly = df.groupby('HOUR')['AC_POWER'].mean()
    fig, ax = plt.subplots(figsize=(12,3.5))
    ax.bar(hourly.index, hourly.values, color=C1, alpha=0.85, edgecolor='none')
    ax.set_xlabel('Hour of Day'); ax.set_ylabel('Avg AC Power (kW)')
    ax.set_title('Diurnal Generation Profile')
    st.pyplot(fig, use_container_width=True); plt.close()

    # Daily totals
    st.markdown("### 📆 Daily Totals — Best & Worst Day")
    daily = df.groupby('DATE_STR')['AC_POWER'].sum().sort_index()
    best_d, worst_d = daily.idxmax(), daily.idxmin()
    fig, ax = plt.subplots(figsize=(16,3.5))
    ax.bar(range(len(daily)), daily.values,
           color=[C3 if d==best_d else C2 if d==worst_d else C1 for d in daily.index],
           edgecolor='none', alpha=0.9)
    ax.set_xticks(range(len(daily)))
    ax.set_xticklabels(daily.index, rotation=45, ha='right', fontsize=8)
    ax.axhline(daily.mean(), color='#8b949e', lw=1.2, ls='--',
               label=f'Mean: {daily.mean():.0f}')
    ax.legend(fontsize=9); ax.set_ylabel('Total AC Power (kW)')
    col_l,col_r = st.columns(2)
    col_l.success(f"🟢 Best: **{best_d}** — `{daily[best_d]:,.0f}` kW")
    col_r.error(  f"🔴 Worst: **{worst_d}** — `{daily[worst_d]:,.0f}` kW")
    st.pyplot(fig, use_container_width=True); plt.close()

    # CoV fault table
    st.markdown("### ⚡ Fault Severity — Coefficient of Variation")
    agg    = df.groupby(['DATE_STR','HOUR'])['DC_POWER'].sum().reset_index()
    dt_agg = agg[agg['DC_POWER'] > 0]
    cov_df = dt_agg.groupby('DATE_STR')['DC_POWER'].agg(['std','mean'])
    cov_df['CoV'] = cov_df['std']/(cov_df['mean']+1e-6)
    cov_df['Severity'] = cov_df['CoV'].apply(lambda v:
        '🔴 Very High' if v>0.7 else '🟡 High' if v>0.4
        else '🟢 Low' if v>0.2 else '✅ Stable')
    cov_df = (cov_df.sort_values('CoV',ascending=False)
                    .reset_index()[['DATE_STR','CoV','Severity']])
    cov_df.columns = ['Date','CoV','Severity']
    st.dataframe(cov_df, use_container_width=True, height=300)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ML MODELS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## 🌳 Machine Learning Models")
    st.markdown("""
    <div class="callout callout-ok">
    ✅ <b>Leakage-free feature set:</b> DC_POWER (ρ=0.99 with target), EFFICIENCY
    (AC/IRRAD — uses target), DC_AC_RATIO (DC/AC — uses target), and DAILY_YIELD
    (cumulative within-day counter) are all excluded. Rolling means use a 1-step
    shift so they contain only past values at t-1, t-2, t-3.
    </div>
    """, unsafe_allow_html=True)

    if not run_button:
        st.info("Click **Run Full Analysis** in the sidebar to train models.")
    else:

        # Make sure data is in correct time order
        df = df.sort_values(['SOURCE_KEY', 'DATE_TIME'])

        # Create future target (next time step)
        df['TARGET'] = df.groupby('SOURCE_KEY')['AC_POWER'].shift(-1)

        # Remove last rows (they have no future value)
        df = df.dropna().reset_index(drop=True)

        TARGET   = 'TARGET'

        # Safe feature set — zero leakage
        CANDIDATE = [
            'AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION',
            'TOTAL_MINUTES_PASSED','HOUR','MONTH',
            'DELTA_TEMPERATURE','IRRAD_X_MODULE',
            'SOURCE_KEY_NUM',
            'AC_POWER_ROLL3_MEAN','IRRADIATION_ROLL3_MEAN',
            'MODULE_TEMPERATURE_ROLL3_MEAN',
            'AC_POWER_LAG1','AC_POWER_LAG3',
        ]
        FEATURES = [f for f in CANDIDATE if f in df.columns]

        X = df[FEATURES].values
        y = df[TARGET].values
        split = int(0.8*len(X))
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        results = {}

        with st.spinner("Training Linear Regression..."):
            sc    = StandardScaler()
            lr    = LinearRegression().fit(sc.fit_transform(X_tr), y_tr)
            yp_lr = lr.predict(sc.transform(X_te))
            results['Linear Regression'] = (yp_lr, compute_metrics(y_te, yp_lr))

        with st.spinner("Training Decision Tree..."):
            dt    = DecisionTreeRegressor(max_depth=15, min_samples_split=10,
                                          random_state=SEED).fit(X_tr, y_tr)
            yp_dt = dt.predict(X_te)
            results['Decision Tree'] = (yp_dt, compute_metrics(y_te, yp_dt))

        with st.spinner(f"Training Random Forest ({rf_trees} trees)..."):
            rf    = RandomForestRegressor(n_estimators=rf_trees, max_depth=rf_depth,
                                          min_samples_split=5, max_features='sqrt',
                                          n_jobs=-1, random_state=SEED).fit(X_tr, y_tr)
            yp_rf = rf.predict(X_te)   # predict on held-out test set
            results['Random Forest'] = (yp_rf, compute_metrics(y_te, yp_rf))

        # Metrics
        st.markdown("### 📋 Test Set Performance")
        cols = st.columns(3)
        for (name,(_, m)), col in zip(results.items(), cols):
            with col:
                st.markdown(f"**{name}**")
                for k,v in m.items():
                    col.metric(k, f"{v}{'%' if k in ['R²','MAPE'] else ''}")

        best_ml = min(results, key=lambda k: results[k][1]['RMSE'])
        st.markdown(
            f'<div class="callout callout-ok">🏆 Best model (RMSE): '
            f'<b>{best_ml}</b> — RMSE={results[best_ml][1]["RMSE"]}, '
            f'R²={results[best_ml][1]["R²"]}%</div>',
            unsafe_allow_html=True
        )

        # ── Forecast vs Actual — time-series plot for all models ─────────────
        st.markdown("### 📉 Forecast vs Actual — Test Period (All Models)")
        n_show = min(500, len(y_te))
        fig, ax = plt.subplots(figsize=(16,5))
        ax.plot(range(n_show), y_te[:n_show], color='#c9d1d9', lw=1.8,
                label='Actual', zorder=3)
        ax.plot(range(n_show), yp_lr[:n_show], color=C4, lw=1.2, ls=':',
                alpha=0.85, label='Linear Regression')
        ax.plot(range(n_show), yp_dt[:n_show], color=C3, lw=1.2, ls='--',
                alpha=0.85, label='Decision Tree')
        ax.plot(range(n_show), yp_rf[:n_show], color=C2, lw=1.4, ls='-',
                alpha=0.9,  label='Random Forest')
        ax.fill_between(range(n_show), y_te[:n_show], yp_rf[:n_show],
                        alpha=0.08, color=C2)
        ax.set_xlabel(f'Test time steps (15-min intervals, first {n_show} of {len(y_te)} shown)')
        ax.set_ylabel('AC Power (kW)')
        ax.set_title('Forecast vs Actual — Test Period')
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        # Feature importance
        st.markdown("### 🔍 Random Forest — Feature Importance (Gini)")
        importances = rf.feature_importances_
        top_idx     = np.argsort(importances)[-15:]
        fig, ax = plt.subplots(figsize=(10,5))
        fi_colors = [C2 if i>=len(top_idx)-3 else C1 for i in range(len(top_idx))]
        ax.barh([FEATURES[i] for i in top_idx], importances[top_idx],
                color=fi_colors, edgecolor='none')
        ax.set_title('Top-15 Feature Importances')
        ax.set_xlabel('Gini Importance')
        st.pyplot(fig, use_container_width=True); plt.close()

        # Scatter + residual for RF
        st.markdown("### 🎯 Random Forest — Scatter & Residual")
        fig, axes = plt.subplots(1, 2, figsize=(14,5))
        ax = axes[0]
        ax.scatter(y_te, yp_rf, alpha=0.2, s=8, color=C1, rasterized=True)
        lo = float(min(y_te.min(), yp_rf.min()))
        hi = float(max(y_te.max(), yp_rf.max()))
        ax.plot([lo,hi],[lo,hi], color=C2, lw=2, ls='--', label='Ideal')
        ax.set_xlabel('Actual AC Power (kW)'); ax.set_ylabel('Predicted AC Power (kW)')
        ax.set_title('Actual vs Predicted'); ax.legend(fontsize=8)
        txt = '\n'.join([f'{k}: {v}' for k,v in results['Random Forest'][1].items()])
        ax.text(0.05,0.95,txt,transform=ax.transAxes,va='top',fontsize=9,
                family='monospace',color='#c9d1d9',
                bbox=dict(boxstyle='round',facecolor=PANEL_BG,alpha=0.85))
        ax = axes[1]
        residuals = y_te - yp_rf
        ax.scatter(yp_rf, residuals, alpha=0.2, s=8, color=C3, rasterized=True)
        ax.axhline(0, color=C2, lw=1.5, ls='--')
        ax.set_xlabel('Predicted (kW)'); ax.set_ylabel('Residual')
        ax.set_title('Residual Plot')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        # Cross-check table
        st.markdown("### 📄 Prediction Cross-Check (first 20 test rows)")
        cc = pd.DataFrame({
            'Actual':  y_te[:20].round(2),
            'LR Pred': yp_lr[:20].round(2),
            'DT Pred': yp_dt[:20].round(2),
            'RF Pred': yp_rf[:20].round(2),
        })
        cc['RF |Error|'] = (cc['Actual']-cc['RF Pred']).abs().round(2)
        st.dataframe(cc, use_container_width=True)
        st.markdown(
            f"RF within ±5 kW: **{(np.abs(y_te-yp_rf)<=5).mean()*100:.1f}%** &nbsp;|&nbsp; "
            f"within ±20 kW: **{(np.abs(y_te-yp_rf)<=20).mean()*100:.1f}%**"
        )

        st.session_state['ml_results'] = {k:v[1] for k,v in results.items()}
        st.session_state['best_ml']    = best_ml


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SARIMA TIME-SERIES
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## 📈 SARIMA — Seasonal Time-Series Forecasting")

    if use_hourly:
        season = 24; freq_label = "hour"; y_unit = "Hourly AC Power Sum (kW)"
        sarima_label = "SARIMA(1,1,1)(1,0,0,24) — Hourly"
        st.markdown("""
        <div class="callout callout-ok">
        ✅ <b>Hourly mode:</b> ~816 data points · seasonal period s=24 (daily bell-curve)
        · seasonal_order D=0 avoids over-differencing · full four-panel diagnostics valid.
        </div>""", unsafe_allow_html=True)
    else:
        season = 7; freq_label = "day"; y_unit = "Daily AC Power Sum (kW)"
        sarima_label = "ARIMA(1,1,1) — Daily"
        st.markdown("""
        <div class="callout callout-warn">
        ⚠️ <b>Daily mode:</b> only 34 data points · simple ARIMA(1,1,1) used
        (no seasonal component — s=7 seasonal differencing on 27 training points
        is statistically unreliable) · simplified two-panel residual analysis only.
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    **{sarima_label}** operates on plant-level aggregated AC power (sum across all inverters).
    It uses only the historical power series — no weather features — making it a pure
    time-series baseline that complements the weather-feature-based ML models.
    """)

    if not run_button:
        st.info("Click **Run Full Analysis** in the sidebar to run SARIMA.")
    else:
        # Build series
        if use_hourly:
            df_copy = df.copy()
            df_copy['HOUR_DT'] = df_copy['DATE_TIME'].dt.floor('H')
            series = (df_copy.groupby('HOUR_DT')['AC_POWER']
                             .sum().sort_index().rename('AC_POWER'))
        else:
            series = (df.groupby('DATE_STR')['AC_POWER']
                        .sum().sort_index().rename('AC_POWER'))
            series.index = pd.to_datetime(series.index)

        n_pts   = len(series)
        n_test  = max(season*2, int(n_pts*0.2))
        n_train = n_pts - n_test
        train_s = series.iloc[:n_train]
        test_s  = series.iloc[n_train:]

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Points",       n_pts)
        c2.metric("Seasonal Period",     season if use_hourly else "N/A (no seasonal)")
        c3.metric(f"Train {freq_label}s", n_train)
        c4.metric(f"Test {freq_label}s",  n_test)

        # Split plot
        st.markdown("### 🔀 Train / Test Split")
        show_train = train_s.iloc[-season*3:]
        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(show_train.index, show_train.values, color=C1, lw=1.4,
                label=f'Train (last {season*3} {freq_label}s shown)')
        ax.plot(test_s.index,  test_s.values,  color=C3, lw=2, label='Test (actual)')
        ax.axvline(train_s.index[-1], color='#8b949e', lw=1.5, ls='--', alpha=0.7)
        ax.set_ylabel(y_unit)
        ax.set_title(f'{sarima_label} — Data Split')
        ax.legend(fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()

        with st.spinner(f"Fitting {sarima_label}..."):
            try:
                if use_hourly:
                    # D=0: seasonal AR only, no seasonal differencing
                    # Avoids over-differencing the zero-heavy nighttime series
                    sarima_model = SARIMAX(
                        train_s,
                        order=(1,1,1),
                        seasonal_order=(1,0,0,season),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                else:
                    # Daily: simple ARIMA, no seasonal component
                    # With only 27 training points, s=7 seasonal differencing
                    # removes too many degrees of freedom → unreliable
                    sarima_model = SARIMAX(
                        train_s,
                        order=(1,1,1),
                        seasonal_order=(0,0,0,0),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )

                sarima_result = sarima_model.fit(disp=False, maxiter=200)

                fc_obj  = sarima_result.get_forecast(steps=n_test)
                fc_mean = fc_obj.predicted_mean
                fc_ci   = fc_obj.conf_int(alpha=0.05)
                fc_mean.index = test_s.index
                fc_ci.index   = test_s.index

                fc_vals  = np.clip(fc_mean.values, 0, None)
                m_sarima = compute_metrics(test_s.values, fc_vals)

                # Metrics
                st.markdown("### 📋 SARIMA Performance")
                st.markdown(
                    '<div class="callout">Metrics are on aggregated plant totals '
                    '— not directly comparable to per-inverter 15-min ML metrics. '
                    'R² is the most useful scale-independent indicator.</div>',
                    unsafe_allow_html=True
                )
                mcols = st.columns(4)
                for col,(k,v) in zip(mcols, m_sarima.items()):
                    col.metric(k, f"{v}{'%' if k in ['R²','MAPE'] else ''}")

                # Forecast vs Actual
                st.markdown("### 📉 Forecast vs Actual")
                fig, ax = plt.subplots(figsize=(14,5))
                ax.plot(show_train.index, show_train.values, color=C1, lw=1.4,
                        label='Train history')
                ax.plot(test_s.index, test_s.values, color=C3, lw=2.5,
                        label='Actual (test)')
                ax.plot(test_s.index, fc_vals, color=C2, lw=2, ls='--',
                        marker='o', ms=5, label=f'{sarima_label} Forecast')
                ax.fill_between(fc_ci.index,
                                np.clip(fc_ci.iloc[:,0].values, 0, None),
                                fc_ci.iloc[:,1].values,
                                color=C2, alpha=0.15, label='95% CI')
                ax.axvline(train_s.index[-1], color='#8b949e', lw=1.5, ls='--', alpha=0.7)
                ax.set_ylabel(y_unit)
                ax.set_title(f'{sarima_label} — Forecast with 95% Confidence Interval')
                ax.legend(fontsize=9)
                st.pyplot(fig, use_container_width=True); plt.close()

                # Diagnostics
                if use_hourly:
                    st.markdown("### 🔬 SARIMA Diagnostics (Full)")
                    st.markdown("""
                    <div class="callout callout-ok">
                    ✅ With ~816 training points, all four diagnostic panels are statistically meaningful:
                    standardised residuals should show no pattern, histogram should be approximately normal,
                    Q-Q plot should follow the diagonal, ACF should have no significant lags.
                    </div>
                    """, unsafe_allow_html=True)
                    try:
                        import scipy.stats as stats
                        from statsmodels.graphics.tsaplots import plot_acf

                        resid = sarima_result.resid.dropna()
                        std_resid = resid / resid.std()

                        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

                        # 1️⃣ Standardized residuals (time plot)
                        ax = axes[0, 0]
                        ax.plot(std_resid, color=C1, lw=1.2)
                        ax.axhline(0, color=C2, ls='--')
                        ax.set_title('Standardized Residuals')

                        # 2️⃣ Histogram + Estimated Density
                        ax = axes[0, 1]

                        # Histogram (normalized)
                        ax.hist(std_resid, bins=20, density=True, color=C1, alpha=0.6, edgecolor='none')

                        # KDE (estimated density)
                        import scipy.stats as stats
                        kde = stats.gaussian_kde(std_resid)
                        x_vals = np.linspace(std_resid.min(), std_resid.max(), 200)
                        ax.plot(x_vals, kde(x_vals), color=C2, lw=2, label='KDE')

                        # Normal distribution reference (very useful)
                        from scipy.stats import norm
                        ax.plot(x_vals, norm.pdf(x_vals, 0, 1), color=C3, lw=2, ls='--', label='Normal')

                        ax.set_title('Histogram + Density')
                        ax.set_xlabel('Standardized Residual')
                        ax.set_ylabel('Density')
                        ax.legend(fontsize=8)

                        # 3️⃣ Q-Q plot
                        ax = axes[1, 0]
                        stats.probplot(std_resid, dist="norm", plot=ax)
                        ax.set_title('Normal Q-Q')

                        # 4️⃣ ACF (correlogram)
                        ax = axes[1, 1]
                        plot_acf(std_resid, lags=40, ax=ax)
                        ax.set_title('Correlogram (ACF)')

                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                    except Exception as diag_e:
                        st.warning(f"Diagnostics plot error: {diag_e}")
                else:
                    st.markdown("### 🔬 Residual Diagnostics (Daily — Limited Reliability)")
                    st.markdown("""
                    <div class="callout callout-warn">
                    ⚠️ Only 34 daily points — diagnostics are shown for reference only.
                    Q-Q and ACF may not be statistically reliable. Switch to <b>Hourly</b> mode for proper diagnostics.
                    </div>
                    """, unsafe_allow_html=True)

                    import scipy.stats as stats
                    from statsmodels.graphics.tsaplots import plot_acf

                    residuals = sarima_result.resid.dropna()

                    # ── EXACT SAME AS NOTEBOOK ─────────────────────────────
                    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

                    # 1. Residuals vs Time
                    axes[0, 0].plot(residuals, color=C1)
                    axes[0, 0].axhline(0, color=C2, linestyle='--')
                    axes[0, 0].set_title('Residuals vs Time')
                    axes[0, 0].set_xlabel('Time')
                    axes[0, 0].set_ylabel('Residual')

                    # 2. Histogram (NO KDE — same as notebook)
                    axes[0, 1].hist(residuals, bins=10, color=C1, alpha=0.85, edgecolor='none')
                    axes[0, 1].set_title('Residual Distribution')
                    axes[0, 1].set_xlabel('Residual')
                    axes[0, 1].set_ylabel('Frequency')

                    # 3. Q-Q Plot
                    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
                    axes[1, 0].set_title('Q-Q Plot')

                    # 4. ACF Plot
                    plot_acf(residuals, lags=10, ax=axes[1, 1])
                    axes[1, 1].set_title('Residual Autocorrelation')

                    fig.suptitle('SARIMA Residual Analysis',
                                fontsize=13, y=1.02,
                                color='#c9d1d9', fontweight='bold')

                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                # Model summary
                with st.expander("📋 SARIMA Model Summary"):
                    st.text(str(sarima_result.summary()))

                st.session_state['sarima_metrics'] = m_sarima
                st.session_state['sarima_label']   = sarima_label

            except Exception as e:
                st.error(f"SARIMA fitting failed: {e}")
                st.info("Try switching to Hourly mode, or check that the series has sufficient "
                        "non-zero values (too many consecutive nighttime zeros can cause issues).")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## 🏆 Model Comparison")

    if 'ml_results' not in st.session_state:
        st.info("Run the analysis first (sidebar button).")
    else:
        ml_results = st.session_state['ml_results']
        best_ml    = st.session_state.get('best_ml',
                         min(ml_results, key=lambda k: ml_results[k]['RMSE']))

        st.success(f"🏆 Best ML model on 15-min inverter data (RMSE): **{best_ml}**")

        if 'sarima_metrics' in st.session_state:
            sl = st.session_state.get('sarima_label','SARIMA')
            agg_type = "hourly" if "Hourly" in sl else "daily"
            st.markdown(f"""
            <div class="callout callout-warn">
            ⚠️ <b>Scale note:</b> ML models operate on 15-minute per-inverter rows.
            {sl} operates on {agg_type} plant-level totals. Their absolute metrics
            (MAE, RMSE) are <b>not directly comparable</b> — different units and scale.
            R² is the most useful cross-scale indicator.
            </div>""", unsafe_allow_html=True)

        # ML comparison charts
        st.markdown("### 📊 ML Models — Same Scale (15-min Inverter Data)")
        ml_names  = list(ml_results.keys())
        ml_colors = [C4,C3,C1][:len(ml_names)]
        fig, axes = plt.subplots(1,3, figsize=(16,5))
        fig.suptitle('ML Model Comparison — 15-min Inverter Scale',
                     fontsize=12, color='#c9d1d9')
        for ax, metric in zip(axes, ['MAE','RMSE','R²']):
            vals  = [ml_results[m][metric] for m in ml_names]
            bars  = ax.bar(ml_names, vals, color=ml_colors, edgecolor='none',
                           alpha=0.9, width=0.5)
            ax.set_title(f'{metric}  ({"higher" if metric=="R²" else "lower"} = better)')
            ax.set_ylabel(metric+(' (%)' if metric=='R²' else ''))
            ax.tick_params(axis='x', labelsize=8, rotation=12)
            best_v = max(vals) if metric=='R²' else min(vals)
            for bar,v in zip(bars,vals):
                bar.set_alpha(1.0 if v==best_v else 0.55)
                ax.text(bar.get_x()+bar.get_width()/2, v+max(vals)*0.01,
                        f'{v:.2f}', ha='center', fontsize=8, color='#c9d1d9')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        # Full metrics table
        st.markdown("### 📋 All Model Metrics")
        all_m = dict(ml_results)
        if 'sarima_metrics' in st.session_state:
            all_m[st.session_state.get('sarima_label','SARIMA')] = \
                st.session_state['sarima_metrics']
        df_m = pd.DataFrame(all_m).T.round(3)
        st.dataframe(df_m, use_container_width=True)

        report = {
            'generated_at': str(pd.Timestamp.now()),
            'notes': {
                'ml_scale'    : '15-minute per-inverter rows; temporal 80/20 split',
                'sarima_scale': 'Hourly or daily plant totals; different scale from ML',
                'features_excluded': 'DC_POWER, DC_AC_RATIO, EFFICIENCY, DAILY_YIELD',
            },
            'models': all_m,
            'best_ml_model': best_ml,
        }
        st.download_button(
            "⬇️ Download Metrics Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name='solar_forecast_report.json',
            mime='application/json',
        )

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8b949e;font-size:0.85em;'>"
    "Solar Power Forecasting · Adarsh Chauhan · Roll 230107008 · CL653"
    "</div>", unsafe_allow_html=True
)
