import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

st.set_page_config(
    page_title="PGF · Gestión Energética Inteligente",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top:1.4rem; padding-bottom:2rem; max-width:1400px;}
[data-testid="stSidebar"] {background:#081927;}
[data-testid="stSidebar"] * {color:#eef5f7 !important;}
.hero {background:linear-gradient(120deg,#081927 0%,#123c51 70%,#185b74 100%);
padding:28px 34px;border-radius:18px;margin-bottom:22px;color:white;}
.hero h1 {font-size:34px;margin:0 0 6px 0;color:white;}
.hero p {margin:0;color:#cfe1e7;font-size:15px;}
.kpi {background:#f3f7f8;border:1px solid #d9e5e9;border-radius:14px;padding:18px 20px;}
.kpi .label {font-size:12px;color:#708590;text-transform:uppercase;letter-spacing:.08em;}
.kpi .value {font-size:28px;font-weight:700;color:#081927;margin-top:4px;}
.kpi .note {font-size:11px;color:#708590;margin-top:3px;}
.section {font-size:20px;font-weight:700;color:#081927;margin:8px 0 10px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "synthetic_datacenter_hourly_3days.csv"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {path}")
    return pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")

@st.cache_resource
def train_model(df):
    d = df.copy()
    d["hour"] = d["timestamp"].dt.hour
    d["weekday"] = d["timestamp"].dt.dayofweek
    d = d.set_index("timestamp")

    d["it_load_kW_lag1"] = d["it_load_kW"].shift(1)
    d["total_consumption_kW_lag1"] = d["total_consumption_kW"].shift(1)
    d["it_load_kW_roll3"] = d["it_load_kW"].rolling(3).mean()
    d["hour_sin"] = np.sin(2*np.pi*d["hour"]/24)
    d["hour_cos"] = np.cos(2*np.pi*d["hour"]/24)
    d["weekday_sin"] = np.sin(2*np.pi*d["weekday"]/7)
    d["weekday_cos"] = np.cos(2*np.pi*d["weekday"]/7)

    features = [
        "it_load_kW","hvac_kW","it_load_kW_lag1",
        "total_consumption_kW_lag1","it_load_kW_roll3",
        "hour_sin","hour_cos","weekday_sin","weekday_cos",
        "temperature_C"
    ]

    d = d.dropna(subset=features + ["total_consumption_kW"])
    n = int(len(d) * 0.70)
    train = d.iloc[:n]
    test = d.iloc[n:]

    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(train[features], train["total_consumption_kW"])
    pred = model.predict(test[features])

    return model, features, train, test, pred

# IMPORTANT: data/model are created BEFORE any tab uses df.
df = load_data()
model, features, train, test, pred = train_model(df)

with st.sidebar:
    st.markdown("## CONTROL DE LA DEMO")
    st.markdown("**Modelo**  \nRandom Forest")
    st.markdown("**Datos**  \nData Center sintético")
    st.markdown("**Flujo**  \nPredicción · XAI · Escenarios")
    st.divider()
    st.caption("Los escenarios son demostrativos y no representan ahorro validado de Oil Malal.")

st.markdown("""
<div class="hero">
<h1>⚡ PGF · Gestión Energética Inteligente</h1>
<p>De medir → entender → predecir → explicar → explorar</p>
<p style="margin-top:8px;font-size:12px;">
Demostración ML · datos sintéticos del notebook Procesoinicialpgfenergia2 · USM
</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["01 · DESEMPEÑO","02 · PREDICCIÓN","03 · XAI / SHAP","04 · ESCENARIOS"])

with tabs[0]:
    st.markdown('<div class="section">01 · Comportamiento energético</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,label,value,note in [
        (c1,"OBSERVACIONES",len(df),"dataset completo"),
        (c2,"ENTRENAMIENTO",len(train),"70 % del conjunto"),
        (c3,"PRUEBA",len(test),"30 % del conjunto"),
    ]:
        col.markdown(
            f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="note">{note}</div></div>',
            unsafe_allow_html=True
        )

    fig, ax = plt.subplots(figsize=(10,3.6))
    ax.plot(df["timestamp"], df["total_consumption_kW"], label="Consumo total")
    ax.set_ylabel("kW"); ax.set_xlabel("Tiempo")
    ax.grid(alpha=.25); ax.legend()
    st.pyplot(fig, clear_figure=True)

with tabs[1]:
    st.markdown('<div class="section">02 · Predicción · Real vs. Predicho</div>', unsafe_allow_html=True)
    real = test["total_consumption_kW"].values
    mae = mean_absolute_error(real, pred)
    rmse = np.sqrt(mean_squared_error(real, pred))
    r2 = r2_score(real, pred)

    c1,c2,c3 = st.columns(3)
    for col,label,value,note in [
        (c1,"MAE",f"{mae:.3f} kW","error absoluto medio"),
        (c2,"RMSE",f"{rmse:.3f} kW","raíz del error cuadrático"),
        (c3,"R²",f"{r2:.3f}","capacidad explicativa"),
    ]:
        col.markdown(
            f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="note">{note}</div></div>',
            unsafe_allow_html=True
        )

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(test.index, real, label="Real")
    ax.plot(test.index, pred, label="Predicho")
    ax.set_ylabel("Consumo [kW]"); ax.set_xlabel("Hora")
    ax.grid(alpha=.25); ax.legend()
    st.pyplot(fig, clear_figure=True)

with tabs[2]:
    st.markdown('<div class="section">03 · XAI · ¿Por qué predijo eso?</div>', unsafe_allow_html=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test[features])
    importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=features
    ).sort_values()

    fig, ax = plt.subplots(figsize=(9,4.5))
    top = importance.tail(8)
    ax.barh([x.replace("_"," ") for x in top.index], top.values)
    ax.set_xlabel("Impacto medio absoluto SHAP")
    ax.grid(axis="x", alpha=.25)
    st.pyplot(fig, clear_figure=True)
    st.info("SHAP muestra qué variables influyen en la salida del modelo; no implica causalidad física.")

with tabs[3]:
    st.markdown('<div class="section">04 · Escenarios · explorar antes de actuar</div>', unsafe_allow_html=True)
    st.write(
        "Demostración metodológica con el mismo modelo. "
        "Los cambios son simulados y no representan ahorro validado de Oil Malal."
    )

    base_row = test.iloc[-1][features].copy()
    it = st.slider(
        "IT Load [kW]",
        float(test["it_load_kW"].min()),
        float(test["it_load_kW"].max()),
        float(base_row["it_load_kW"])
    )
    hv = st.slider(
        "HVAC [kW]",
        float(test["hvac_kW"].min()),
        float(test["hvac_kW"].max()),
        float(base_row["hvac_kW"])
    )

    row = base_row.copy()
    row["it_load_kW"] = it
    row["hvac_kW"] = hv
    y = float(model.predict(pd.DataFrame([row], columns=features))[0])

    c1,c2,c3 = st.columns(3)
    c1.metric("IT Load", f"{it:.2f} kW")
    c2.metric("HVAC", f"{hv:.2f} kW")
    c3.metric("Consumo predicho", f"{y:.2f} kW")

st.divider()
st.caption("PGF · Paul Gálvez · USM · Demo técnica")
