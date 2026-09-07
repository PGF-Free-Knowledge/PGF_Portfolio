
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
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root { --navy:#081927; --blue:#185b74; --cyan:#2cAab8; --orange:#ef8919; --green:#459d69; }
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px;}
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
.note {font-size:12px;color:#708590;}
.stTabs [data-baseweb="tab-list"] {gap:7px;}
.stTabs [data-baseweb="tab"] {height:44px;border-radius:10px 10px 0 0;padding:0 18px;}
.stButton>button {border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>⚡ PGF · Gestión Energética Inteligente</h1>
  <p>De medir → entender → predecir → explicar → explorar</p>
  <p style="margin-top:8px;font-size:12px;">Demostración ML · datos sintéticos del notebook Procesoinicialpgfenergia2 · USM</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## CONTROL DE LA DEMO")
    st.markdown("**Modelo**  \nRandom Forest")
    st.markdown("**Datos**  \nData Center sintético")
    st.markdown("**Flujo**  \nPredicción · XAI · Escenarios")
    st.divider()
    st.caption("Los escenarios son demostrativos y no representan ahorro validado de Oil Malal.")

tabs=st.tabs(["01 · DESEMPEÑO","02 · PREDICCIÓN","03 · XAI / SHAP","04 · ESCENARIOS"])

with tabs[0]:
    st.markdown('<div class="section">01 · Comportamiento energético</div>', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    for col,label,value,note in [
        (c1,"OBSERVACIONES",len(df),"dataset completo"),
        (c2,"ENTRENAMIENTO",len(tr),"70 % del conjunto"),
        (c3,"PRUEBA",len(te),"30 % del conjunto")]:
        col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)
    fig,ax=plt.subplots(figsize=(10,3.6))
    ax.plot(df["timestamp"],df["total_consumption_kW"],label="Consumo total")
    ax.set_ylabel("kW"); ax.set_xlabel("Tiempo"); ax.grid(alpha=.25)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

with tabs[1]:
    st.markdown('<div class="section">02 · Predicción · Real vs. Predicho</div>', unsafe_allow_html=True)
    mae=mean_absolute_error(real,pred); rmse=np.sqrt(mean_squared_error(real,pred)); r2=r2_score(real,pred)
    c1,c2,c3=st.columns(3)
    for col,label,value,note in [
        (c1,"MAE",f"{mae:.3f} kW","error absoluto medio"),
        (c2,"RMSE",f"{rmse:.3f} kW","raíz del error cuadrático"),
        (c3,"R²",f"{r2:.3f}","capacidad explicativa")]:
        col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(te.index,real,label="Real")
    ax.plot(te.index,pred,label="Predicho")
    ax.set_ylabel("Consumo [kW]"); ax.set_xlabel("Hora")
    ax.grid(alpha=.25); ax.legend()
    st.pyplot(fig, clear_figure=True)

with tabs[2]:
    st.markdown('<div class="section">03 · XAI · ¿Por qué predijo eso?</div>', unsafe_allow_html=True)
    expl=shap.TreeExplainer(model)
    sv=expl.shap_values(te[features])
    imp=pd.Series(np.abs(sv).mean(axis=0),index=features).sort_values()
    fig,ax=plt.subplots(figsize=(9,4.5))
    top=imp.tail(8)
    ax.barh([x.replace("_"," ") for x in top.index],top.values)
    ax.set_xlabel("Impacto medio absoluto SHAP")
    ax.grid(axis="x",alpha=.25)
    st.pyplot(fig, clear_figure=True)
    st.info("SHAP muestra qué variables explican el comportamiento del modelo; no implica causalidad física.")

with tabs[3]:
    st.markdown('<div class="section">04 · Escenarios · explorar antes de actuar</div>', unsafe_allow_html=True)
    st.write("Demostración metodológica con el mismo modelo. Los cambios son simulados y no representan ahorro validado de Oil Malal.")
    base_row=te.iloc[-1][features].copy()
    it=st.slider("IT Load [kW]",float(te["it_load_kW"].min()),float(te["it_load_kW"].max()),float(base_row["it_load_kW"]))
    hv=st.slider("HVAC [kW]",float(te["hvac_kW"].min()),float(te["hvac_kW"].max()),float(base_row["hvac_kW"]))
    row=base_row.copy(); row["it_load_kW"]=it; row["hvac_kW"]=hv
    y=float(model.predict(pd.DataFrame([row],columns=features))[0])
    c1,c2,c3=st.columns(3)
    c1.metric("IT Load",f"{it:.2f} kW")
    c2.metric("HVAC",f"{hv:.2f} kW")
    c3.metric("Consumo predicho",f"{y:.2f} kW")
    st.caption("Modifica las variables y observa cómo responde la predicción. Esta es la demostración interactiva para la charla.")

st.divider()
st.caption("PGF · Paul Gálvez · USM · Demo técnica")
