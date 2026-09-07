
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

st.set_page_config(page_title="PGF · Gestión Energética Inteligente", layout="wide")

st.title("PGF · Gestión Energética Inteligente")
st.caption("Demostración ML · datos sintéticos del notebook Procesoinicialpgfenergia2")

@st.cache_data
def load_data():
    # Resolver la ruta desde la raíz del proyecto, no desde la carpeta
    # actual de PowerShell. Así funciona con: streamlit run app\\app.py
    ROOT = Path(__file__).resolve().parents[1]
    p = ROOT / "data" / "synthetic_datacenter_hourly_3days.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en: {p}\\n"
            "Verifica que el CSV esté dentro de la carpeta data del proyecto."
        )
    df=pd.read_csv(p, parse_dates=["timestamp"])
    return df.sort_values("timestamp")

@st.cache_resource
def train_model(df):
    d=df.copy()
    d["hour"]=d["timestamp"].dt.hour
    d["weekday"]=d["timestamp"].dt.dayofweek
    d=d.set_index("timestamp")
    d["it_load_kW_lag1"]=d["it_load_kW"].shift(1)
    d["total_consumption_kW_lag1"]=d["total_consumption_kW"].shift(1)
    d["it_load_kW_roll3"]=d["it_load_kW"].rolling(3).mean()
    d["hour_sin"]=np.sin(2*np.pi*d["hour"]/24)
    d["hour_cos"]=np.cos(2*np.pi*d["hour"]/24)
    d["weekday_sin"]=np.sin(2*np.pi*d["weekday"]/7)
    d["weekday_cos"]=np.cos(2*np.pi*d["weekday"]/7)
    features=['it_load_kW','hvac_kW','it_load_kW_lag1','total_consumption_kW_lag1',
              'it_load_kW_roll3','hour_sin','hour_cos','weekday_sin','weekday_cos','temperature_C']
    d=d.dropna(subset=features+["total_consumption_kW"])
    n=int(len(d)*.7)
    tr,te=d.iloc[:n],d.iloc[n:]
    model=RandomForestRegressor(n_estimators=200,max_depth=10,random_state=42,n_jobs=-1)
    model.fit(tr[features],tr["total_consumption_kW"])
    pred=model.predict(te[features])
    return model,features,tr,te,pred

df=load_data()
model,features,tr,te,pred=train_model(df)
real=te["total_consumption_kW"].values

tabs=st.tabs(["01 · Desempeño","02 · Predicción","03 · XAI / SHAP","04 · Escenarios"])

with tabs[0]:
    st.subheader("Comportamiento energético")
    c1,c2,c3=st.columns(3)
    c1.metric("Observaciones",len(df))
    c2.metric("Entrenamiento",len(tr))
    c3.metric("Prueba",len(te))
    fig,ax=plt.subplots(figsize=(10,3.6))
    ax.plot(df["timestamp"],df["total_consumption_kW"],label="Consumo total")
    ax.set_ylabel("kW"); ax.set_xlabel("Tiempo"); ax.grid(alpha=.25)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

with tabs[1]:
    st.subheader("Predicción · Real vs. Predicho")
    mae=mean_absolute_error(real,pred); rmse=np.sqrt(mean_squared_error(real,pred)); r2=r2_score(real,pred)
    c1,c2,c3=st.columns(3)
    c1.metric("MAE",f"{mae:.3f} kW")
    c2.metric("RMSE",f"{rmse:.3f} kW")
    c3.metric("R²",f"{r2:.3f}")
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(te.index,real,label="Real")
    ax.plot(te.index,pred,label="Predicho")
    ax.set_ylabel("Consumo [kW]"); ax.set_xlabel("Hora")
    ax.grid(alpha=.25); ax.legend()
    st.pyplot(fig, clear_figure=True)

with tabs[2]:
    st.subheader("XAI · ¿Por qué predijo eso?")
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
    st.subheader("Escenarios · explorar antes de actuar")
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
