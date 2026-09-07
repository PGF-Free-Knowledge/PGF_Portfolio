import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

st.set_page_config(
    page_title="PGF · Gestión Energética Inteligente",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# ESTILO
# ============================================================
st.markdown("""
<style>
.block-container {padding-top:1.2rem; padding-bottom:2rem; max-width:1400px;}
[data-testid="stSidebar"] {background:#081927;}
[data-testid="stSidebar"] * {color:#eef5f7 !important;}
.hero {background:linear-gradient(120deg,#081927 0%,#123c51 70%,#185b74 100%);
padding:26px 32px;border-radius:18px;margin-bottom:18px;color:white;}
.hero h1 {font-size:33px;margin:0 0 6px 0;color:white;}
.hero p {margin:0;color:#cfe1e7;font-size:14px;}
.kpi {background:#f3f7f8;border:1px solid #d9e5e9;border-radius:14px;padding:16px 18px;}
.kpi .label {font-size:11px;color:#708590;text-transform:uppercase;letter-spacing:.08em;}
.kpi .value {font-size:27px;font-weight:700;color:#081927;margin-top:3px;}
.kpi .note {font-size:11px;color:#708590;margin-top:2px;}
.section {font-size:20px;font-weight:700;color:#081927;margin:6px 0 10px;}
.callout {background:#eef5f7;border-left:5px solid #185b74;border-radius:8px;padding:12px 15px;margin-top:10px;}
.small {font-size:11px;color:#708590;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATOS Y MODELO
# ============================================================
@st.cache_data
def load_data():
    # Mantiene la estructura del proyecto original: /data/archivo.csv
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "synthetic_datacenter_hourly_3days.csv"
    if not path.exists():
        # Permite ejecutar este archivo directamente desde /mnt/data para pruebas.
        candidates = [
            Path(__file__).resolve().parent / "synthetic_datacenter_hourly_3days(4).csv",
            Path(__file__).resolve().parent / "synthetic_datacenter_hourly_3days.csv",
        ]
        path = next((p for p in candidates if p.exists()), path)
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
        "it_load_kW", "hvac_kW", "it_load_kW_lag1",
        "total_consumption_kW_lag1", "it_load_kW_roll3",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
        "temperature_C"
    ]

    d = d.dropna(subset=features + ["total_consumption_kW"])
    n = int(len(d) * 0.70)
    train = d.iloc[:n]
    test = d.iloc[n:]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[features], train["total_consumption_kW"])
    pred = model.predict(test[features])
    return model, features, train, test, pred


df = load_data()
model, features, train, test, pred = train_model(df)
real = test["total_consumption_kW"].values
mae = mean_absolute_error(real, pred)
rmse = np.sqrt(mean_squared_error(real, pred))
r2 = r2_score(real, pred)

# ============================================================
# HILO DE LA CHARLA: OBSERVACIÓN 21
# ============================================================
# El conjunto de prueba tiene 21 observaciones; la observación 21
# corresponde a la última fila del conjunto de prueba.
OBS_TEST_IDX = len(test) - 1
OBS_NUMBER = len(test)
base_row = test.iloc[OBS_TEST_IDX][features].copy()
base_real = float(test.iloc[OBS_TEST_IDX]["total_consumption_kW"])
base_prediction = float(pred[OBS_TEST_IDX])

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## CONTROL DE LA DEMO")
    st.markdown("**Modelo**  \nRandom Forest")
    st.markdown("**Datos**  \nData Center sintético")
    st.markdown("**Entrenamiento / prueba**  \n70 % / 30 %")
    st.divider()
    st.markdown("### Hilo de demostración")
    st.markdown(
        f"**Observación {OBS_NUMBER}**  \n"
        f"Real: **{base_real:.2f} kW**  \n"
        f"Predicho: **{base_prediction:.2f} kW**"
    )
    st.divider()
    st.caption("Desempeño → Predicción → XAI → Escenarios")
    st.divider()
    st.caption("Los datos y escenarios son demostrativos y no representan ahorro validado de Oil Malal.")

# ============================================================
# CABECERA
# ============================================================
st.markdown("""
<div class="hero">
<h1>⚡ PGF · Gestión Energética Inteligente</h1>
<p>De medir → entender → predecir → explicar → explorar → decidir</p>
<p style="margin-top:7px;font-size:12px;">
Demostración ML · datos sintéticos del notebook Procesoinicialpgfenergia2 · USM
</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "01 · DESEMPEÑO",
    "02 · PREDICCIÓN",
    "03 · XAI / SHAP",
    "04 · ESCENARIOS"
])

# ============================================================
# 01 DESEMPEÑO
# ============================================================
with tabs[0]:
    st.markdown('<div class="section">01 · Desempeño energético</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,label,value,note in [
        (c1,"OBSERVACIONES",len(df),"dataset completo"),
        (c2,"ENTRENAMIENTO",len(train),"70 %"),
        (c3,"PRUEBA",len(test),"30 %"),
        (c4,"CONSUMO MEDIO",f"{df['total_consumption_kW'].mean():.1f} kW","período completo"),
    ]:
        col.markdown(
            f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="note">{note}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("#### Selección temporal")
    min_d, max_d = df["timestamp"].min().date(), df["timestamp"].max().date()
    selected = st.date_input("Período", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(selected, tuple) and len(selected) == 2:
        fdf = df[(df["timestamp"].dt.date >= selected[0]) & (df["timestamp"].dt.date <= selected[1])]
    else:
        fdf = df.copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fdf["timestamp"], y=fdf["total_consumption_kW"],
        mode="lines+markers", name="Consumo total",
        hovertemplate="%{x|%d-%m %H:%M}<br>%{y:.1f} kW<extra></extra>"
    ))
    fig.update_layout(height=390, margin=dict(l=20,r=20,t=15,b=20),
                      xaxis_title="Tiempo", yaxis_title="Consumo [kW]",
                      hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": True})
    st.markdown('<div class="callout"><b>Lectura:</b> esta vista responde primero “¿qué está pasando?” antes de pasar a la predicción.</div>', unsafe_allow_html=True)

# ============================================================
# 02 PREDICCIÓN
# ============================================================
with tabs[1]:
    st.markdown('<div class="section">02 · Predicción · Real vs. Predicho</div>', unsafe_allow_html=True)
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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test.index, y=real, mode="lines+markers", name="Real",
                             hovertemplate="%{x|%d-%m %H:%M}<br>Real: %{y:.1f} kW<extra></extra>"))
    fig.add_trace(go.Scatter(x=test.index, y=pred, mode="lines+markers", name="Predicho",
                             hovertemplate="%{x|%d-%m %H:%M}<br>Predicho: %{y:.1f} kW<extra></extra>"))
    fig.update_layout(height=400, margin=dict(l=20,r=20,t=15,b=20),
                      xaxis_title="Hora", yaxis_title="Consumo [kW]",
                      hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": True})

    residual = real - pred
    fig2 = go.Figure()
    fig2.add_hline(y=0, line_width=1)
    fig2.add_trace(go.Scatter(x=test.index, y=residual, mode="lines+markers", name="Error",
                               hovertemplate="%{x|%d-%m %H:%M}<br>Error: %{y:.1f} kW<extra></extra>"))
    fig2.update_layout(height=260, margin=dict(l=20,r=20,t=15,b=20),
                       xaxis_title="Hora", yaxis_title="Error [kW]", template="plotly_white")
    st.plotly_chart(fig2, width="stretch", config={"displayModeBar": True})

    st.markdown(
        f'<div class="callout"><b>Observación {OBS_NUMBER}:</b> Real = {base_real:.2f} kW · Predicho = {base_prediction:.2f} kW.</div>',
        unsafe_allow_html=True
    )

# ============================================================
# 03 XAI / SHAP
# ============================================================
with tabs[2]:
    st.markdown('<div class="section">03 · XAI · ¿Por qué predijo eso?</div>', unsafe_allow_html=True)
    st.caption("Dos niveles de explicación: qué domina el modelo y qué ocurrió en una observación concreta.")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test[features])
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=features).sort_values(ascending=False)

    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown("#### 1 · Visión global")
        top = importance.head(8).sort_values()
        fig = go.Figure(go.Bar(
            orientation="h", x=top.values,
            y=[x.replace("_", " ") for x in top.index],
            text=[f"{v:.2f}" for v in top.values], textposition="outside",
            hovertemplate="%{y}<br>Impacto medio absoluto: %{x:.2f}<extra></extra>"
        ))
        fig.update_layout(height=410, margin=dict(l=10,r=90,t=10,b=35),
                          xaxis_title="Impacto medio absoluto SHAP", yaxis_title="", template="plotly_white")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": True})
        dominant = importance.index[0].replace("_", " ")
        st.markdown(f'<div class="callout"><b>Variable dominante:</b> {dominant}. La importancia SHAP indica influencia en las predicciones del modelo, no causalidad física.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("#### 2 · Observación concreta")
        # El valor por defecto es la Observación 21, para mantener el hilo de la charla.
        idx = st.slider("Observación de prueba", 1, len(test), OBS_NUMBER, key="shap_obs")
        row = test.iloc[idx-1]
        local = pd.Series(shap_values[idx-1], index=features)
        local_top = local.reindex(local.abs().sort_values(ascending=False).head(6).index)
        base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0]) if np.asarray(explainer.expected_value).size else 0.0
        pred_value = float(pred[idx-1])
        contrib_sum = float(local.sum())
        fig2 = go.Figure(go.Bar(
            orientation="h", x=local_top.values[::-1],
            y=[x.replace("_", " ") for x in local_top.index[::-1]],
            text=[f"{v:+.2f}" for v in local_top.values[::-1]], textposition="outside",
            hovertemplate="%{y}<br>Contribución SHAP: %{x:+.2f}<extra></extra>"
        ))
        fig2.add_vline(x=0, line_width=1)
        fig2.update_layout(height=410, margin=dict(l=10,r=90,t=10,b=35),
                           xaxis_title="Contribución SHAP", yaxis_title="", template="plotly_white")
        st.plotly_chart(fig2, width="stretch", config={"displayModeBar": True})
        m1,m2=st.columns(2)
        m1.metric("Real", f"{row['total_consumption_kW']:.2f} kW")
        m2.metric("Predicho", f"{pred_value:.2f} kW")
        st.caption(f"Valor base del modelo: {base_value:.2f} kW · suma SHAP: {contrib_sum:+.2f} kW")

    st.markdown('<div class="callout"><b>Lectura para la charla:</b> primero identificamos qué variables pesan en el modelo; después seleccionamos una observación y vemos qué variables empujaron esa predicción concreta.</div>', unsafe_allow_html=True)

# ============================================================
# 04 ESCENARIOS — OPCIÓN B: MISMO RANDOM FOREST
# ============================================================
with tabs[3]:
    st.markdown('<div class="section">04 · Escenarios · explorar antes de actuar</div>', unsafe_allow_html=True)
    st.caption(f"Misma observación {OBS_NUMBER} · mismas variables derivadas · mismo Random Forest. Solo modificamos las condiciones seleccionadas.")

    # --------------------------------------------------------
    # REFERENCIA: exactamente la misma observación usada en
    # predicción y XAI.
    # --------------------------------------------------------
    st.markdown(f"#### 1 · Condición de referencia · Observación {OBS_NUMBER}")
    b1,b2,b3,b4 = st.columns(4)
    b1.metric("IT Load base", f"{float(base_row['it_load_kW']):.2f} kW")
    b2.metric("HVAC base", f"{float(base_row['hvac_kW']):.2f} kW")
    b3.metric("Real", f"{base_real:.2f} kW")
    b4.metric("Predicción base", f"{base_prediction:.2f} kW")

    st.markdown(
        '<div class="callout"><b>Regla del escenario:</b> se mantiene todo el contexto de la Observación 21 y se modifica únicamente IT Load y/o HVAC. La nueva predicción la calcula el mismo Random Forest entrenado arriba.</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SELECCIÓN
    # --------------------------------------------------------
    st.markdown("#### 2 · Selección del escenario")
    preset = st.selectbox(
        "Modo de exploración",
        [
            "Manual",
            "Reducir HVAC 10 %",
            "Reducir IT Load 10 %",
            "Aumentar IT Load 10 %",
            "Aumentar HVAC 10 %",
        ],
        index=0,
        key="scenario_preset",
    )

    # En modo predefinido no dependemos del estado anterior de los sliders.
    if preset == "Manual":
        c1,c2 = st.columns(2)
        with c1:
            it = st.slider(
                "IT Load [kW]",
                min_value=float(test["it_load_kW"].min()),
                max_value=float(test["it_load_kW"].max()),
                value=float(base_row["it_load_kW"]),
                step=0.1,
                key="scenario_it_manual",
            )
        with c2:
            hv = st.slider(
                "HVAC [kW]",
                min_value=float(test["hvac_kW"].min()),
                max_value=float(test["hvac_kW"].max()),
                value=float(base_row["hvac_kW"]),
                step=0.1,
                key="scenario_hv_manual",
            )
    else:
        it = float(base_row["it_load_kW"])
        hv = float(base_row["hvac_kW"])
        if preset == "Reducir IT Load 10 %":
            it *= 0.90
        elif preset == "Aumentar IT Load 10 %":
            it *= 1.10
        elif preset == "Reducir HVAC 10 %":
            hv *= 0.90
        elif preset == "Aumentar HVAC 10 %":
            hv *= 1.10

        c1,c2 = st.columns(2)
        c1.metric("IT Load escenario", f"{it:.2f} kW", f"{(it/float(base_row['it_load_kW'])-1)*100:+.1f} %")
        c2.metric("HVAC escenario", f"{hv:.2f} kW", f"{(hv/float(base_row['hvac_kW'])-1)*100:+.1f} %")

    # --------------------------------------------------------
    # CONSTRUCCIÓN DEL ESCENARIO
    # --------------------------------------------------------
    scenario = base_row.copy()
    scenario["it_load_kW"] = it
    scenario["hvac_kW"] = hv

    # IMPORTANTE: aquí se ejecuta el MISMO modelo entrenado.
    baseline = float(model.predict(pd.DataFrame([base_row], columns=features))[0])
    scenario_y = float(model.predict(pd.DataFrame([scenario], columns=features))[0])
    delta = scenario_y - baseline
    pct = (delta / baseline * 100) if baseline else 0.0

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------
    st.markdown("#### 3 · Resultado calculado por el Random Forest")
    k1,k2,k3,k4 = st.columns(4)
    for col,label,value,note in [
        (k1,"BASE",f"{baseline:.2f} kW","referencia · mismo modelo"),
        (k2,"ESCENARIO",f"{scenario_y:.2f} kW","nueva predicción"),
        (k3,"Δ",f"{delta:+.2f} kW","escenario − base"),
        (k4,"VARIACIÓN",f"{pct:+.2f} %","respecto de la base"),
    ]:
        col.markdown(
            f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>',
            unsafe_allow_html=True
        )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Base"], y=[baseline], name="Base",
        text=[f"{baseline:.2f} kW"], textposition="outside",
        hovertemplate="Base<br>%{y:.2f} kW<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=["Escenario"], y=[scenario_y], name="Escenario",
        text=[f"{scenario_y:.2f} kW"], textposition="outside",
        hovertemplate="Escenario<br>%{y:.2f} kW<extra></extra>"
    ))
    fig.update_layout(
        height=360,
        margin=dict(l=20,r=20,t=45,b=20),
        yaxis_title="Consumo predicho [kW]",
        template="plotly_white",
        showlegend=False,
    )
    fig.add_annotation(
        x=0.5,
        y=max(baseline, scenario_y),
        text=f"Δ = {delta:+.2f} kW",
        showarrow=True,
        arrowhead=2,
        yshift=18,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": True})

    if abs(delta) < 0.005:
        msg = "El escenario coincide con la condición de referencia."
    elif delta < 0:
        msg = f"El escenario reduce la predicción en {abs(delta):.2f} kW ({abs(pct):.2f} %) respecto de la base."
    else:
        msg = f"El escenario aumenta la predicción en {delta:.2f} kW ({pct:.2f} %) respecto de la base."

    st.markdown(f'<div class="callout"><b>Resultado:</b> {msg}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="callout"><b>Alcance para la charla:</b> esto es una simulación del modelo con datos sintéticos. No afirmamos “ahorro” físico ni sustituimos una evaluación energética, operacional o económica real.</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.caption("PGF · Paul Gálvez · USM · Demo técnica · Opción B: escenarios calculados por el mismo Random Forest")
