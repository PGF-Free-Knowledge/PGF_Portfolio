# -*- coding: utf-8 -*-
"""
Dashboard: De medir a predecir — IA para el desempeño energético en
infraestructuras críticas.

Flujo: Planta -> Datos -> Desempeño -> ML -> Predicción -> XAI/SHAP ->
Escenarios -> Decisión.

Cómo correrlo:
    pip install -r requirements.txt
    streamlit run dashboard_energia.py

Datasets incluidos (junto a este script):
  - synthetic_datacenter_hourly_3days.csv  -> dataset piloto REAL usado para
    desarrollar el pipeline de ML/SHAP del proyecto (Proyecto_ML_Energia_PGF).
  - synthetic_tower_hourly_3days.csv       -> dataset piloto alternativo
    (torre de telecomunicaciones). OJO: en este dataset el consumo total es
    exactamente la suma de dos variables ya incluidas como features, por lo
    que un modelo lineal ajusta perfecto (R²≈1) sin que eso implique una
    capacidad predictiva real — se deja visible como caso de estudio, no
    como el resultado principal.
  - Si ninguno de los dos CSV está presente, o el usuario elige "Planta
    industrial (demo)", se usa un generador sintético in-memory, claramente
    etiquetado como DEMO.

Autor: Paul Gálvez — apoyo de construcción: Claude
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import shap

st.set_page_config(
    page_title="Desempeño Energético Inteligente",
    page_icon="⚡",
    layout="wide",
)

RANDOM_STATE = 42
COSTO_KWH = 0.15  # USD/kWh, solo referencial para el bloque de decisión
HERE = os.path.dirname(os.path.abspath(__file__))


# =======================================================================
# 1. DATASETS — perfiles (real: Data Center / Torre, y demo sintético)
# =======================================================================

def _add_time_features(df, load_col, target_col):
    """Agrega lag1, media móvil 3h y codificación cíclica hora/día —
    exactamente el mismo esquema de ingeniería de variables usado en el
    notebook Procesoinicialpgfenergia2.ipynb."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday

    df[f"{load_col}_lag1"] = df[load_col].shift(1)
    df[f"{target_col}_lag1"] = df[target_col].shift(1)
    df[f"{load_col}_roll3"] = df[load_col].rolling(window=3).mean()

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    return df


@st.cache_data
def generar_datos_sinteticos(n_dias: int = 90, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Dataset horario SINTÉTICO de una planta industrial / Data Center
    genérico, para el modo demo (no proviene de mediciones reales)."""
    rng_state = np.random.default_rng(seed)
    n = 24 * n_dias
    fechas = pd.date_range("2025-01-01", periods=n, freq="h")
    hora = fechas.hour
    dia_anio = fechas.dayofyear

    carga_ti = 40 + 15 * np.sin(2 * np.pi * (hora - 9) / 24) + rng_state.normal(0, 3, n)
    carga_ti = np.clip(carga_ti, 10, 95)

    temp_ext = (
        18
        + 8 * np.sin(2 * np.pi * (dia_anio - 200) / 365)
        + 3 * np.sin(2 * np.pi * (hora - 14) / 24)
        + rng_state.normal(0, 1.5, n)
    )

    ocupacion = np.where((hora >= 8) & (hora <= 19), 1.0, 0.3) + rng_state.normal(0, 0.05, n)
    ocupacion = np.clip(ocupacion, 0, 1.2)

    setpoint_enfriamiento = np.clip(21 + rng_state.normal(0, 1.5, n), 18, 24)
    setpoint_optimo = 20 + 0.25 * (temp_ext - 18)
    penalidad_setpoint = 3.0 * (setpoint_enfriamiento - setpoint_optimo) ** 2

    ruido = rng_state.normal(0, 4, n)
    energia_kwh = carga_ti * 8.5 + ocupacion * 10 + penalidad_setpoint + ruido

    return pd.DataFrame(
        {
            "timestamp": fechas,
            "carga_ti_pct": carga_ti,
            "temp_exterior_c": temp_ext,
            "ocupacion_idx": ocupacion,
            "setpoint_enfriamiento": setpoint_enfriamiento,
            "energia_kwh": energia_kwh,
        }
    )


@st.cache_data
def cargar_csv_real(filename):
    path = os.path.join(HERE, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# Perfiles de dataset: cada uno define su columna objetivo, su variable de
# carga principal (para LBEn/CUSUM) y los sliders del simulador de escenarios.
DATASETS = {
    "datacenter": {
        "label": "Data Center — dataset piloto real (Proyecto_ML_Energia_PGF)",
        "file": "synthetic_datacenter_hourly_3days.csv",
        "target": "total_consumption_kW",
        "load_col": "it_load_kW",
        "unit": "kW",
        "base_features": ["it_load_kW", "hvac_kW"],
        "extra_features_end": ["temperature_C"],
        "scenario_specs": [
            {"col": "it_load_kW", "label": "Carga TI (kW)", "min": 150, "max": 450},
            {"col": "hvac_kW", "label": "Carga HVAC (kW)", "min": 80, "max": 220},
            {"col": "temperature_C", "label": "Temperatura exterior (°C)", "min": 15, "max": 32},
        ],
        "note": None,
        "is_demo": False,
    },
    "torre": {
        "label": "Torre de telecomunicaciones — dataset piloto real",
        "file": "synthetic_tower_hourly_3days.csv",
        "target": "total_consumption_kW",
        "load_col": "radio_load_kW",
        "unit": "kW",
        "base_features": ["radio_load_kW", "battery_soc_%", "generator_kW"],
        "extra_features_end": ["ambient_temp_C"],
        "scenario_specs": [
            {"col": "radio_load_kW", "label": "Carga de radio (kW)", "min": 4, "max": 16},
            {"col": "generator_kW", "label": "Generador (kW)", "min": 0, "max": 5},
            {"col": "battery_soc_%", "label": "Batería (SOC %)", "min": 55, "max": 100},
            {"col": "ambient_temp_C", "label": "Temperatura ambiente (°C)", "min": 8, "max": 36},
        ],
        "note": (
            "⚠️ En este dataset, `total_consumption_kW` es exactamente la suma de "
            "`radio_load_kW + generator_kW`. Un modelo lineal ajusta perfecto (R²≈1) "
            "porque es una identidad algebraica, no una relación aprendida — se deja "
            "visible como caso de estudio, no como resultado destacado."
        ),
        "is_demo": False,
    },
    "demo": {
        "label": "Planta industrial — datos sintéticos (demo)",
        "file": None,
        "target": "energia_kwh",
        "load_col": "carga_ti_pct",
        "unit": "kWh",
        "base_features": ["carga_ti_pct", "temp_exterior_c", "ocupacion_idx", "setpoint_enfriamiento"],
        "extra_features_end": [],
        "scenario_specs": [
            {"col": "carga_ti_pct", "label": "Carga TI (%)", "min": 10, "max": 95},
            {"col": "temp_exterior_c", "label": "Temperatura exterior (°C)", "min": 0, "max": 35},
            {"col": "ocupacion_idx", "label": "Índice de ocupación", "min": 0.0, "max": 1.2},
            {"col": "setpoint_enfriamiento", "label": "Setpoint de enfriamiento (°C)", "min": 18, "max": 24},
        ],
        "note": None,
        "is_demo": True,
    },
}


def preparar_dataset(profile_key, archivo_subido=None):
    """Devuelve (df_modelo, FEATURES, TARGET, df_crudo, es_sintetico)."""
    profile = DATASETS[profile_key]
    target = profile["target"]
    load_col = profile["load_col"]

    if archivo_subido is not None:
        df_raw = pd.read_csv(archivo_subido)
        es_sintetico = False
    elif profile["is_demo"]:
        df_raw = generar_datos_sinteticos()
        es_sintetico = True
    else:
        df_raw = cargar_csv_real(profile["file"])
        if df_raw is None:
            st.warning(
                f"No se encontró {profile['file']} junto al script — usando datos "
                "sintéticos de respaldo con la misma estructura."
            )
            df_raw = generar_datos_sinteticos()
            es_sintetico = True
        else:
            es_sintetico = False

    if profile["is_demo"] and "timestamp" in df_raw.columns and load_col == "carga_ti_pct":
        # dataset demo: usa directamente sus columnas, sin ingeniería temporal adicional
        df_model = df_raw.dropna(subset=profile["base_features"] + [target]).copy()
        features = profile["base_features"]
    else:
        df_time = _add_time_features(df_raw, load_col, target)
        features = profile["base_features"] + [
            f"{load_col}_lag1",
            f"{target}_lag1",
            f"{load_col}_roll3",
            "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
        ] + profile.get("extra_features_end", [])
        df_model = df_time.dropna(subset=features + [target]).copy()

    return df_model, features, target, df_raw, es_sintetico


# =======================================================================
# 2. MODELOS — Ridge + Random Forest (mismo esquema que el notebook)
# =======================================================================
@st.cache_resource
def entrenar_modelos(df, features, target):
    X = df[features]
    y = df[target]
    n_train = int(len(df) * 0.7)
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    metrics = {
        "ridge": {
            "r2": r2_score(y_test, y_pred_ridge),
            "mae": mean_absolute_error(y_test, y_pred_ridge),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        },
        "rf": {
            "r2": r2_score(y_test, y_pred_rf),
            "mae": mean_absolute_error(y_test, y_pred_rf),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        },
    }
    return {
        "ridge": ridge, "rf": rf,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred_ridge": y_pred_ridge, "y_pred_rf": y_pred_rf,
        "metrics": metrics,
    }


@st.cache_resource
def calcular_shap(_modelo_rf, X_muestra: pd.DataFrame):
    explainer = shap.TreeExplainer(_modelo_rf)
    shap_values = explainer(X_muestra)
    return explainer, shap_values


# =======================================================================
# SIDEBAR
# =======================================================================
st.sidebar.title("⚡ Configuración")
st.sidebar.caption(
    "Desarrollo de modelos y arquitecturas inteligentes para la gestión, "
    "predicción, explicación y optimización del desempeño energético en "
    "infraestructuras críticas."
)

profile_key = st.sidebar.selectbox(
    "Dataset",
    options=list(DATASETS.keys()),
    format_func=lambda k: DATASETS[k]["label"],
    index=0,
)
profile = DATASETS[profile_key]

archivo = st.sidebar.file_uploader(
    "O cargar tu propio CSV (mismas columnas que el dataset elegido)",
    type=["csv"],
)

df, FEATURES, TARGET, df_raw, es_sintetico = preparar_dataset(profile_key, archivo)
LOAD_COL = profile["load_col"]
UNIT = profile["unit"]

if profile["note"]:
    st.sidebar.info(profile["note"])
if es_sintetico:
    st.sidebar.warning(
        "Modo DEMO: datos sintéticos de referencia. Ninguna métrica de este "
        "modo representa una planta real."
    )
else:
    st.sidebar.success(f"Dataset piloto real cargado ({len(df_raw)} registros horarios).")

modelos = entrenar_modelos(df, FEATURES, TARGET)
metrics = modelos["metrics"]

st.sidebar.divider()
st.sidebar.metric("Registros (modelo)", f"{len(df):,}")
st.sidebar.metric("R² Ridge", f"{metrics['ridge']['r2']:.3f}")
st.sidebar.metric("R² Random Forest", f"{metrics['rf']['r2']:.3f}")


# =======================================================================
# TÍTULO
# =======================================================================
st.title("⚡ De medir a predecir")
st.subheader(
    "Gestión, predicción, explicación y optimización del desempeño "
    "energético en infraestructuras críticas"
)

tabs = st.tabs(
    ["🏭 Planta", "📊 Desempeño", "🤖 Predicción ML", "🔍 XAI / SHAP", "🎛️ Escenarios", "✅ Decisión"]
)

# ---------------------------------------------------------------------
# TAB 1 — PLANTA / ESTADO GENERAL
# ---------------------------------------------------------------------
with tabs[0]:
    st.markdown("### Estado general de la instalación")
    ultimo = df.iloc[-1]
    cols = st.columns(len(profile["base_features"]) + 1)
    for i, feat in enumerate(profile["base_features"]):
        cols[i].metric(feat, f"{ultimo[feat]:.1f}")
    cols[-1].metric(f"{TARGET}", f"{ultimo[TARGET]:.1f} {UNIT}")

    st.line_chart(df.set_index("timestamp")[[TARGET]])
    st.caption(f"Consumo energético ({TARGET}) — serie completa del dataset cargado.")

# ---------------------------------------------------------------------
# TAB 2 — DESEMPEÑO ENERGÉTICO (LBEn / CUSUM)
# ---------------------------------------------------------------------
with tabs[1]:
    st.markdown("### Línea base energética (LBEn) y desviación acumulada (CUSUM)")
    st.caption(
        f"Línea base: regresión simple de {TARGET} sobre {LOAD_COL} (lógica ISO 50001). "
        "El CUSUM evidencia sobreconsumo o ahorro sostenido en el tiempo."
    )

    coef = np.polyfit(df[LOAD_COL], df[TARGET], 1)
    baseline_pred = np.polyval(coef, df[LOAD_COL])
    desviacion = df[TARGET].values - baseline_pred
    cusum = np.cumsum(desviacion)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(df["timestamp"], df[TARGET], label="Real", lw=1)
    ax1.plot(df["timestamp"], baseline_pred, label="Línea base (LBEn)", lw=1)
    ax1.set_ylabel(UNIT)
    ax1.legend()
    ax1.set_title("Consumo real vs. línea base")

    ax2.plot(df["timestamp"], cusum, color="darkorange")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_ylabel(f"CUSUM ({UNIT})")
    ax2.set_title("Desviación acumulada")
    fig.autofmt_xdate()
    st.pyplot(fig)

# ---------------------------------------------------------------------
# TAB 3 — PREDICCIÓN ML
# ---------------------------------------------------------------------
with tabs[2]:
    st.markdown("### Modelos predictivos — Ridge vs. Random Forest")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ridge (regularizado)**")
        st.metric("R²", f"{metrics['ridge']['r2']:.3f}")
        st.metric("MAE", f"{metrics['ridge']['mae']:.2f} {UNIT}")
        st.metric("RMSE", f"{metrics['ridge']['rmse']:.2f} {UNIT}")
    with c2:
        st.markdown("**Random Forest (usado para XAI/SHAP)**")
        st.metric("R²", f"{metrics['rf']['r2']:.3f}")
        st.metric("MAE", f"{metrics['rf']['mae']:.2f} {UNIT}")
        st.metric("RMSE", f"{metrics['rf']['rmse']:.2f} {UNIT}")

    modelo_grafico = st.radio("Ver predicción de:", ["Random Forest", "Ridge"], horizontal=True)
    y_test = modelos["y_test"]
    y_pred = modelos["y_pred_rf"] if modelo_grafico == "Random Forest" else modelos["y_pred_ridge"]

    fig2, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(y_test.values, label="Real", color="black", linewidth=2)
    ax.plot(y_pred, label=f"Predicción ({modelo_grafico})", linestyle="--")
    ax.set_xlabel("Tiempo (horas, set de validación)")
    ax.set_ylabel(f"{TARGET} ({UNIT})")
    ax.set_title(f"Comparación predicción vs. real — {modelo_grafico}")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig2)

# ---------------------------------------------------------------------
# TAB 4 — XAI / SHAP
# ---------------------------------------------------------------------
with tabs[3]:
    st.markdown("### Interpretabilidad del modelo (SHAP sobre Random Forest)")
    st.caption(
        "SHAP explica cuánto aporta cada variable a la predicción del modelo: "
        "vista global (importancia general) y vista local (una observación específica)."
    )

    X_test = modelos["X_test"]
    X_muestra = X_test if len(X_test) <= 300 else X_test.sample(300, random_state=RANDOM_STATE)
    explainer, shap_values = calcular_shap(modelos["rf"], X_muestra)

    st.markdown("**Vista global — importancia de variables**")
    fig3 = plt.figure()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig3)
    plt.close(fig3)

    st.markdown("**Vista global — dispersión de impacto (beeswarm)**")
    fig4 = plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown("**Vista local — explicación de un caso puntual**")
    idx_local = st.slider("Selecciona un registro de la muestra", 0, len(X_muestra) - 1, 0)
    fig5 = plt.figure()
    shap.plots.waterfall(shap_values[idx_local], show=False)
    st.pyplot(fig5)
    plt.close(fig5)

    st.caption("Una contribución SHAP no debe interpretarse como una relación causal.")

# ---------------------------------------------------------------------
# TAB 5 — ESCENARIOS (simulación what-if)
# ---------------------------------------------------------------------
with tabs[4]:
    st.markdown("### Simulación de escenarios (what-if)")
    st.caption(
        "Ajusta las variables operativas y observa el impacto proyectado en el "
        "consumo energético, según el modelo Random Forest entrenado."
    )

    ultimo = df.iloc[-1]
    valores_escenario = {}
    cols = st.columns(2)
    for i, spec in enumerate(profile["scenario_specs"]):
        with cols[i % 2]:
            valores_escenario[spec["col"]] = st.slider(
                spec["label"], float(spec["min"]), float(spec["max"]),
                float(np.clip(ultimo[spec["col"]], spec["min"], spec["max"])),
            )
    hora_sim = st.slider("Hora del día", 0, 23, int(ultimo["hour"]) if "hour" in df.columns else 12)

    fila = {}
    for feat in FEATURES:
        if feat in valores_escenario:
            fila[feat] = valores_escenario[feat]
        elif feat == "hour_sin":
            fila[feat] = np.sin(2 * np.pi * hora_sim / 24)
        elif feat == "hour_cos":
            fila[feat] = np.cos(2 * np.pi * hora_sim / 24)
        else:
            fila[feat] = ultimo[feat]  # lag/roll/weekday: se fijan al último valor observado

    X_escenario = pd.DataFrame([fila])[FEATURES]
    pred_escenario = modelos["rf"].predict(X_escenario)[0]
    pred_promedio = modelos["rf"].predict(df[FEATURES]).mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Consumo estimado (escenario)", f"{pred_escenario:.1f} {UNIT}")
    c2.metric("Consumo promedio histórico", f"{pred_promedio:.1f} {UNIT}")
    c3.metric("Diferencia vs. promedio", f"{pred_escenario - pred_promedio:+.1f} {UNIT}")

    st.caption(
        "Las variables de rezago (hora anterior) y media móvil se fijan al último "
        "valor observado — este simulador mueve las condiciones instantáneas."
    )

# ---------------------------------------------------------------------
# TAB 6 — DECISIÓN (comparación de estrategias)
# ---------------------------------------------------------------------
with tabs[5]:
    st.markdown("### Comparación de estrategias — reactiva (ISO 50001) vs. predictiva (IA)")

    if profile["is_demo"]:
        st.caption(
            "Estrategia ISO 50001: setpoint fijo. Estrategia IA: setpoint óptimo "
            "sugerido por el modelo para cada hora, dentro de un rango operacional seguro."
        )
        dias_sim = st.slider("Días a simular", 7, min(30, int(len(df) / 24)), min(30, int(len(df) / 24)))
        bloque = df.tail(24 * dias_sim).copy()

        setpoint_fijo = df["setpoint_enfriamiento"].median()
        X_iso = bloque[FEATURES].copy()
        X_iso["setpoint_enfriamiento"] = setpoint_fijo
        consumo_iso = modelos["rf"].predict(X_iso).sum()

        grilla = np.linspace(18, 24, 13)
        mejor_consumo = np.full(len(bloque), np.inf)
        for sp in grilla:
            X_ia = bloque[FEATURES].copy()
            X_ia["setpoint_enfriamiento"] = sp
            pred = modelos["rf"].predict(X_ia)
            mejor_consumo = np.minimum(mejor_consumo, pred)
        consumo_ia = mejor_consumo.sum()

        ahorro_kwh = consumo_iso - consumo_ia
        ahorro_pct = 100 * ahorro_kwh / consumo_iso
        etiqueta_periodo = f"{dias_sim} días simulados"
    else:
        st.caption(
            "Estrategia reactiva: se asume que la operación reacciona con retardo, "
            "corrigiendo apenas un 1% sobre el valor de la hora anterior (proxy de "
            "gestión manual tipo ISO 50001). Estrategia predictiva: se corrige un 3% "
            "de forma proactiva sobre la predicción del modelo. Ambos supuestos son "
            "ilustrativos — no provienen de una medición de planta — y se aplican "
            "sobre el mismo conjunto de validación para que la comparación sea justa."
        )
        y_test = modelos["y_test"]
        lag_col = f"{TARGET}_lag1"
        baseline_reactivo = modelos["X_test"][lag_col].values * 0.99
        pred_predictivo = modelos["y_pred_rf"] * 0.97

        consumo_real = y_test.sum()
        consumo_iso = baseline_reactivo.sum()
        consumo_ia = pred_predictivo.sum()
        ahorro_kwh = consumo_iso - consumo_ia
        ahorro_pct = 100 * ahorro_kwh / consumo_iso
        etiqueta_periodo = f"{len(y_test)} horas de validación"

    c1, c2, c3 = st.columns(3)
    c1.metric("Consumo — reactiva (ISO 50001)", f"{consumo_iso:,.0f} {UNIT}")
    c2.metric("Consumo — predictiva (IA)", f"{consumo_ia:,.0f} {UNIT}")
    c3.metric("Ahorro estimado", f"{ahorro_pct:.1f} %", f"-{ahorro_kwh:,.0f} {UNIT}")

    ahorro_costo = ahorro_kwh * COSTO_KWH
    st.info(
        f"En {etiqueta_periodo}, la estrategia predictiva sugiere un ahorro "
        f"aproximado de **{ahorro_kwh:,.0f} {UNIT}** "
        f"(~USD {ahorro_costo:,.0f} a costo referencial de {COSTO_KWH} USD/kWh)."
    )

    if es_sintetico or not profile["is_demo"]:
        st.warning(
            "Los supuestos de corrección (1% reactivo / 3% predictivo, o el setpoint "
            "óptimo en modo demo) son ilustrativos. Antes de usar esta cifra en la "
            "charla, valida el porcentaje de mejora contra datos reales de operación."
        )
