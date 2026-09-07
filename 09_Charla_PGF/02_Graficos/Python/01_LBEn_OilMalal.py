import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# D6 - LINEA BASE ENERGETICA OIL MALAL
# Grafico para presentacion "De medir a predecir"
# ============================================================

# ------------------------------------------------------------
# 1. RUTAS
# ------------------------------------------------------------

carpeta_proyecto = Path(__file__).resolve().parents[2]

archivo = carpeta_proyecto / "Prevaluación_Paper_TGII_v60_pgf.xlsx"
hoja = "Consumo Eléctrico"

carpeta_salida = carpeta_proyecto / "02_Graficos" / "Finales"
carpeta_salida.mkdir(parents=True, exist_ok=True)

archivo_salida = carpeta_salida / "LBEn_OilMalal_D6.png"

# ------------------------------------------------------------
# 2. LECTURA DEL EXCEL
# ------------------------------------------------------------

df = pd.read_excel(
    archivo,
    sheet_name=hoja,
    header=None
)

# ------------------------------------------------------------
# 3. DATOS DEL MODELO FILTRADO
# ------------------------------------------------------------

# Producción [m³/mes]  -> columna Y del Excel
# Consumo [kWh/mes]    -> columna Z del Excel

datos = df.iloc[105:128, [24, 25]].copy()

datos.columns = [
    "Produccion_m3_mes",
    "Consumo_kWh_mes"
]

datos["Produccion_m3_mes"] = pd.to_numeric(
    datos["Produccion_m3_mes"],
    errors="coerce"
)

datos["Consumo_kWh_mes"] = pd.to_numeric(
    datos["Consumo_kWh_mes"],
    errors="coerce"
)

datos = datos.dropna()

# ------------------------------------------------------------
# 4. VARIABLES
# ------------------------------------------------------------

X = datos["Produccion_m3_mes"].to_numpy()
Y = datos["Consumo_kWh_mes"].to_numpy()

# ------------------------------------------------------------
# 5. REGRESION LINEAL
# ------------------------------------------------------------

pendiente, intercepto = np.polyfit(X, Y, 1)

Y_pred = pendiente * X + intercepto

# ------------------------------------------------------------
# 6. R²
# ------------------------------------------------------------

ss_res = np.sum((Y - Y_pred) ** 2)
ss_tot = np.sum((Y - np.mean(Y)) ** 2)

r2 = 1 - (ss_res / ss_tot)

# ------------------------------------------------------------
# 7. LINEA BASE
# ------------------------------------------------------------

x_linea = np.linspace(
    X.min(),
    X.max(),
    200
)

y_linea = pendiente * x_linea + intercepto

# ------------------------------------------------------------
# 8. GRAFICO
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(12, 7),
    dpi=160
)

# Datos observados
ax.scatter(
    X,
    Y,
    s=65,
    label="Consumo observado",
    zorder=3
)

# Línea base energética
ax.plot(
    x_linea,
    y_linea,
    linewidth=2.5,
    label="Línea base energética",
    zorder=2
)

# ------------------------------------------------------------
# 9. ECUACION Y R²
# ------------------------------------------------------------

ecuacion = (
    f"y = {pendiente:.2f}x + {intercepto:,.0f}\n"
    f"$R^2$ = {r2:.4f}"
)

ax.text(
    0.97,
    0.05,
    ecuacion,
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=13,
    bbox=dict(
        boxstyle="round,pad=0.45",
        facecolor="white",
        edgecolor="0.65"
    )
)

# ------------------------------------------------------------
# 10. FORMATO
# ------------------------------------------------------------

ax.set_title(
    "Línea Base Energética — Oil Malal",
    fontsize=20,
    pad=15
)

ax.set_xlabel(
    "Producción [m³/mes]",
    fontsize=14
)

ax.set_ylabel(
    "Consumo energético [kWh/mes]",
    fontsize=14
)

ax.tick_params(
    axis="both",
    labelsize=11
)

ax.grid(
    True,
    alpha=0.25
)

ax.legend(
    fontsize=11,
    loc="upper left"
)

# Márgenes
ax.margins(x=0.04, y=0.08)

plt.tight_layout()

# ------------------------------------------------------------
# 11. GUARDAR
# ------------------------------------------------------------

plt.savefig(
    archivo_salida,
    dpi=300,
    bbox_inches="tight"
)

# ------------------------------------------------------------
# 12. RESULTADOS
# ------------------------------------------------------------

print("=" * 60)
print("D6 - LINEA BASE ENERGETICA OIL MALAL")
print("=" * 60)

print(f"Observaciones : {len(X)}")
print(f"Pendiente     : {pendiente:.6f}")
print(f"Intercepto    : {intercepto:.3f}")
print(f"R²            : {r2:.6f}")

print("\nEcuacion de la linea base:")
print(
    f"Consumo = {pendiente:.4f} × Produccion + "
    f"{intercepto:.2f}"
)

print("\nGRAFICO GENERADO CORRECTAMENTE")
print("=" * 60)
print(f"Archivo: {archivo_salida}")

plt.show()