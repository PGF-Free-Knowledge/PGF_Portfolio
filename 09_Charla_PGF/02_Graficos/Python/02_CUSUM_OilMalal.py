import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# D7 - CUSUM OIL MALAL
# Fuente: Excel original
# ============================================================

# ------------------------------------------------------------
# 1. RUTAS
# ------------------------------------------------------------

carpeta_proyecto = Path(__file__).resolve().parents[2]

archivo = carpeta_proyecto / "Prevaluación_Paper_TGII_v60_pgf.xlsx"
hoja = "Consumo Eléctrico"

carpeta_salida = carpeta_proyecto / "02_Graficos" / "Finales"
carpeta_salida.mkdir(parents=True, exist_ok=True)

archivo_salida = carpeta_salida / "CUSUM_OilMalal_D7.png"

# ------------------------------------------------------------
# 2. LECTURA DEL EXCEL
# ------------------------------------------------------------

df = pd.read_excel(
    archivo,
    sheet_name=hoja,
    header=None
)

# ------------------------------------------------------------
# 3. TABLA CUSUM
# ------------------------------------------------------------

# Tabla ubicada en:
# Mes | Producción | Consumo | Consumo Teórico |
# E-real-E-LBE | CUSUM

datos = df.iloc[695:719, [23, 24, 25, 26, 27, 28]].copy()

datos.columns = [
    "Mes",
    "Produccion",
    "Consumo",
    "Consumo_Teorico",
    "Desviacion",
    "CUSUM"
]

# Convertir CUSUM a número
datos["CUSUM"] = pd.to_numeric(
    datos["CUSUM"],
    errors="coerce"
)

datos = datos.dropna(subset=["CUSUM"])

# ------------------------------------------------------------
# 4. GRAFICO
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(12, 7),
    dpi=160
)

ax.plot(
    datos["Mes"],
    datos["CUSUM"],
    marker="o",
    linewidth=2.2,
    markersize=5,
    label="CUSUM"
)

# Línea de referencia
ax.axhline(
    0,
    linewidth=1.2,
    linestyle="--"
)

ax.set_title(
    "CUSUM: evolución acumulada del desempeño energético",
    fontsize=19,
    pad=15
)

ax.set_xlabel(
    "Período",
    fontsize=13
)

ax.set_ylabel(
    "CUSUM [kWh/mes]",
    fontsize=13
)

ax.tick_params(
    axis="x",
    rotation=45,
    labelsize=10
)

ax.grid(
    True,
    alpha=0.25
)

ax.legend(
    fontsize=11,
    loc="best"
)

plt.tight_layout()

# ------------------------------------------------------------
# 5. GUARDAR
# ------------------------------------------------------------

plt.savefig(
    archivo_salida,
    dpi=300,
    bbox_inches="tight"
)

print("=" * 60)
print("CUSUM GENERADO CORRECTAMENTE")
print("=" * 60)
print(f"Observaciones : {len(datos)}")
print(f"Archivo       : {archivo_salida}")

plt.show()