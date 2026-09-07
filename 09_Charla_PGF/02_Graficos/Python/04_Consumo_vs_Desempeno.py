import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from PIL import Image

# ============================================================
# G4 - CONSUMO VS. DESEMPEÑO
# ============================================================

# ------------------------------------------------------------
# 1. RUTA DE SALIDA
# ------------------------------------------------------------

carpeta_proyecto = Path(__file__).resolve().parents[2]

carpeta_salida = carpeta_proyecto / "02_Graficos" / "Finales"
carpeta_salida.mkdir(parents=True, exist_ok=True)

archivo_salida = carpeta_salida / "G4_Consumo_vs_Desempeno.png"

# ------------------------------------------------------------
# 2. FIGURA 16:9
# ------------------------------------------------------------

fig = plt.figure(
    figsize=(16, 9),
    dpi=300,
    facecolor="white"
)

ax = fig.add_axes([0, 0, 1, 1])

ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

# ------------------------------------------------------------
# 3. COLORES
# ------------------------------------------------------------

dark = "#263238"
navy = "#173F5F"
gray = "#607D8B"
light = "#F4F7F9"

blue = "#2E86AB"
orange = "#F18F01"

# ------------------------------------------------------------
# 4. TÍTULO
# ------------------------------------------------------------

ax.text(
    8, 8.25,
    "Consumo energético vs. desempeño",
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold",
    color=dark
)

ax.text(
    8, 7.72,
    "El consumo debe interpretarse en función de lo que el proceso produce",
    ha="center",
    va="center",
    fontsize=13.5,
    color=gray
)

# ------------------------------------------------------------
# 5. ESCENARIOS
# ------------------------------------------------------------

escenarios = [
    (
        1.25,
        "ESCENARIO A",
        "Menor consumo",
        blue,
        100,
        80,
        1.25
    ),
    (
        8.75,
        "ESCENARIO B",
        "Mayor consumo",
        orange,
        120,
        120,
        1.00
    )
]

for (
    x,
    titulo,
    subtitulo,
    accent,
    consumo,
    produccion,
    intensidad
) in escenarios:

    # Caja principal
    ax.add_patch(
        FancyBboxPatch(
            (x, 3.05),
            6.0,
            3.55,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.8,
            edgecolor=accent,
            facecolor="white"
        )
    )

    # Encabezado
    ax.add_patch(
        FancyBboxPatch(
            (x, 5.92),
            6.0,
            0.68,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            linewidth=0,
            facecolor=accent
        )
    )

    ax.text(
        x + 3.0,
        6.26,
        titulo,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="white"
    )

    ax.text(
        x + 3.0,
        5.58,
        subtitulo,
        ha="center",
        va="center",
        fontsize=11,
        color=gray
    )

    # Consumo
    ax.text(
        x + 1.55,
        4.80,
        "Consumo",
        ha="center",
        va="center",
        fontsize=10.5,
        color=dark
    )

    ax.text(
        x + 1.55,
        4.42,
        f"{consumo} kWh",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=dark
    )

    # Producción
    ax.text(
        x + 4.45,
        4.80,
        "Producción",
        ha="center",
        va="center",
        fontsize=10.5,
        color=dark
    )

    ax.text(
        x + 4.45,
        4.42,
        f"{produccion} unidades",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=dark
    )

    # Indicador
    ax.text(
        x + 3.0,
        3.72,
        "Indicador de desempeño",
        ha="center",
        va="center",
        fontsize=10.5,
        color=gray
    )

    ax.text(
        x + 3.0,
        3.36,
        f"Intensidad energética = {intensidad:.2f} kWh/unidad",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=navy
    )

# ------------------------------------------------------------
# 6. CONCLUSIÓN
# ------------------------------------------------------------

ax.add_patch(
    FancyBboxPatch(
        (3.00, 1.45),
        10.00,
        0.90,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.8,
        edgecolor=navy,
        facecolor=light
    )
)

ax.text(
    8,
    1.98,
    "El escenario B consume más energía, pero presenta mejor desempeño relativo",
    ha="center",
    va="center",
    fontsize=13.5,
    fontweight="bold",
    color=navy
)

ax.text(
    8,
    1.68,
    "Por eso, consumo y desempeño no deben analizarse como si fueran lo mismo.",
    ha="center",
    va="center",
    fontsize=10.5,
    color=dark
)

# ------------------------------------------------------------
# 7. GUARDAR
# ------------------------------------------------------------

fig.savefig(
    archivo_salida,
    dpi=300,
    facecolor="white",
    edgecolor="none"
)

plt.close(fig)

# ------------------------------------------------------------
# 8. MOSTRAR EL PNG GUARDADO
# ------------------------------------------------------------

imagen = Image.open(archivo_salida)

fig_mostrar = plt.figure(
    figsize=(16, 9),
    dpi=100,
    facecolor="white"
)

ax_mostrar = fig_mostrar.add_axes([0, 0, 1, 1])

ax_mostrar.imshow(
    imagen,
    aspect="auto",
    interpolation="nearest"
)

ax_mostrar.axis("off")

print("=" * 60)
print("G4 GENERADO CORRECTAMENTE")
print("=" * 60)
print(f"Archivo: {archivo_salida}")
print("La imagen mostrada corresponde al PNG guardado.")

plt.show()