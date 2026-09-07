import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from PIL import Image

# ============================================================
# G5 - EL CONSUMO NECESITA CONTEXTO
# Gráfico conceptual para la charla
# ============================================================

# ------------------------------------------------------------
# 1. RUTA DE SALIDA
# ------------------------------------------------------------

carpeta_proyecto = Path(__file__).resolve().parents[2]

carpeta_salida = carpeta_proyecto / "02_Graficos" / "Finales"
carpeta_salida.mkdir(parents=True, exist_ok=True)

archivo_salida = carpeta_salida / "G5_Consumo_Necesita_Contexto.png"

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
green = "#6A994E"
purple = "#8E6BBE"
red = "#C94C5B"

# ------------------------------------------------------------
# 4. TÍTULO
# ------------------------------------------------------------

ax.text(
    8,
    8.25,
    "El consumo energético necesita contexto",
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold",
    color=dark
)

ax.text(
    8,
    7.72,
    "Comparar consumos requiere considerar las condiciones en que opera el proceso",
    ha="center",
    va="center",
    fontsize=13.5,
    color=gray
)

# ------------------------------------------------------------
# 5. CONSUMO OBSERVADO
# ------------------------------------------------------------

ax.add_patch(
    FancyBboxPatch(
        (0.85, 3.45),
        3.10,
        1.70,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=2,
        edgecolor=navy,
        facecolor=light
    )
)

ax.text(
    2.40,
    4.55,
    "CONSUMO",
    ha="center",
    va="center",
    fontsize=17,
    fontweight="bold",
    color=navy
)

ax.text(
    2.40,
    4.02,
    "¿Cuánto se consumió?",
    ha="center",
    va="center",
    fontsize=12,
    color=dark
)

# ------------------------------------------------------------
# 6. FLECHA
# ------------------------------------------------------------

ax.add_patch(
    FancyArrowPatch(
        (4.05, 4.30),
        (5.10, 4.30),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2,
        color=gray
    )
)

# ------------------------------------------------------------
# 7. CONTEXTO OPERACIONAL
# ------------------------------------------------------------

ax.text(
    8,
    6.45,
    "CONTEXTO OPERACIONAL",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    color=dark
)

variables = [
    (5.05, 4.95, "Producción", blue),
    (7.15, 4.95, "Carga", green),
    (9.25, 4.95, "Temperatura", orange),
    (11.35, 4.95, "Horas de operación", purple),
    (7.15, 3.45, "Proceso", red),
]

for x, y, texto, color in variables:

    ancho = 1.85

    if texto == "Horas de operación":
        ancho = 2.25

    ax.add_patch(
        FancyBboxPatch(
            (x - ancho / 2, y - 0.42),
            ancho,
            0.84,
            boxstyle="round,pad=0.015,rounding_size=0.08",
            linewidth=1.5,
            edgecolor=color,
            facecolor="white"
        )
    )

    ax.text(
        x,
        y,
        texto,
        ha="center",
        va="center",
        fontsize=9.2 if texto == "Horas de operación" else 10.5,
        fontweight="bold",
        color=dark
    )

# ------------------------------------------------------------
# 8. FLECHA HACIA DESEMPEÑO
# ------------------------------------------------------------

ax.add_patch(
    FancyArrowPatch(
        (11.95, 4.30),
        (12.95, 4.30),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2,
        color=gray
    )
)

# ------------------------------------------------------------
# 9. DESEMPEÑO
# ------------------------------------------------------------

ax.add_patch(
    FancyBboxPatch(
        (13.00, 3.45),
        2.15,
        1.70,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=2,
        edgecolor=green,
        facecolor="#F3F8F1"
    )
)

ax.text(
    14.075,
    4.55,
    "DESEMPEÑO",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
    color=green
)

ax.text(
    14.075,
    4.02,
    "¿Es eficiente?",
    ha="center",
    va="center",
    fontsize=12,
    color=dark
)

# ------------------------------------------------------------
# 10. MENSAJE INFERIOR
# ------------------------------------------------------------

ax.add_patch(
    FancyBboxPatch(
        (2.35, 1.25),
        11.30,
        1.05,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.8,
        edgecolor=navy,
        facecolor=light
    )
)

ax.text(
    8,
    1.93,
    "Para evaluar el desempeño, primero debemos separar el efecto de las condiciones de operación.",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
    color=navy
)

ax.text(
    8,
    1.58,
    "Consumo observado + contexto operacional → desempeño energético comparable",
    ha="center",
    va="center",
    fontsize=10.5,
    color=dark
)

# ------------------------------------------------------------
# 11. GUARDAR
# ------------------------------------------------------------

fig.savefig(
    archivo_salida,
    dpi=300,
    facecolor="white",
    edgecolor="none"
)

plt.close(fig)

# ------------------------------------------------------------
# 12. MOSTRAR EL PNG REALMENTE GUARDADO
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
print("G5 GENERADO CORRECTAMENTE")
print("=" * 60)
print(f"Archivo: {archivo_salida}")
print("La imagen mostrada corresponde al PNG guardado.")

plt.show()