import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from PIL import Image

# ============================================================
# G3 - PROBLEMA MULTIVARIABLE DEL DESEMPEÑO ENERGÉTICO
# Versión final
# ============================================================

# ------------------------------------------------------------
# 1. RUTA DE SALIDA
# ------------------------------------------------------------

carpeta_proyecto = Path(__file__).resolve().parents[2]

carpeta_salida = carpeta_proyecto / "02_Graficos" / "Finales"
carpeta_salida.mkdir(parents=True, exist_ok=True)

archivo_salida = carpeta_salida / "G3_Problema_Multivariable_Energia.png"

# ------------------------------------------------------------
# 2. CREAR FIGURA
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
line = "#AAB5BC"

# ------------------------------------------------------------
# 4. TÍTULO
# ------------------------------------------------------------

ax.text(
    8,
    8.25,
    "El consumo energético depende de múltiples variables",
    ha="center",
    va="center",
    fontsize=21,
    fontweight="bold",
    color=dark
)

ax.text(
    8,
    7.72,
    "El contexto operacional debe incorporarse al análisis",
    ha="center",
    va="center",
    fontsize=13.5,
    color=gray
)

# ------------------------------------------------------------
# 5. TARJETAS DE VARIABLES
# ------------------------------------------------------------

cards = [
    (0.45, "Producción", "Volumen producido", "#2E86AB"),
    (3.55, "Temperatura", "Condiciones térmicas", "#F18F01"),
    (6.65, "Carga", "Nivel de utilización", "#6A994E"),
    (9.75, "Horas de operación", "Tiempo de\nfuncionamiento", "#8E6BBE"),
    (12.85, "Proceso", "Condiciones\noperacionales", "#C94C5B"),
]

card_y = 5.20
card_w = 2.70
card_h = 1.25
header_h = 0.38

for x, title, subtitle, accent in cards:

    # Caja principal
    ax.add_patch(
        FancyBboxPatch(
            (x, card_y),
            card_w,
            card_h,
            boxstyle="round,pad=0.015,rounding_size=0.08",
            linewidth=1.8,
            edgecolor=accent,
            facecolor="white"
        )
    )

    # Encabezado
    ax.add_patch(
        FancyBboxPatch(
            (x, card_y + card_h - header_h),
            card_w,
            header_h,
            boxstyle="round,pad=0.008,rounding_size=0.06",
            linewidth=0,
            facecolor=accent
        )
    )

    # Tamaño del título
    if title == "Horas de operación":
        title_size = 9.2
    else:
        title_size = 11.2

    # Título
    ax.text(
        x + card_w / 2,
        card_y + card_h - header_h / 2,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color="white"
    )

    # Descripción
    ax.text(
        x + card_w / 2,
        card_y + 0.39,
        subtitle,
        ha="center",
        va="center",
        fontsize=8.8,
        color=dark,
        linespacing=1.0
    )

# ------------------------------------------------------------
# 6. FLECHAS
# ------------------------------------------------------------

node_y = 2.15
node_h = 1.42

for x, *_ in cards:

    ax.add_patch(
        FancyArrowPatch(
            (x + card_w / 2, card_y - 0.03),
            (8, node_y + node_h + 0.06),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.3,
            color=line,
            connectionstyle="arc3,rad=0.035"
        )
    )

# ------------------------------------------------------------
# 7. NODO CENTRAL
# ------------------------------------------------------------

node_x = 4.65
node_w = 6.70

ax.add_patch(
    FancyBboxPatch(
        (node_x, node_y),
        node_w,
        node_h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=2.2,
        edgecolor=navy,
        facecolor=light
    )
)

ax.text(
    8,
    2.83,
    "CONSUMO ENERGÉTICO",
    ha="center",
    va="center",
    fontsize=15.5,
    fontweight="bold",
    color=navy
)

ax.text(
    8,
    2.47,
    "Resultado de múltiples condiciones de operación",
    ha="center",
    va="center",
    fontsize=9.6,
    color=dark
)

# ------------------------------------------------------------
# 8. MENSAJE INFERIOR
# ------------------------------------------------------------

ax.text(
    8,
    1.18,
    "El desafío es identificar y modelar relaciones que actúan simultáneamente",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
    color=dark
)

ax.text(
    8,
    0.68,
    "De variables aisladas  →  a una visión integrada del comportamiento energético",
    ha="center",
    va="center",
    fontsize=10.5,
    color=gray
)

# ------------------------------------------------------------
# 9. GUARDAR LA IMAGEN
# ------------------------------------------------------------

fig.savefig(
    archivo_salida,
    dpi=300,
    facecolor="white",
    edgecolor="none"
)

plt.close(fig)

# ------------------------------------------------------------
# 10. MOSTRAR EL MISMO ARCHIVO GUARDADO
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

# ------------------------------------------------------------
# 11. INFORMACIÓN
# ------------------------------------------------------------

print("=" * 60)
print("G3 GENERADO CORRECTAMENTE")
print("=" * 60)
print(f"Archivo: {archivo_salida}")
print("La imagen mostrada corresponde al PNG guardado.")

plt.show()