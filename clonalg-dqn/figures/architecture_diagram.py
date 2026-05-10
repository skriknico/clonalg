# figures/architecture_diagram.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")

# ── Colores ──────────────────────────────────────────────────────────────
C_ENV    = "#63AF4C"
C_BIO    = "#FF7043"
C_DQN    = "#5C6BC0"
C_DETAIL = "#FFA726"
C_ARROW  = "#455A64"

def box(ax, x, y, w, h, label, sublabel=None, color="#5C6BC0", fontsize=11):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="white",
        linewidth=1.5, alpha=0.92, zorder=3
    )
    ax.add_patch(rect)
    cy = y + h / 2 + (0.18 if sublabel else 0)
    ax.text(x + w/2, cy, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.28, sublabel,
                ha="center", va="center", fontsize=8.5,
                color="white", alpha=0.88, zorder=4)

def arrow(ax, x1, y1, x2, y2, label=None, color=C_ARROW, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=1.8, connectionstyle="arc3,rad=0.0"),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.18, label, ha="center", va="bottom",
                fontsize=8, color=color, style="italic", zorder=5)

def curved_arrow(ax, x1, y1, x2, y2, label=None, rad=0.25, color=C_ARROW):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=1.8, connectionstyle=f"arc3,rad={rad}"),
                zorder=2)
    if label:
        mx = (x1+x2)/2 + rad*1.2
        my = (y1+y2)/2
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=8, color=color, style="italic", zorder=5)

# ── Módulos principales ───────────────────────────────────────────────────
# Entorno
box(ax, 0.3, 3.5, 2.4, 1.2,
    "Entorno", "LunarLander-v3", C_ENV)

# Buffer CLONALG (módulo bioinspirado)
box(ax, 4.2, 5.2, 3.2, 1.8,
    "Buffer CLONALG", "Módulo bioinspirado", C_BIO)

# Agente DQN
box(ax, 9.5, 3.5, 3.8, 1.2,
    "Agente DQN", "Red Q + Target net", C_DQN)

# ── Sub-módulos internos de CLONALG ──────────────────────────────────────
box(ax, 4.3, 3.5, 1.4, 1.1, "Repertorio", "Anticuerpos", C_DETAIL, fontsize=9)
box(ax, 5.9, 3.5, 1.4, 1.1, "Clonación", "+ mutación",  C_DETAIL, fontsize=9)
box(ax, 4.3, 2.1, 3.0, 1.1, "Supresión clonal", "Diversidad garantizada", C_DETAIL, fontsize=9)

# ── Flechas principales ───────────────────────────────────────────────────
# Env → CLONALG (transición)
arrow(ax, 2.7, 4.1, 4.2, 5.8, "(s, a, r, s')", C_ENV)

# CLONALG → DQN (mini-batch)
arrow(ax, 7.4, 6.1, 9.5, 4.1, "mini-batch diverso", C_BIO)

# DQN → Env (acción)
curved_arrow(ax, 9.5, 3.8, 2.7, 3.8, "acción a", rad=-0.35, color=C_DQN)

# DQN → CLONALG (TD-errors, retroalimentación)
curved_arrow(ax, 11.0, 4.7, 7.4, 6.5, "TD-errors", rad=-0.3, color=C_DQN)

# Flechas internas CLONALG
arrow(ax, 4.3, 3.5, 4.8, 3.2, color=C_DETAIL)   # Repertorio → Supresión
arrow(ax, 5.9, 3.5, 5.8, 3.2, color=C_DETAIL)   # Clonación → Supresión
arrow(ax, 5.3, 5.2, 5.0, 4.6, color=C_BIO)       # CLONALG → Repertorio
arrow(ax, 5.8, 4.6, 6.3, 4.6, color=C_DETAIL)   # Repertorio → Clonación
arrow(ax, 5.8, 3.2, 5.8, 3.0, color=C_DETAIL)   # Supresión → salida

# ── Recuadro punteado zona bioinspirada ───────────────────────────────────
bio_rect = mpatches.FancyBboxPatch(
    (4.0, 1.8), 3.8, 5.5,
    boxstyle="round,pad=0.1",
    facecolor="none", edgecolor=C_BIO,
    linewidth=1.5, linestyle="--", alpha=0.6, zorder=1
)
ax.add_patch(bio_rect)
ax.text(5.9, 7.4, "Componente Bioinspirado",
        ha="center", fontsize=9, color=C_BIO,
        style="italic", alpha=0.85)

# ── Hiperparámetros ───────────────────────────────────────────────────────
hp_rect = mpatches.FancyBboxPatch(
    (0.2, 0.2), 3.2, 1.5,
    boxstyle="round,pad=0.1",
    facecolor="#ECEFF1", edgecolor="#90A4AE",
    linewidth=1, linestyle=":", alpha=0.9, zorder=1
)
ax.add_patch(hp_rect)
ax.text(1.8, 1.55, "Hiperparámetros barridos",
        ha="center", fontsize=8.5, fontweight="bold", color="#455A64")
params = [
    "N: capacidad buffer  {500, 1000, 2000}",
    "β: clone_rate  {0.3, 0.5, 0.7}",
    "lr: tasa aprendizaje  {1e-4, 5e-4, 1e-3}",
]
for i, p in enumerate(params):
    ax.text(0.4, 1.2 - i*0.33, p, fontsize=7.5, color="#455A64")

# ── Leyenda ───────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=C_ENV,    label="Entorno RL"),
    mpatches.Patch(facecolor=C_BIO,    label="Componente bioinspirado"),
    mpatches.Patch(facecolor=C_DQN,    label="Agente DQN"),
    mpatches.Patch(facecolor=C_DETAIL, label="Operaciones CLONALG"),
]
ax.legend(handles=legend_elements, loc="lower right",
          fontsize=9, framealpha=0.9)

ax.set_title("Arquitectura del Sistema Híbrido CLONALG + DQN",
             fontsize=13, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig("figures/architecture.png", dpi=200, bbox_inches="tight")
plt.savefig("figures/architecture.pdf", bbox_inches="tight")
print("Guardado en figures/architecture.png y figures/architecture.pdf")
plt.show()