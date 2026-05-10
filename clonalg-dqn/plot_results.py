import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import glob
import os
from scipy import stats

os.makedirs("figures", exist_ok=True)

WINDOW     = 20
SEEDS      = [0, 1, 2, 3, 4]
N_EPISODES = 500

# ── Utilidades ────────────────────────────────────────────────────────────
def moving_average(x, w):
    if len(x) < w:
        return np.array(x, dtype=float)
    return np.convolve(x, np.ones(w)/w, mode="valid")

def load_runs(tag, log_dir="logs"):
    runs = []
    for seed in SEEDS:
        path = os.path.join(log_dir, f"{tag}_seed{seed}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path,
                         names=["episode","return","epsilon",
                                "loss","mean_affinity"])
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        runs.append(df["return"].values)
    return runs

def mean_std(runs):
    min_len = min(len(r) for r in runs)
    arr = np.array([r[:min_len] for r in runs])
    return arr.mean(axis=0), arr.std(axis=0)

def load_column(tag, col, log_dir="logs"):
    runs = []
    for seed in SEEDS:
        path = os.path.join(log_dir, f"{tag}_seed{seed}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path,
                         names=["episode","return","epsilon",
                                "loss","mean_affinity"])
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        if col in df.columns:
            runs.append(df[col].values)
    return runs

# ── 1. Curvas de aprendizaje (híbrido vs baseline) ────────────────────────
def plot_learning_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Curvas de aprendizaje — CLONALG+DQN vs DQN baseline",
                 fontsize=13, fontweight="bold")

    for tag, label, color in [("baseline", "DQN baseline", "#5C6BC0"),
                               ("clonalg",  "CLONALG+DQN", "#FF7043")]:
        runs = load_runs(tag)
        if not runs:
            print(f"No se encontraron runs para {tag}")
            continue
        mu, sigma = mean_std(runs)
        sm_mu     = moving_average(mu, WINDOW)
        ep        = np.arange(len(sm_mu))

        # Banda de varianza (± std suavizada)
        sm_sigma = moving_average(sigma, WINDOW)

        axes[0].plot(ep, sm_mu, label=label, color=color, linewidth=2)
        axes[0].fill_between(ep, sm_mu - sm_sigma,
                             sm_mu + sm_sigma,
                             alpha=0.2, color=color)

        # Curvas individuales por semilla
        for r in runs:
            axes[1].plot(moving_average(r, WINDOW),
                         alpha=0.35, linewidth=1, color=color)

    axes[0].axhline(200, color="gray", linestyle="--",
                    linewidth=1, label="Umbral éxito (200)")
    axes[0].set_title("Media ± desv. estándar entre semillas")
    axes[0].set_xlabel("Episodio")
    axes[0].set_ylabel("Retorno")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Curvas individuales por semilla")
    axes[1].set_xlabel("Episodio")
    axes[1].set_ylabel("Retorno")
    axes[1].axhline(200, color="gray", linestyle="--", linewidth=1)
    axes[1].grid(True, alpha=0.3)

    # Leyenda manual para el subplot derecho
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color="#5C6BC0", lw=2, label="DQN baseline"),
               Line2D([0],[0], color="#FF7043", lw=2, label="CLONALG+DQN")]
    axes[1].legend(handles=handles)

    plt.tight_layout()
    plt.savefig("figures/learning_curves.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/learning_curves.pdf", bbox_inches="tight")
    print("Guardado: figures/learning_curves.png")
    plt.close()

# ── 2. Estudio de ablación ────────────────────────────────────────────────
def plot_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Estudio de ablación — contribución del módulo CLONALG",
                 fontsize=13, fontweight="bold")

    results = {}
    for tag, label, color in [("baseline", "DQN baseline", "#5C6BC0"),
                               ("clonalg",  "CLONALG+DQN", "#FF7043")]:
        runs = load_runs(tag)
        if not runs:
            continue
        mu, sigma = mean_std(runs)
        results[tag] = {"mu": mu, "sigma": sigma,
                        "runs": runs, "label": label, "color": color}

        sm = moving_average(mu, WINDOW)
        ep = np.arange(len(sm))
        axes[0].plot(ep, sm, label=label, color=color, linewidth=2)
        axes[0].fill_between(ep,
                             moving_average(mu - sigma, WINDOW),
                             moving_average(mu + sigma, WINDOW),
                             alpha=0.2, color=color)

    axes[0].axhline(200, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Retorno medio suavizado")
    axes[0].set_xlabel("Episodio")
    axes[0].set_ylabel("Retorno")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot con retornos finales (últimos 50 episodios)
    box_data  = []
    box_labels = []
    box_colors = []
    for tag in ["baseline", "clonalg"]:
        if tag not in results:
            continue
        finals = [r[-50:].mean() for r in results[tag]["runs"]]
        box_data.append(finals)
        box_labels.append(results[tag]["label"])
        box_colors.append(results[tag]["color"])

    bp = axes[1].boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                         medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    axes[1].set_title("Retorno medio — últimos 50 episodios por semilla")
    axes[1].set_ylabel("Retorno medio")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Test estadístico Welch
    if "baseline" in results and "clonalg" in results:
        a = [r[-50:].mean() for r in results["baseline"]["runs"]]
        b = [r[-50:].mean() for r in results["clonalg"]["runs"]]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        axes[1].set_xlabel(f"Prueba t de Welch: t={t:.2f}, p={p:.3f}")

    plt.tight_layout()
    plt.savefig("figures/ablation.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/ablation.pdf", bbox_inches="tight")
    print("Guardado: figures/ablation.png")
    plt.close()

# ── 3. Dinámica interna de CLONALG (diagnóstico bioinspirado) ─────────────
def plot_bioinspired_diagnostics():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Diagnósticos del módulo CLONALG",
                 fontsize=13, fontweight="bold")

    aff_runs = load_column("clonalg", "mean_affinity")
    if aff_runs:
        # Recorta todas las runs a la misma longitud
        min_len  = min(len(r) for r in aff_runs)
        aff_arr  = np.array([r[:min_len] for r in aff_runs])
        mu_aff   = aff_arr.mean(axis=0)
        std_aff  = aff_arr.std(axis=0)
        ep       = np.arange(min_len)
        axes[0].plot(ep, mu_aff, color="#FF7043", linewidth=2)
        axes[0].fill_between(ep, mu_aff - std_aff,
                             mu_aff + std_aff,
                             alpha=0.2, color="#FF7043")
    axes[0].set_title("Afinidad media del repertorio")
    axes[0].set_xlabel("Episodio")
    axes[0].set_ylabel("Afinidad media (≈ |TD-error|)")
    axes[0].grid(True, alpha=0.3)

    for tag, label, color in [("baseline", "DQN baseline", "#5C6BC0"),
                               ("clonalg",  "CLONALG+DQN", "#FF7043")]:
        loss_runs = load_column(tag, "loss")
        if not loss_runs:
            continue
        min_len  = min(len(r) for r in loss_runs)
        mu_loss  = np.array([r[:min_len] for r in loss_runs]).mean(axis=0)
        sm_loss  = moving_average(np.clip(mu_loss, 0, 500), WINDOW)
        axes[1].plot(sm_loss, label=label, color=color,
                     linewidth=2, alpha=0.85)

    axes[1].set_title("Loss de entrenamiento (clippeada a 500)")
    axes[1].set_xlabel("Episodio")
    axes[1].set_ylabel("MSE Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/bioinspired_diagnostics.png",
                dpi=150, bbox_inches="tight")
    plt.savefig("figures/bioinspired_diagnostics.pdf", bbox_inches="tight")
    print("Guardado: figures/bioinspired_diagnostics.png")
    plt.close()

# ── 4. Barrido de hiperparámetros ─────────────────────────────────────────
def plot_sweep():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Barrido de hiperparámetros",
                 fontsize=13, fontweight="bold")

    sweeps = [
        ("buffer_capacity", [500, 1000, 2000],
         ["sweep_cap500", "sweep_cap1000", "sweep_cap2000"],
         "Capacidad buffer (N)"),
        ("clone_rate", [0.3, 0.5, 0.7],
         ["sweep_cr0.3", "sweep_cr0.5", "sweep_cr0.7"],
         "Tasa de clonación (β)"),
        ("lr", [1e-4, 5e-4, 1e-3],
         ["sweep_lr0.0001", "sweep_lr0.0005", "sweep_lr0.001"],
         "Tasa de aprendizaje (lr)"),
    ]

    for ax, (param, values, tags, xlabel) in zip(axes, sweeps):
        means, stds = [], []
        for tag in tags:
            runs = load_runs(tag)
            if not runs:
                means.append(0)
                stds.append(0)
                continue
            finals = [r[-50:].mean() for r in runs]
            means.append(np.mean(finals))
            stds.append(np.std(finals))

        x = np.arange(len(values))
        bars = ax.bar(x, means, yerr=stds, capsize=6,
                      color="#FF7043", alpha=0.8,
                      error_kw=dict(elinewidth=1.5, ecolor="#455A64"))
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in values])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Retorno medio (últimos 50 ep)")
        ax.set_title(f"Barrido: {param}")
        ax.axhline(200, color="gray", linestyle="--",
                   linewidth=1, label="Umbral éxito")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("figures/sweep.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/sweep.pdf", bbox_inches="tight")
    print("Guardado: figures/sweep.png")
    plt.close()

# ── Ejecutar todo ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generando figuras del informe...")
    plot_learning_curves()
    plot_ablation()
    plot_bioinspired_diagnostics()
    plot_sweep()
    print("\nTodas las figuras generadas en figures/")