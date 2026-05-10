import subprocess
import sys
import time

SWEEP = {
    "buffer_capacity": [500, 1000, 2000],
    "clone_rate":      [0.3, 0.5, 0.7],
    "lr":              [1e-4, 5e-4, 1e-3],
}

SEEDS      = [0, 1, 2]
N_EPISODES = 300

def run(tag, capacity, clone_rate, lr, seed):
    result = subprocess.run([
        sys.executable, "-m", "src.train",
        "--mode",       "clonalg",
        "--seed",       str(seed),
        "--episodes",   str(N_EPISODES),
        "--capacity",   str(capacity),
        "--clone_rate", str(clone_rate),
        "--lr",         str(lr),
        "--tag",        tag,
    ])
    if result.returncode != 0:
        print(f"ERROR en {tag} seed={seed}, reintentando...")
        time.sleep(5)
        subprocess.run([
            sys.executable, "-m", "src.train",
            "--mode",       "clonalg",
            "--seed",       str(seed),
            "--episodes",   str(N_EPISODES),
            "--capacity",   str(capacity),
            "--clone_rate", str(clone_rate),
            "--lr",         str(lr),
            "--tag",        tag,
        ])
    time.sleep(2)

if __name__ == "__main__":
    print("=== Barrido: buffer_capacity ===")
    for cap in SWEEP["buffer_capacity"]:
        for seed in SEEDS:
            tag = f"sweep_cap{cap}"
            print(f"  {tag} seed={seed}")
            run(tag, cap, 0.5, 1e-4, seed)

    print("=== Barrido: clone_rate ===")
    for cr in SWEEP["clone_rate"]:
        for seed in SEEDS:
            tag = f"sweep_cr{cr}"
            print(f"  {tag} seed={seed}")
            run(tag, 2000, cr, 1e-4, seed)

    print("=== Barrido: lr ===")
    for lr in SWEEP["lr"]:
        for seed in SEEDS:
            tag = f"sweep_lr{lr}"
            print(f"  {tag} seed={seed}")
            run(tag, 2000, 0.5, lr, seed)

    print("\nBarrido completado. CSVs en logs/")