import subprocess
import sys
import time

seeds = [0, 1, 2, 3, 4]
n_episodes = 300

print("=" * 50)
print("Lanzando experimentos Baseline (5 semillas)")
print("=" * 50)

for seed in seeds:
    print(f"\nBaseline seed={seed}")
    result = subprocess.run([
        sys.executable, "-m", "src.train",
        "--mode",     "baseline",
        "--seed",     str(seed),
        "--episodes", str(n_episodes),
    ])
    if result.returncode != 0:
        print(f"ERROR en seed={seed}, reintentando en 10s...")
        time.sleep(10)
        subprocess.run([
            sys.executable, "-m", "src.train",
            "--mode",     "baseline",
            "--seed",     str(seed),
            "--episodes", str(n_episodes),
        ], check=True)
    time.sleep(3)

print("\nTodos los experimentos Baseline completados.")

print("=" * 50)
print("Lanzando experimentos Clonalg (5 semillas)")
print("=" * 50)

for seed in seeds:
    print(f"\nCLONALG seed={seed}")
    result = subprocess.run([
        sys.executable, "-m", "src.train",
        "--mode",     "clonalg",
        "--seed",     str(seed),
        "--episodes", str(n_episodes),
    ])
    if result.returncode != 0:
        print(f"ERROR en seed={seed}, reintentando en 10s...")
        time.sleep(10)
        subprocess.run([
            sys.executable, "-m", "src.train",
            "--mode",     "clonalg",
            "--seed",     str(seed),
            "--episodes", str(n_episodes),
        ], check=True)
    time.sleep(3)

print("\nTodos los experimentos CLONALG completados.")

