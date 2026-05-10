# CLONALG + DQN — Sistema Híbrido Bioinspirado

Proyecto semestral de Aprendizaje Automático Bioinspirado (APBIO)  
Universidad de Vigo — Curso 2025–2026

## Descripción

Sistema híbrido que acopla el algoritmo de selección clonal **CLONALG**
como buffer de replay inteligente para un agente **DQN** en el entorno
`LunarLander-v3` de Gymnasium.

Las transiciones se almacenan como anticuerpos con afinidad proporcional
al TD-error. El sampling usa selección clonal con supresión para
garantizar diversidad en el mini-batch.

## Estructura
clonalg-dqn/
├── src/
│   ├── dqn.py              # Red Q y agente DQN
│   ├── clonalg_buffer.py   # Buffer inmune CLONALG
│   ├── train.py            # Bucle de entrenamiento
│   └── utils.py            # Semillas y helpers
├── experiments/
│   ├── run_all.py          # Lanza 5 semillas baseline + CLONALG
│   └── run_sweep.py        # Barrido de hiperparámetros
├── logs/                   # CSVs por semilla (generados)
├── figures/                # Figuras del informe (generadas)
├── report/
│   ├── main.tex            # Informe LaTeX
│   └── references.bib      # Referencias
├── plot_results.py         # Genera todas las figuras
└── requirements.txt

## Instalación

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Instalar PyTorch con CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Instalar resto de dependencias
pip install swig
pip install "gymnasium[box2d]"
pip install matplotlib pandas scipy tqdm
```

## Reproducir el experimento principal

```bash
python experiments/run_all.py
```

Lanza baseline y CLONALG con 5 semillas × 300 episodios cada una.
Tiempo estimado: ~60 minutos con GPU (RTX 3060).
Los CSVs se guardan en `logs/`.

**Salida esperada (retorno medio últimos 50 episodios):**

| Configuración | Retorno medio |
|---------------|---------------|
| DQN baseline  | 208.7 ± 22.1  |
| CLONALG + DQN | -245.4 ± 130.2|

## Reproducir el barrido de hiperparámetros

```bash
python experiments/run_sweep.py
```

Barre `buffer_capacity`, `clone_rate` y `lr` con 3 semillas × 300 episodios.
Tiempo estimado: ~90 minutos con GPU.

## Generar figuras

```bash
python figures/architecture_diagram.py
python plot_results.py
```

Las figuras se guardan en `figures/` como `.png` y `.pdf`.

## Experimento rápido (sin GPU, ~2 minutos)

```bash
python -m src.train --mode both --fast --seed 0
```

100 episodios, buffer pequeño, útil para verificar que el código funciona.

## Dependencias

torch>=2.0 (cu121)
gymnasium[box2d]>=0.29
numpy>=1.24
matplotlib>=3.7
pandas>=2.0
scipy>=1.10
tqdm>=4.65
