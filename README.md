# Windy TSP: Asymmetric Neural Combinatorial Optimization

This repository provides a deep Reinforcement Learning (RL) framework designed to solve the **Windy Traveling Salesman Problem (Windy TSP)**—a variant of the **Asymmetric TSP (ATSP)** where travel costs are heavily influenced by directional wind vectors.

By extending the standard Transformer and Graph Neural Network paradigms, this project introduces architectures capable of natively understanding non-Euclidean, directed graphs without suffering from catastrophic FP16 overflow or gradient collapse.

---

# Acknowledgments & Credits

This project is an advanced fork and extension of the excellent **learning-tsp** repository by **chaitjo**, which itself implements:

- The foundational **Attention Model** by **Kool et al. (2019)**
- The **GCN framework** by **Bresson et al. (2018)**

## Key Architectural Contributions in this Fork

- **Asymmetric Node Embeddings (ANE)**
  - A multi-modal *Fat Vector* packing:
    - Spatial coordinates
    - Physical wind vectors
    - Local topology
    - Global cost statistics
  - Safely compressed via log-scaling.

- **Neural Adaptive Bias (NAB)**
  - A trainable module that explicitly projects asymmetric cost matrices directly into the logit space of Attention mechanisms and GNN gating layers.

- **Adaptation Attention-Free Module (AAFM)**
  - A novel, numerically stable (FP16-safe) routing module that replaces multi-head attention with explicit LogSumExp NAB integration.

- **Directional GNNs**
  - Upgraded Graph Convolutional layers supporting:
    - Forward message passing
    - Backward message passing
    - Dual (bi-directional fusion) message passing
  - Designed to prevent over-smoothing on directed edges.

- **Physics-Aware Environments**
  - Custom continuous data generators
  - Robust reachability sparsification (kNN with minimum in-degrees)
  - Decoupled RL reward / feature scaling pipelines

---

# ⚙️ Installation & Setup

## 1. Python Environment

You can install the dependencies locally or use the provided Docker setup.

### Local Installation

```bash
git clone https://github.com/YOUR_USERNAME/windy_tsp.git
cd windy_tsp

pip install -r requirements.txt
```

### Docker Installation

```bash
docker build -t windy-tsp .

docker docker run --user $(id -u):$(id -g) --gpus all -it --rm --ipc=host -v $(pwd):/workspace -v /mnt/Data-fast/gms:/mnt/Data-fast/gms windy-tsp-modern:latest bash
```

---

## 2. Setting up the Optimal Baseline Solver (LKH-3)

This repository uses **LKH-3** to generate exact (or highly optimized) ground-truth tours for the Asymmetric TSP.

Because LKH-3 is compiled from C source, it must be downloaded and built locally (it is intentionally excluded from version control).

```bash
# Download and extract LKH-3 into the repository root

wget http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.6.tgz

tar xvfz LKH-3.0.6.tgz

cd LKH-3.0.6

# Compile the binary
make

cd ..
```

> **Note:** `eval_baseline.py` automatically locates the compiled LKH executable.

---

# 🚀 Running the Code

To streamline large-scale experimentation, this repository includes heavily automated Bash scripts that handle everything from data generation to parallel GPU round-robin execution.

---

## 1. Training (Ablation Studies)

Launch a large grid search of architectures (encoders, NAB configurations, GNN directions):

```bash
bash train_ablations.sh
```

### What it does

- Automatically generates validation `.pkl` datasets
- Computes the optimal LKH-3 baseline
- Launches background Python jobs
- Uses strict concurrency controls (`MAX_PARALLEL_JOBS`)
- Automatically assigns GPUs

### Monitoring
Track the master log via the tail -f shown in the terminal

---

## 2. Smart Continuation

If training is interrupted or you want to continue training to a higher epoch count (e.g. 100 → 200 epochs), update `TARGET_TOTAL_EPOCHS` inside `resume_ablations.sh` and run:

```bash
bash resume_ablations.sh
```

### What it does

- Parses existing output directories
- Finds the highest `.pt` checkpoint for every valid configuration
- Seamlessly resumes training
- Preserves existing logs

---

## 3. Neural Network Evaluation

Evaluate all trained checkpoints inside an output directory:

```bash
bash eval_ablations.sh
```

### What it does

- Loads every trained model
- Runs Greedy decoding
- Runs Beam Search decoding
- Computes optimality gaps against LKH-3
- Measures latent-space metrics such as:
  - GNN Dirichlet Energy
  - Over-smoothing indicators
- Aggregates results into a single CSV

---

# 📊 Traditional Heuristics & Diagnostics

## Operations Research Baselines

Standard OR heuristics are evaluated using **pyCombinatorial**, including:

- Nearest Insertion
- Tabu Search
- Genetic Algorithms
- Other classical heuristics

```bash
python eval_heuristics.py \
    --method all \
    --dataset_path data/windy_tsp/windy_tsp50_val.pkl
```

> **Note:** Ensure LKH-3 has been compiled first, since this script compares heuristic performance against LKH ground-truth solutions.

---

## Visual Diagnostics

To visually verify that the model respects asymmetric edge masking and correctly models wind physics:

```bash
python plot_diagnostic.py
```

The script performs inference on a single graph and generates:

```
diagnostic_tour.png
```

The visualization includes:

- Legal traversals
- Graph-mask violations
- Global wind vector overlay

---

# 📁 Repository Structure

```text
nets/
├── encoders/
│   ├── aafm_encoder.py
│   ├── edge_gat_encoder.py
│   ├── gnn_encoder.py
│   └── ...

├── attention_model.py
└── critic_network.py

problems/
└── tsp/
    ├── problem_tsp.py
    └── ...

data/
└── windy_tsp/
    └── generate_windy_tsp.py

utils/
```

## Main Components

### `nets/`

Core neural architectures.

- `encoders/`
  - `aafm_encoder.py`
  - `edge_gat_encoder.py`
  - `gnn_encoder.py`
  - Additional encoder implementations

- `attention_model.py`
  - Primary autoregressive actor model

- `critic_network.py`
  - Value estimator baseline (standard TSP only)

### `problems/tsp/`

Environment definitions.

- `problem_tsp.py`
  - WindyTSP environment
  - Custom kNN masking
  - Feature/reward scaling split

### `data/windy_tsp/`

Dataset generation.

- `generate_windy_tsp.py`
  - Generates `.pkl` datasets
  - Validates asymmetric physics through strict assertions:

```math
C_{ij} \neq C_{ji}
```

### `utils/`

Utility scripts for:

- Logging
- Beam search
- Checkpoint management

---
