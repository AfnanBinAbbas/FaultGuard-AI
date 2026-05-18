# FaultGuard-AI: Next-Gen Log-Based Fault Diagnosis System

> **A State-of-the-Art, End-to-End Fault Diagnosis Platform with Cross-Domain Adaptation, 6 Diagnostic Tasks, and Dynamic Multi-Task Interaction. Built on Chimera.**

---

## Key Features

* **Interactive Multi-Task Learning:** Bridges the gap between anomaly detection and root cause localization, achieving state-of-the-art diagnostic capabilities.
* **Cross-Domain Adaptation:** Fully integrated domain-adaptation layers allowing robust models to perform diagnostics across varying system environments.
* **6 Concurrent Diagnostic Tasks:** Scaled up from standard systems to coordinate six distinct diagnostic and classification tasks simultaneously.
* **AI-Driven Automation:** Dynamically balances task weight distributions during training, achieving a **65% to 75% performance gain** over baseline independent systems.

---

## System Architecture

`FaultGuard-AI` is structured as a parent orchestration repository that integrates the core research engine via a tracked **Git Submodule** (`chimera`):

```
FaultGuard-AI  (Parent Orchestrator)
 ├─ .gitmodules       # Submodule configuration
 ├─ README.md         # Parent system documentation
 ├─ LICENSE           # System license
 └─ chimera/          # [Submodule] Core Research Engine (Forked & Optimized)
     ├─ src/          # Multi-task expansion & domain adaptation models
     ├─ scripts/      # Best checkpoint selectors & comparison plotters
     ├─ report_output/# Evaluated summaries, charts, and metrics
     └─ main.py       # Training and evaluation entrypoint
```

---

## Quick Start Guide

### 1. Clone the Repository Recursively
Because `FaultGuard-AI` utilizes `chimera` as a submodule, you must clone recursively to fetch all files:
```bash
git clone --recursive https://github.com/AfnanBinAbbas/FaultGuard-AI.git
cd FaultGuard-AI
```
*(If you already cloned, run `git submodule update --init --recursive` to pull the submodule content).*

### 2. Environment Setup
Enter the core engine directory and install dependencies:
```bash
cd chimera
pip install -r requirements.txt
```

### 3. Preparing Datasets
Put your raw log data (like BGL and Thunderbird) under `chimera/data/`. Pre-trained GloVe embeddings should go under `chimera/glove/` (refer to [chimera/README.md](chimera/README.md) for data preparation steps).

### 4. Training the Multi-Task Model
Train the unified model for 150 epochs with dynamic task weights and automated validation checkpointing:
```bash
python main.py --mode train --dataset BGL --epochs 150 --batch_size 256
```

### 5. Dynamic Checkpoint Selection and Evaluation
Automatically select the best checkpoint from your training history and evaluate it:
```bash
# Select best checkpoint
python scripts/select_best_checkpoint.py --dataset BGL --device cpu --checkpoint-dir checkpoint

# Evaluate the model
python main.py --mode eval --load_checkpoint True --dataset BGL
```

---

## Evaluation & Research Validation

The core engine automatically generates evaluation reports and comparison figures:

* **Performance Charts:** Located at `chimera/report_output/fig_bgl_paper_vs_novelties.png` and `chimera/report_output/fig_bgl_novelty_auxiliary_heads.png`.
* **Novelty Breakdown:** See [chimera/NOVELTIES.md](chimera/NOVELTIES.md) for details on the implemented expansions, domain adaptation, and dual-attention interactions.
* **Full Research Report:** Detailed human-readable metrics comparisons are located at [chimera/report_output/PROJECT_REPORT.md](chimera/report_output/PROJECT_REPORT.md).

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
