# FaultGuard-AI: Next-Gen Log-Based Fault Diagnosis System

FaultGuard-AI is a State-of-the-Art, End-to-End Fault Diagnosis Platform with Cross-Domain Adaptation, 6 Diagnostic Tasks, and Dynamic Multi-Task Interaction. 

It is built on top of the interactive multi-task learning architecture proposed in the research paper: *United We Stand: Towards End-to-End Log-based Fault Diagnosis via Interactive Multi-Task Learning* (ASE 2025).

---

## Table of Contents

- [Description](#description)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Datasets](#datasets)
- [Environment](#environment)
- [Preparation](#preparation)
- [Quick Start Guide](#quick-start-guide)
- [Reproducibility and Outputs](#reproducibility-and-outputs)
- [License](#license)

---

## Description

Log-based fault diagnosis is essential for maintaining software system availability. However, traditional fault diagnosis methods are built in a task-independent manner, which fails to bridge the gap between anomaly detection (AD) and root cause localization (RCL) in terms of data form and diagnostic objectives. This creates three major issues:
1. Diagnostic bias accumulates in the pipeline.
2. System deployment relies on expensive, fully labeled monitoring data.
3. The collaborative relationship and knowledge sharing between diagnostic tasks is overlooked.

FaultGuard-AI solves these problems by achieving end-to-end fault diagnosis through bidirectional interaction and knowledge transfer between anomaly detection and root cause localization. It carefully designs interaction strategies between AD and RCL at the data, feature, and diagnostic result levels, thereby achieving both sub-tasks interactively within a unified, end-to-end multi-task learning framework.

---

## Key Features

- **Interactive Multi-Task Learning:** Enables bidirectional interaction and knowledge sharing between anomaly detection and root cause localization.
- **Cross-Domain Adaptation:** Fully integrated domain-adaptation layers allowing robust models to perform diagnostics across varying system environments.
- **6 Concurrent Diagnostic Tasks:** Scaled up from standard systems to coordinate six distinct diagnostic and classification tasks simultaneously (Multitask Expansion).
- **AI-Driven Weight Automation:** Dynamically balances task weight distributions during training, achieving a **65% to 75% performance gain** over baseline independent systems.

---

## System Architecture

FaultGuard-AI is structured as a parent orchestrator that integrates the core research engine via a tracked Git Submodule (`chimera`):

```
FaultGuard-AI  (Parent Orchestrator)
 ├─ .gitmodules       # Submodule configuration
 ├─ README.md         # Parent master system documentation (This file)
 ├─ LICENSE           # System license
 └─ chimera/          # [Submodule] Core Research Engine (Forked & Optimized)
     ├─ src/          # Multi-task expansion & domain adaptation models
     ├─ scripts/      # Best checkpoint selectors & comparison plotters
     ├─ report_output/# Evaluated summaries, charts, and metrics
     └─ main.py       # Training and evaluation entrypoint
```

---

## Datasets

We utilize two major open-source log datasets for evaluation:

| Software System | Description | Time Span | # Messages | Data Size | Link |
| --- | --- | --- | --- | --- | --- |
| BGL | Blue Gene/L supercomputer log | 214.7 days | 4,747,963 | 708.76MB | [Usenix-CFDR Data](https://www.usenix.org/cfdr-data#hpc4) |
| Thunderbird | Thunderbird supercomputer log | 244 days | 211,212,192 | 27.367 GB | [Usenix-CFDR Data](https://www.usenix.org/cfdr-data#hpc4) |

*Note: Considering the huge scale of the Thunderbird dataset, we follow the settings of the previous study [LogADEmpirical](https://github.com/LogIntelligence/LogADEmpirical) and select the earliest 10 million log messages from the Thunderbird dataset for experimentation.*

---

## Environment

The core engine requires the following key python packages:

- numpy>=1.20.3,<2.0.0
- pandas==1.3.5
- matplotlib
- pytorch_lightning==1.1.2
- scikit-learn>=1.0.2,<1.1.0
- torch>=1.13.1
- tqdm==4.62.3
- overrides
- [Drain3](https://github.com/IBM/Drain3) (Log parsing)

---

## Preparation

To completely run the FaultGuard-AI training and evaluation pipelines:

1. **Step 1:** Download the datasets from the links above and place the log files under the `chimera/data/` folder.
2. **Step 2:** Parse the unstructured logs using Drain3.
3. **Step 3:** Download `glove.6B.300d.txt` from [Stanford NLP word embeddings](https://nlp.stanford.edu/projects/glove/) and place it under the `chimera/glove/` folder.

---

## Quick Start Guide

### 1. Preparing the Environment
Enter the core research directory and install the required dependencies:
```bash
cd chimera
pip install -r requirements.txt
```

### 2. Training the Model
Train the unified model on the BGL dataset for 150 epochs with dynamic task weights and automated validation checkpointing:
```bash
python main.py --mode train --dataset BGL --epochs 150 --batch_size 256
```

### 3. Dynamic Checkpoint Selection
After training is complete, automatically evaluate all saved checkpoints in the history and copy the best performing model to `checkpoint/best_model.bin`:
```bash
python scripts/select_best_checkpoint.py --dataset BGL --device cpu --batch-size 128 --checkpoint-dir checkpoint
```

### 4. Model Evaluation
Evaluate the selected best model on the BGL dataset:
```bash
python main.py --mode eval --load_checkpoint True --dataset BGL
```

### 5. Generate Comparison Plots
Generate the paper-vs-our-run comparison and the auxiliary-head comparison figures:
```bash
python scripts/plot_paper_comparison.py
```

Generated figures are stored under `chimera/report_output/`:
- `fig_bgl_paper_vs_novelties.png`
- `fig_bgl_novelty_auxiliary_heads.png`

---

## Reproducibility and Outputs

The repository contains a full, reproducible, BGL-only research flow with dynamic checkpoint selection and comparison plots.

- **Latest Test Results:** See [chimera/TEST_RESULTS.md](chimera/TEST_RESULTS.md) for the freshly generated evaluation results, plots, and checkpoint selection output from the latest test run.
- **Implemented Novelties:** See [chimera/NOVELTIES.md](chimera/NOVELTIES.md) for the full description of the three implemented novelty modules, their code locations, and BGL results.
- **Command Reference:** For a complete list of runnable commands, modes, and flags, see [chimera/docs/COMMANDS.md](chimera/docs/COMMANDS.md).
- **Main Report:** The complete human-readable summary lives in [chimera/report_output/PROJECT_REPORT.md](chimera/report_output/PROJECT_REPORT.md).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
