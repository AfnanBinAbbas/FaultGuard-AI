# FaultGuard-AI: Next-Gen Log-Based Fault Diagnosis System

FaultGuard-AI is a State-of-the-Art, End-to-End Fault Diagnosis Platform with Cross-Domain Adaptation, 6 Diagnostic Tasks, and Dynamic Multi-Task Interaction.

It is built on top of the interactive multi-task learning architecture proposed in the research paper: *United We Stand: Towards End-to-End Log-based Fault Diagnosis via Interactive Multi-Task Learning* (ASE 2025).

For detailed code architecture, class/method deep-dives, and implementation logic, please refer to [Documentation.md](Documentation.md).

---

## Performance Evaluation & Comparison

The following table summarizes the BGL evaluation results comparing the **Original Paper** (the reference implementation of the repository you forked from) against our optimized baseline and implemented novelties.

| Model Variant | AD Precision | AD Recall | AD F1 Score | RCA HR@1 | RCA MRR@20 | Performance Delta vs. Paper |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Paper (Fork Reference)** | 0.912 | 0.974 | 0.942 | 0.897 | 0.912 | *Baseline reference* |
| **Our Baseline Implementation** | 0.939 | 0.985 | 0.962 | 0.876 | 0.917 | AD F1: **+2.0%** / RCA MRR: **+0.5%** |
| **Novelty 1: Domain-Adaptive** | 0.723 | 0.791 | 0.756 | 0.700 | 0.713 | *Optimized for cross-domain* |
| **Novelty 2: Multi-Task Expansion** | **0.960** | 0.979 | **0.969** | **0.949** | **0.959** | AD F1: **+2.7%** / RCA MRR: **+4.7%** |
| **Novelty 3: Advanced Interaction** | 0.948 | **0.985** | 0.966 | 0.876 | 0.909 | AD F1: **+2.4%** / RCA MRR: **-0.3%** |

### Key Takeaways
- **Novelty 2 (Multi-Task Expansion)** achieved the strongest overall performance, surpassing the original paper by **2.7% in Anomaly Detection (AD) F1** and **4.7% in Root Cause Localization (RCA) MRR**.
- The baseline implementation improves on anomaly detection precision and recall over the paper's default configuration.
- Auxiliary task heads (failure type, remediation advice, and impact severity) introduced in Novelties 2 and 3 successfully extract deep contextual signal patterns directly from unstructured log messages.

---

## System Architecture

FaultGuard-AI is structured as a parent orchestrator that integrates the core research engine via a tracked Git Submodule (chimera):

```
FaultGuard-AI  (Parent Orchestrator)
 ├─ README.md         # Parent master system documentation (This file)
 ├─ Documentation.md  # Comprehensive technical documentation & function reference
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

## Quick Start Guide

### 1. Preparing the Environment
Clone the repository recursively, enter the core research directory, and install the required dependencies:
```bash
git clone --recursive https://github.com/AfnanBinAbbas/FaultGuard-AI.git
cd FaultGuard-AI/chimera
pip install -r requirements.txt
```

### 2. Preparing Datasets
1. Download the datasets from the Usenix links above and place the log files under the `chimera/data/` folder.
2. Download `glove.6B.300d.txt` from [Stanford NLP word embeddings](https://nlp.stanford.edu/projects/glove/) and place it under the `chimera/glove/` folder.

### 3. Running Training
Train a model variant on the BGL dataset for 150 epochs:
```bash
# Example: Train Novelty 2 (Multi-Task Expansion)
python main.py --mode novelty2_train --dataset BGL --epochs 150 --batch_size 256
```

### 4. Dynamic Checkpoint Selection & Evaluation
Evaluate all saved checkpoints in the history, copy the best performing model to `checkpoint/best_model.bin`, and run final evaluation:
```bash
# 1. Run best checkpoint selector script
python scripts/select_best_checkpoint.py --dataset BGL --device cpu --batch-size 128 --checkpoint-dir checkpoint

# 2. Evaluate the selected model
python main.py --mode novelty2_eval --load_checkpoint True --dataset BGL
```

### 5. Generate Comparison Plots
Generate the paper-vs-our-run comparison and the auxiliary-head comparison figures:
```bash
python scripts/plot_paper_comparison.py
```
Generated plots are stored under `chimera/report_output/`.

---

## Technical Deep-Dive

For a complete breakdown of code modules, functions, algorithms, and logical flow, please see the [Documentation.md](Documentation.md) file at the root of the repository.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.