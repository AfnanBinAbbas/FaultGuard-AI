# FaultGuard-AI System Architecture & Documentation

This document provides a comprehensive technical walkthrough of the core code modules, classes, methods, and algorithmic logic implemented in FaultGuard-AI. It explains what we achieve, how we achieve it, and lists all available CLI commands.

---

## 1. Core Architecture & Novelty Modules

### Multi-Task Expansion (`src/multitask_expansion.py`)

#### What it Achieves
Expands the baseline system (which only performed binary Anomaly Detection and Root Cause Localization) into a unified, 6-task framework. By introducing extra auxiliary supervision, the shared model embeddings are forced to capture richer structural and contextual patterns from log sequences.

#### Logical Class: `MultiTaskChimera` (extends `pl.LightningModule`)

- `__init__(self, cfg, embedding_dict, input_size=300, hidden_size=128)`
  - **What it does:** Initializes the model, builds GloVe embedding layers, shared LSTM encoders, and sets up 5 task-specific projection heads:
    1. Anomaly detection classification head (`anomaly_head`)
    2. Root cause localization classification head (`rca_head`)
    3. Failure type classification head (`failure_head`)
    4. Remediation recommendation head (`remediation_head`)
    5. Impact severity regression head (`impact_head`)
  - **How it achieves it:** Leverages PyTorch neural network layers (`nn.Linear`, `nn.Dropout`, `nn.LSTM`) mapped from hyperparameters.

- `deal_batch(self, src, ad_label, rca_label)`
  - **What it does:** Pads and slices incoming variable-length sequence batches to match PyTorch tensor sizes.
  - **How it achieves it:** Calculates the maximum sequence length, constructs batch tensors, and masks out out-of-bounds pad values.

- `encode_streams(self, bag_src)`
  - **What it does:** Encodes sequences into rich hidden feature vectors.
  - **How it achieves it:** Passes GloVe embedded vectors through a bidirectional LSTM to obtain a shared sequence representation.

- `pool_sequence(self, sequence_hidden)`
  - **What it does:** Pools sequence representations over time to construct single, static context vectors.
  - **How it achieves it:** Applies maximum and average pooling layers concatenated together to preserve temporal features.

- `forward(self, batch)`
  - **What it does:** Conducts the forward pass, calculating predictions across all 5 tasks and computing their joint optimization loss.
  - **How it achieves it:** Passes the pooled representations through task heads, calculates individual losses, and combines them using dynamic task weighting.
  - **Joint Loss Formula:**
    $$Loss_{total} = Loss_{AD} + w_1 \cdot Loss_{RCA} + w_2 \cdot Loss_{Failure} + w_3 \cdot Loss_{Remediation} + w_4 \cdot Loss_{Impact}$$
    Task weights ($w_i$) are updated dynamically during training based on relative task learning rates.

---

### Domain Adaptation (`src/domain_adaptation.py`)

#### What it Achieves
Mitigates domain-specific noise and improves model generalization across varying system domains (e.g. training on BGL logs and deploying on Thunderbird or GAIA).

#### Logical Class: `GradientReversalFunction` (extends `torch.autograd.Function`)

- `forward(ctx, inputs, lambda_)`
  - **What it does:** Acts as a standard identity mapping in the forward pass.
- `backward(ctx, grad_output)`
  - **What it does:** Negates backpropagated gradients by multiplying them with $- \lambda_$.
  - **How it achieves it:** Intercepts backward gradients in the PyTorch computational graph to force the encoder to learn domain-invariant features.

#### Logical Class: `DomainAdaptiveChimera` (extends `MultiTaskChimera`)

- `__init__(self, cfg, embedding_dict, input_size=300, hidden_size=128)`
  - **What it does:** Initializes a domain discriminator (`domain_discriminator`) consisting of fully-connected linear projection layers.
- `forward(self, batch)`
  - **What it does:** Computes task losses and domain adversarial loss.
  - **How it achieves it:** Feeds encoded shared features through the `GradientReversalFunction` into the domain discriminator. Domain labels are classified via Cross-Entropy Loss to minimize domain predictability while maximizing task diagnostic accuracy. Includes a scheduling factor where the domain loss weight is warmed up from `0.0` to its maximum over 50 epochs to prevent early training instability.

---

### Advanced Interaction (`src/advanced_interaction.py`)

#### What it Achieves
Establishes rich coupling between anomaly detection and root cause localization, allowing classification signals to actively influence and focus feature maps.

#### Logical Class: `AdvancedInteractionChimera` (extends `MultiTaskChimera`)

- `__init__(self, cfg, embedding_dict, input_size=300, hidden_size=128)`
  - **What it does:** Initializes dynamic dual-attention gates (`ad_gate` and `rca_gate`) to link diagnostic branches.
- `apply_attention_gate(self, features, gate_source)`
  - **What it does:** Uses a sigmoid-gated attention scaling function.
  - **How it achieves it:** Computes scaling weights:
    $$Features_{gated} = Features \otimes \sigma(W \cdot Gate_{source} + b)$$
    This modulates intermediate feature weights based on task-specific diagnostics.
- `forward(self, batch)`
  - **What it does:** Runs the forward pass using bidirectional attention scaling.
  - **How it achieves it:** Extracts features from the anomaly detection branch to gate the input to the root cause localization head, and vice-versa, focusing the model's capacity on anomalous regions.

---

## 2. Orchestration & Scripts

### Orchestrator Entrypoint (`chimera/main.py`)

The main entrypoint manages command line argument parsing, training environments, logging, and model mode execution.

- `main()`
  - **What it does:** Orchestrates the entire pipeline:
    - Parses command-line flags.
    - Selects the appropriate model subclass (`MultiTaskChimera`, `DomainAdaptiveChimera`, or `AdvancedInteractionChimera`) depending on the selected mode.
    - Loads pre-trained GloVe embedding dictionaries.
    - Prepares dataset splits (BGL or Thunderbird).
    - Configures optimization schedulers and monitors validation anomaly-detection F1 (`val_ad_f1`) to trigger automated checkpointing of the best weights.

---

### Checkpoint Selector (`chimera/scripts/select_best_checkpoint.py`)

#### What it Achieves
Saves you from having to guess or manually search for the best model weights. Automatically parses your training logs to isolate the single strongest checkpoint.

#### Logic & Method
- Evaluates every saved checkpoint file inside the `checkpoint/` directory on validation datasets.
- Ranks checkpoints using a balanced score combining Anomaly Detection F1 and Root Cause MRR.
- Automatically copies the top-performing checkpoint to `checkpoint/best_model.bin` to be loaded in the final production deployment.

---

### Comparison Plotter (`chimera/scripts/plot_paper_comparison.py`)

#### What it Achieves
Generates beautiful comparison charts comparing the original research paper metrics against your baseline and novelty runs.

#### Logic & Method
- Loads JSON records from `report_output/paper_comparison_metrics.json`.
- Compiles a dataset matrix for F1, Recall, Precision, HR@1, and MRR.
- Uses Matplotlib to render side-by-side comparison charts stored at:
  - `report_output/fig_bgl_paper_vs_novelties.png` (anomaly detection and root cause metrics)
  - `report_output/fig_bgl_novelty_auxiliary_heads.png` (extra auxiliary task metrics)

---

## 3. CLI Command Reference

Execute all training, checkpoint selection, plotting, and evaluation scripts directly from `chimera/`:

### Training Pipelines

```bash
# Train the Multi-Task Expansion model (Novelty 2)
python main.py --mode novelty2_train --dataset BGL --epochs 150 --batch_size 256

# Train the Domain-Adaptive model (Novelty 1)
python main.py --mode novelty1_train --dataset BGL --epochs 150 --batch_size 256

# Train the Advanced Interaction model (Novelty 3)
python main.py --mode novelty3_train --dataset BGL --epochs 150 --batch_size 256
```

### Evaluation Pipelines

```bash
# Evaluate the best selected model (Novelty 2)
python main.py --mode novelty2_eval --dataset BGL --load_checkpoint True

# Evaluate the Domain-Adaptive model (Novelty 1)
python main.py --mode novelty1_eval --dataset BGL --load_checkpoint True

# Evaluate the Advanced Interaction model (Novelty 3)
python main.py --mode novelty3_eval --dataset BGL --load_checkpoint True
```

### Automation & Plotting

```bash
# 1. Run dynamic checkpoint selection
python scripts/select_best_checkpoint.py --dataset BGL --device cpu --batch-size 128 --checkpoint-dir checkpoint

# 2. Replot BGL-only paper-vs-ours comparison figures
python scripts/plot_paper_comparison.py
```
