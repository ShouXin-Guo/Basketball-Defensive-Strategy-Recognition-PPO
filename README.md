# Defensive Strategy Recognition in Basketball Using Parallel 1D-CNNs and PPO

Research implementation of the methodology described in *Defensive Strategy Recognition in Basketball Using Parallel 1D-CNNs and Proximal Policy Optimization*, by Yuqing Xia (supplied manuscript).

## Overview

The system classifies a pre-identified basketball defensive event as **Switch (0)** or **Trap (1)**. Four parallel one-dimensional convolutional branches learn representations of eleven basketball movement/spatial features. An actor–critic policy is trained with clipped proximal policy optimization (PPO). Incorrect Trap predictions end a training episode early, and rewards emphasize Trap compared with the majority Switch class. The training/test protocol groups examples by game to avoid cross-game information leakage.

**Reproducibility scope:** This repository implements the model and training protocol for **annotated defensive events**. The supplied manuscript does not completely specify how raw player trajectories are converted into event labels or how the four player movement vectors are reduced to scalar features. Those upstream steps require the full labeling algorithm from Tian et al. (2019), expert annotation/consensus where available, and a documented event-feature mapping. This code does **not** invent labels, reproduce expert re-annotation, or claim to have reproduced the published results. An event-level adapter supporting clearly defined input arrays is provided; raw SportVU JSON is not treated as automatically labeled data.

## Key features

- Four configurable independent Conv1D branches, each with 32/64/128 filters, kernel size 3, same padding, ReLU by default, max-pooling after convolution stages 1 and 2, global average pooling, and 128 output units per branch. Concatenating four branches yields a 512-dimensional representation.
- Configurable actor/policy multilayer perceptron, separate critic/value multilayer perceptron, categorical 2-class actions, clipped PPO surrogate, generalized advantage estimation (GAE), value-function loss and entropy regularization.
- Minority-aware immediate rewards: correct/incorrect Trap = +1/−1; correct/incorrect Switch = +λ/−λ. A wrong Trap prediction terminates the episode; otherwise training proceeds to the next shuffled example.
- Grouped train/hold-out split, grouped stratified five-fold cross-validation during optional hyperparameter search, independent training seeds, train-only scaling, checkpoints, metrics, and per-run predictions.
- Ablation configurations for removing the CNN encoder (`no_cnn`) or replacing PPO with ordinary cross-entropy training (`no_ppo`). These are operational ablations, not claims of numerical equivalence to every undocumented implementation choice in the manuscript.

## Scientific methodology

### Feature representation

Each defensive event contains eleven ordered features. F1–F4 summarize motion of O1, O2, D1 and D2, whose defensive relationships change. F5 measures the distance from their midpoint to the basket. F6–F8 are distances from O3, O4 and O5 to that midpoint; F9–F11 represent D3, D4 and D5. The manuscript does not specify the scalar encoding of a movement **vector**, sampling window length, or the raw-to-event preprocessing code. Therefore, feature generation and annotation must be completed separately, with a consistent definition across train and test sets.

In the supplied architecture, each CNN processes the same ordered 11-feature signal, but uses independently initialized weights. For optional per-event sequences, the current implementation averages the input sequence over time before passing the 11-feature vector to the feature-axis convolutions. This is an **explicit implementation convention**, not a claim that the manuscript uses this exact temporal aggregation. When temporal information is critical, provide an agreed event-level transformation, or extend the encoder with evidence from the study authors.

### Reward, policy and advantage estimation

With label `y`, action `a`, minority Trap=1 and reward weight `λ`:

```text
r(a,y) = +1   if a = y = Trap
       = -1   if a != y and y = Trap
       = +λ   if a = y = Switch
       = -λ   if a != y and y = Switch
```

The episode terminates on a wrong Trap action or at the final example. The actor and critic share trainable CNN features. PPO maximizes the clipped surrogate `min(ratio*advantage, clip(ratio,1-eps,1+eps)*advantage)`, with a critic mean-square-error term and entropy regularization. Rewards and action log-probabilities are collected under the frozen rollout policy; each rollout is then reused for the configured number of PPO minibatch epochs. GAE(λ) computes advantages with done-masked bootstrapping. Gradient norm clipping is used. The entropy coefficient, critic weight, gradient norm, and feature encoding are implementation-selected because the paper does not report their values.

### Framework workflow

```text
Annotated event CSV/NPZ with game IDs
    -> load/validate classes and F1..F11
    -> game-disjoint development/test partition
       -> optional grouped, stratified five-fold CV over development games
            -> fit scaler on each fold's training examples only
            -> search candidate hyperparameters
            -> select best average macro-F1
    -> fit final scaler on all development examples
    -> train actor–critic (CNN features -> PPO action/reward -> update)
       OR no-PPO supervised ablation
    -> evaluate unchanged hold-out events
    -> save metrics, predictions, history, checkpoint, settings
```

## Datasets

The paper describes NBA SportVU tracking (NST) for the 2012–2013 NBA season and an external NST-15 dataset from the 2015–2016 season. The original study says the tracking files encode player/ball positions at 25 Hz, about 50–300 frames per possession, and that it retained 32,377 possessions and identified 42,865 labeled events. The labels were derived with the Tian et al. analytical model and then validated through expert consensus. **Raw possession files alone are not sufficient to execute the event classifier in this repository.**

### Data Availability

The source manuscript provides a public NBA movement data repository URL fragment, `https://github.com/sealneaward/`, in its *Availability of data and materials* section. The manuscript text visible in the supplied PDF does not specify an unambiguous repository suffix, release, or link to its expert-corrected event labels; do not assume the annotated event table is hosted there. Verify the availability and license of tracking files with the original publisher. This repository does not redistribute player tracking data.

**Preparing datasets:** Obtain the appropriate tracking data and the event annotation/feature extraction procedure described by Tian et al. (2019), *Use of Machine Learning to Automate the Identification of Basketball Strategies Using Whole Team Player Tracking Data* (Applied Sciences 10(1), 24), and ideally the study-specific validated event annotations. Convert **each labeled event** into eleven features and a game identifier, then export as described below. Where expert-validated labels are unavailable, preserve their provenance; results are not directly comparable to the paper's expert-corrected dataset. Different seasons may be stored in separate input files and evaluated independently.

## Data Preparation

### Supported input 1: event-level CSV

Each row describes **one annotated defensive event**. Required columns are:

```text
F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,label,game_id
0.1,0.4,0.3,0.7,5.0,3.2,2.3,4.0,1.9,2.1,3.0,Switch,game_001
```

CSV stores the eleven scalar event summary features and has no explicit temporal axis. `label` is case-insensitive `Switch`/`Trap` (or 0/1). `game_id` groups all events from the same game; never use possession IDs in place of game IDs when replicating the main split.

### Supported input 2: event-level NPZ

A NumPy archive must contain `x` (float array shaped `(number_of_events, frames, 11)` or `(number_of_events, 11)`), `y` (Switch/Trap or 0/1), and `game_id` (game identifier for each event). It may be supplied directly:

```bash
python main.py --data data/nst_annotated.npz --config trained_configurations/initial.json
```

The implementation temporally averages `frames` (if greater than one) before CNN processing. Required features are numeric and finite; invalid shapes, missing classes and missing files raise explicit errors. All preprocessing standardization is fitted on **training examples only**, including inside cross-validation; hold-out events never influence the scaler. No synthetic data augmentation, imputation, or unsupervised pseudo-labeling is introduced.

Expected local project layout (input files are supplied by the user, not included):

```text
data/
  nst_annotated.csv
  nst15_annotated.npz
```

These are examples of acceptable paths, not files supplied in the archive.

## Repository structure

```text
basketball_ppo_repo/
├── main.py                      # CLI entry point
├── data.py                      # Annotated event loading, grouped splits, scaling
├── models.py                    # Parallel Conv1D and actor–critic networks
├── ppo.py                       # Rollouts, GAE, clipped PPO and no-PPO ablation
├── config.py                    # Schema validation, JSON saving and loading
├── experiment.py                # Cross-validation, optional search, training, checkpoints
├── evaluation.py                # Classification metrics and paired game bootstrap helper
├── requirements.txt            # Runtime dependencies
├── README.md                   # This documentation
├── trained_configurations/
│   └── initial.json            # Valid initialized configuration; no learned weights
└── tests/
    └── test_repository.py      # Independent smoke and configuration tests
```

Runtime files are written under `outputs/` by default (not checked into the repository).

## Requirements

The paper reports Windows 11 (64-bit), Python 3.11, PyTorch 2.6, NumPy 2.1, pandas 2.2, scikit-learn 1.6, CUDA 12.4 and NVIDIA RTX A6000 (48GB), with 64GB system RAM. These are **reported research conditions**, not strict minimum requirements for small-scale execution. Current implementation supports CPU or an available PyTorch CUDA installation. For GPU use, install the PyTorch build compatible with your driver/CUDA environment, using the official PyTorch installation instructions.

## Installation

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py --mode list-configs
```

## Quick Start

After preparing real annotated events according to the above specification:

```bash
python main.py --mode train --data data/nst_annotated.csv --config trained_configurations/initial.json --output outputs/nst --device auto
```

## Basic Usage

Train five independent seeds using Table 1 initialized settings:

```bash
python main.py --mode train --data data/nst_annotated.csv --five-seeds --output outputs/nst_5runs
```

Optimize candidate settings with game-disjoint five-fold CV (can be computationally expensive):

```bash
python main.py --mode optimize --data data/nst_annotated.csv --config trained_configurations/initial.json --output outputs/search
```

Evaluate a trained checkpoint on the corresponding dataset and deterministic hold-out split:

```bash
python main.py --mode evaluate --data data/nst_annotated.csv --checkpoint outputs/nst/model_seed_42.pt --output outputs/rechecked
```

List and validate all installed configurations:

```bash
python main.py --mode list-configs --config-dir trained_configurations
```

## Command-line options

| Option | Meaning | Default |
|---|---|---|
| `--mode` | `train`, `optimize`, `evaluate`, `list-configs` | `train` |
| `--data` | Required annotated CSV or NPZ except `list-configs` | None |
| `--config` | Configuration to initialize a training or search run | `trained_configurations/initial.json` |
| `--config-dir` | Config directory for discovery | `trained_configurations` |
| `--output` | Run artifact directory | `outputs/run` |
| `--device` | `cpu`, `cuda`, or `auto` | `auto` |
| `--checkpoint` | Required path for `evaluate` | None |
| `--five-seeds` | Repeat training with `42,123,2024,3407,2026` | Disabled |

## Hyperparameter Optimization

Table 1 provides the search boundaries, but the paper does not specify **which search algorithm** selected candidates. The implemented `optimize` mode uses an explicit, seeded uniform/random search for the reported parameters, with `search_trials=12` as an implementation-selected default, and stratified grouped five-fold macro-F1 as its selection criterion. For full-scale publication reproduction, adjust the search budget and verify the original trial selection procedure with the study authors. Candidate episodes are in 32–1024; branch counts 1–5; policy depth 2–64 and width 16–1024; value depth 1–5, width 8–128; learning rate 1e−5 to 1e−3; λ between 0 and 1; clipping 0.05–0.30; gamma 0.90–0.999; GAE λ 0.90–1.00; PPO epochs 1–20; minibatch 8–128; activation and optimizer are sampled from Table 1. The manuscript also lists other advantage estimation estimators, while the current implementation is specifically GAE-based as selected in its Table 1. The selected Table 1 values in `initial.json` are reference initialization settings; **they are not output of search in this repository**.

## Saving and Loading Configurations

A valid JSON configuration is included under `trained_configurations/initial.json` (metadata `status=initialized`). It contains the full validated settings schema but no weights. The `config.py` module supports independent discovery, loading, and non-overwriting save; experimental runs write `outputs/<run>/trained_settings.json`, while search mode also writes `optimized.json`. Status `trained` is applied only after a successful training call. A checkpoint `.pt` contains model parameters and training-data scaler statistics; a JSON settings file **is not** a trained model. To resume a training run, a separate optimizer-state checkpoint and an explicit resume routine would be required; this repository does **not** claim to support optimizer-state-resumable training. It does support loading a trained checkpoint for evaluation.

Executable Python API usage:

```python
from config import Settings, save_config, load_config, list_configurations

cfg = load_config('trained_configurations/initial.json')
cfg.reward_lambda = 0.3
save_config(cfg, 'trained_configurations/alternate.json')
loaded = load_config('trained_configurations/alternate.json')
all_configs = list_configurations('trained_configurations')
print(loaded.reward_lambda, list(all_configs))
```

When using `--config`, the file is loaded and schema/ranges are checked. Saved optimized settings can initialize a new run by passing `--config outputs/search/optimized.json`. An absent checkpoint will trigger a runtime error when attempting evaluation; arbitrary initialized model weights cannot substitute for learned weights. The settings file and checkpoint are separate durable artifacts. Existing custom configuration filenames are not overwritten by `save_config` without explicit `overwrite=True`.

## Training and Evaluation

The code uses a game-level group-shuffle hold-out with 20% of games (not an exact quota of instances), and stratified grouped five-fold cross-validation on the remaining games during tuning. Therefore the exact 34,368/8,497 sample split reported in the paper is **not guaranteed** without the original game assignments. At each selected setting, final model training starts from fresh weights. The five optional seeds match the manuscript. Outputs include overall accuracy, macro F1, Trap F1, G-means as geometric mean of class recalls, AUC based on Trap probabilities, and the confusion matrix. The manuscript does not explicitly identify whether its principal F-measure is macro-, weighted-, or positive-class averaging; this repository makes the macro average explicit rather than claiming numerical identity. A helper `paired_game_bootstrap` in `evaluation.py` supports percentile intervals for paired model comparisons when matched prediction arrays exist; a full published significance table, matched baselines, Holm corrections, and expert-label agreement cannot be reconstructed from the available input alone.

## Outputs

```text
outputs/<run>/
    model_seed_42.pt          # Actual trained state dict, settings, scaler, and seed
    history_seed_42.json     # Actual per-episode rewards and lengths (or supervised loss)
    metrics.json             # Model performance computed on held-out data
    test_predictions.npz     # True labels, predictions for each seed, and game IDs
    trained_settings.json    # Valid settings status='trained'
    search_results.json      # Optional: per-candidate CV values
    optimized.json           # Optional: selected CV configuration
    checkpoint_metrics.json  # Optional: results from evaluate mode
```

No benchmark numbers or pretrained weights are bundled. Testing without annotated real data proves functionality, not result replication.

## Reproducibility

Random seeds are propagated to NumPy and PyTorch during training. CUDA kernel reproducibility can vary by hardware and backend and is not asserted to be perfectly deterministic. Group partitioning is seeded and repeated seeds share the same test set. Training-only scalers prevent cross-partition normalization leakage. A search is scored exclusively on grouped training/validation folds; the independent hold-out is not used to choose hyperparameters. Configurations, per-run histories and actual checkpoint weights are written to disk. Experiments may be expensive with 200 training episodes and many grouped CV trials; reduce episode count only for diagnostic checks and report those runs separately from the manuscript protocol.

## Computational considerations and limitations

1. Processing the entire set of candidate configurations under five grouped folds can be resource-intensive, especially at large network depths; `--mode train` avoids the search stage.
2. The paper does not provide expert-corrected event tables, complete event-extraction code, exact game assignments, the policy/critic's layerwise implementation, all optimizer hyperparameters, or exact F-score averaging. Defaults chosen for those items are transparently documented.
3. The paper reports a preprocessing step mapping player coordinates to a common 400 × 360 court grid and detecting switches/traps from consecutive five-frame segments. **This code starts downstream of that stage** and expects its outputs; it must not be misrepresented as a raw SportVU possession parser.
4. No out-of-sample external NST-15 results, baseline numerical comparisons, or statistical significance are claimed without the relevant labeled data and matched baseline predictions.
5. Source-derived hyperparameters are not learned results. Configuration metadata distinguishes initialization from completed runs.

## Testing

From the repository directory run:

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```

The unit tests construct a tiny **test fixture solely to validate software mechanics**, not a substitute for the real experimental datasets. Checkpoints are restored and re-evaluated on the identical group-based hold-out. Full NST experimental evaluation requires acquisition and validation of annotated basketball events.

## References and citation

- Xia, Y. *Defensive Strategy Recognition in Basketball Using Parallel 1D-CNNs and Proximal Policy Optimization*. Manuscript supplied by the user.
- Tian, C., De Silva, V., Caine, M., & Swanson, S. (2019). *Use of Machine Learning to Automate the Identification of Basketball Strategies Using Whole Team Player Tracking Data*. Applied Sciences, 10(1), 24.
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.

## Licensing and ethics

No license is inferred from the supplied manuscript or from any third-party NBA data. Obtain permission as needed before redistributing data, expert-generated labels, or software derived from proprietary resources.
