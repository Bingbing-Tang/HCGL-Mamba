# HCGL-Mamba

Official implementation of **HCGL-Mamba: Hierarchical Cluster-Guided Graph Learning with Bidirectional Mamba for Long-Term Traffic Flow Forecasting**.

HCGL-Mamba uses traffic-pattern clustering as a structural prior while preserving the complete road topology. It constructs complementary spatial relations for non-outlier and behavior-deviating nodes and combines them with a Mamba-based temporal encoder for long-term traffic forecasting.

## Repository structure

```text
HCGL-Mamba/
├── config.py          # Dataset, graph, model, and training configuration
├── preprocess.py      # Training-only clustering and graph construction
├── model.py           # HCGL-Mamba architecture
├── train.py           # Training, validation, and test evaluation
├── DATA.md            # Dataset format and preparation
├── requirements.txt   # Core dependencies
└── .gitignore         # Excludes datasets, checkpoints, and generated artifacts
```

## Implementation principles

- **No validation/test leakage:** graph priors and normalization statistics are constructed from the training split only.
- **Chronological evaluation:** data are split in temporal order with a default ratio of `6:2:2`.
- **Fair model selection:** early stopping uses validation MAE; the test set is evaluated only after the best validation checkpoint is loaded.
- **Compact test output:** only MAE/RMSE metrics are saved. Prediction and target arrays are not written to disk.

## Environment

The supplied environment snapshot used the following key versions:

```text
PyTorch          2.1.2+cu118
CUDA             11.8 (PyTorch build)
NumPy            1.26.4
Pandas           2.3.3
NetworkX         3.4.2
scikit-learn     1.7.2
scikit-learn-extra 0.3.0
tslearn          0.8.1
causal-conv1d    1.2.0.post1
mamba-ssm        1.2.0.post1
```

Because `mamba-ssm` and `causal-conv1d` contain CUDA extensions, install a PyTorch build compatible with your CUDA environment first. The training script includes a native-Mamba CUDA forward/backward preflight check and fails explicitly when the installed kernels are incompatible.

Install the remaining dependencies with:

```bash
pip install -r requirements.txt
```

## Data

Datasets are not committed to the repository. See [DATA.md](DATA.md) for the expected file format and directory structure.

Default PEMS04 layout:

```text
data/PEMS04/PEMS04.npz
data/PEMS04/PEMS04_adjacency_matrix.csv
```

## Usage

### 1. Configure paths and hyperparameters

Edit `config.py` if your dataset location or experimental settings differ from the defaults.

### 2. Build graph priors

```bash
python preprocess.py
```

This creates only the graph bundle required for training and a compact preprocessing summary under `artifacts/PEMS04/`.

### 3. Train and evaluate

```bash
python train.py
```

The best checkpoint is selected by validation MAE. After training, the test split is evaluated once using the selected checkpoint.

## Outputs

```text
artifacts/PEMS04/
├── graph_bundle.npz
└── preprocess_summary.json

results/PEMS04/
├── best_model.pt
└── test_metrics.csv
```

`test_metrics.csv` contains aggregate and exact-horizon MAE/RMSE values. No test prediction arrays or target arrays are saved.

## Evaluation horizons

The default PEMS04 configuration uses 5-minute sampling and evaluates exact forecast steps:

```text
3, 6, 12, 24, 48, 96
```

These correspond to 15, 30, 60, 120, 240, and 480 minutes, respectively. Modify `evaluation_horizon_steps` and `sampling_interval_minutes` in `config.py` when using a different protocol.

## Notes for reproducibility

1. Run preprocessing separately for each dataset/configuration because the graph bundle depends on the training-period clustering parameters.
2. Keep node ordering identical between the traffic tensor and adjacency matrix.
3. Use the same data split, seed, and evaluation horizon definitions when comparing against baselines.
4. Do not commit local datasets, generated graph bundles, checkpoints, or result files; they are excluded by `.gitignore`.

## Citation

Citation information will be added after publication.
