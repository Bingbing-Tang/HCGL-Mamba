# Public-Code Cleanup Summary

This repository is a compact public-release refactor of the supplied HCGL-Mamba implementation. The forecasting method and experimental protocol are preserved; the changes focus on portability, evaluation hygiene, and repository clarity.

## Evaluation/output cleanup

- Removed all MAPE computation and reporting.
- Removed saving of full test predictions and targets.
- Removed saving of per-horizon prediction/target arrays.
- Exact-horizon MAE/RMSE are accumulated online during evaluation, so test arrays do not need to be retained in memory.
- Test output is reduced to a single `test_metrics.csv` file.
- Training-history CSV, scaler dumps, extra test CSVs, and file-based training logs were removed from the default public workflow.
- The best validation checkpoint is still saved as `best_model.pt`.

## Preprocessing cleanup

- `graph_bundle.npz` now stores only tensors required by training.
- Large intermediate DTW matrices and redundant metadata tables are no longer saved.
- A compact `preprocess_summary.json` records the essential preprocessing statistics.

## Reproducibility/fairness

- Chronological train/validation/test splitting is preserved.
- Normalization is fitted on the training period only.
- Clustering, outlier detection, and graph-prior construction use the training period only.
- Early stopping uses validation MAE.
- Test evaluation is performed only after loading the best validation checkpoint.
- Random seeds and deterministic cuDNN settings are retained.

## Code organization

- Renamed public modules to `config.py`, `preprocess.py`, `model.py`, and `train.py`.
- Replaced private absolute paths with repository-relative defaults.
- Reduced unnecessary line breaks, repeated comments, and duplicated bookkeeping.
- Kept clear module/function boundaries rather than aggressively compressing independent logic.
- Added shape/consistency checks at graph and data boundaries.

## Environment cleanup

The supplied environment file included machine-local wheel URLs for `causal-conv1d` and `mamba-ssm`. Those paths are not portable, so the public `requirements.txt` uses package/version declarations and the README explains that PyTorch/CUDA/Mamba binaries must be mutually compatible.

## Files intentionally excluded from Git

The `.gitignore` excludes local datasets, graph bundles, checkpoints, and generated results. This keeps the Git repository source-focused and avoids accidentally committing large binary files.
