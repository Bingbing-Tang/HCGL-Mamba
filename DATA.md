# Data Preparation

The public code uses a unified input format for traffic datasets. Dataset files are **not included** in this repository. The PEMS03, PEMS04, PEMS07, and PEMS08 benchmark datasets used in our experiments can be downloaded from the following public repository:

**Dataset source:**  
https://github.com/guoshnBJTU/ASTGNN/tree/main/data

After downloading, please place the corresponding dataset files locally under the `data/` directory following the structure described below.

## Expected directory structure

```text
data/
└── PEMS04/
    ├── PEMS04.npz
    └── PEMS04_adjacency_matrix.csv
```

The default paths are defined in `config.py` and can be changed for another dataset.

## Traffic data

`PEMS04.npz` must contain an array under the key `data`:

```text
data.shape = (T, N, F)
```

where:

- `T`: number of time steps;
- `N`: number of traffic nodes/sensors;
- `F`: number of raw traffic features.

The default configuration uses the first three raw features and predicts feature index `0`.

A minimal check is:

```python
import numpy as np
x = np.load("data/PEMS04/PEMS04.npz")["data"]
print(x.shape, x.dtype)
```

## Road adjacency matrix

`PEMS04_adjacency_matrix.csv` must store an `N × N` weighted adjacency matrix. The current loader reads the first CSV column as the row index:

```python
pd.read_csv(path, index_col=0)
```

Therefore, the remaining numeric block must be exactly `N × N` and aligned with the node order in the traffic tensor.

## Chronological split and leakage control

The implementation uses a chronological `6:2:2` split for training, validation, and testing. The feature scaler is fitted only on the training period. All clustering, outlier detection, and graph-prior construction also use the training period only. Validation data are used for model selection and early stopping; test data are evaluated only after the best validation checkpoint is selected.

## Preprocessing output

Running `preprocess.py` creates:

```text
artifacts/PEMS04/
├── graph_bundle.npz
└── preprocess_summary.json
```

`graph_bundle.npz` contains only the tensors required by training:

- `A_en_road`: Trend-Enhanced Road Graph;
- `A_func`: Intra-Cluster Functional Graph;
- `A_semantic`: Inter-Cluster Semantic Graph;
- `A_steiner`: Steiner Subgraph;
- `cluster_membership`;
- `outlier_soft_assignment`;
- `outlier_mask`;
- `steiner_node_mask`.

Large raw DTW matrices, prediction arrays, and target arrays are intentionally not stored.

