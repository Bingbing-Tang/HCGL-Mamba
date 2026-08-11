from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Config:
    """HCGL-Mamba configuration for the public implementation."""

    # Paths
    data_path: str = "data/PEMS04/PEMS04.npz"
    adj_path: str = "data/PEMS04/PEMS04_adjacency_matrix.csv"
    graph_dir: str = "artifacts/PEMS04"
    output_dir: str = "results/PEMS04"

    # Data
    num_nodes: int = 307
    raw_feature_dim: int = 3
    target_feature: int = 0
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    input_len: int = 96
    pred_len: int = 96
    stride: int = 1
    sampling_interval_minutes: int = 5
    evaluation_horizon_steps: Tuple[int, ...] = (3, 6, 12, 24, 48, 96)
    steps_per_day: int = 288

    # Cluster-guided graph construction
    normal_cluster_count: int = 5
    outlier_percentile: float = 95.0
    sample_weeks: int = 7
    downsample_factor: int = 6
    cluster_seed: int = 42
    cluster_standardize: bool = True
    road_enhance_lambda: float = 0.5
    functional_topk: int = 6
    functional_nonphysical_only: bool = True
    semantic_temperature: float = 1.0
    outlier_assignment_temperature: float = 1.0
    steiner_use_trend_cost: bool = True

    # Spatial encoder
    spatial_dim: int = 64
    spatial_dropout: float = 0.1
    graph_layers: int = 2
    normal_gate_init_road: float = 0.3
    normal_gate_init_func: float = 0.4
    normal_gate_init_semantic: float = 0.3
    outlier_gate_init_road: float = 0.6
    outlier_gate_init_steiner: float = 0.2
    outlier_gate_init_semantic: float = 0.2

    # Temporal encoder
    mamba_d_model: int = 64
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    temporal_dropout: float = 0.1
    bidirectional_mamba: bool = True

    # Optimization
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 30
    min_delta: float = 1e-4
    grad_clip: float = 5.0
    num_workers: int = 0
    seed: int = 42
    outlier_loss_weight: float = 1.0
    metric_threshold: float = 5.0

    # Runtime
    gpu_id: int = 0
    native_mamba_preflight: bool = True
    device: Optional[object] = None

    def prepare_dirs(self) -> None:
        Path(self.graph_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
