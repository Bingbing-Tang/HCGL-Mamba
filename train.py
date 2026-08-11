"""Training and evaluation for HCGL-Mamba.

The validation split is used for model selection and early stopping. The test
split is evaluated only after loading the best validation checkpoint. Test-time
outputs contain metrics only; predictions and targets are not saved.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from config import Config
from model import HCGLMamba

try:
    from mamba_ssm import Mamba
except ImportError:
    from mamba_ssm.modules.mamba_simple import Mamba


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FeatureStandardScaler:
    """Feature-wise scaler fitted on the training split only."""

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        self.mean = data.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self.std = np.maximum(data.std(axis=(0, 1), keepdims=True).astype(np.float32), 1e-6)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted")
        return ((data - self.mean) / self.std).astype(np.float32)

    def inverse_target(self, value: torch.Tensor, feature: int) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted")
        mean = float(self.mean[..., feature].reshape(-1)[0])
        std = float(self.std[..., feature].reshape(-1)[0])
        return value * std + mean


class TrafficWindowDataset(Dataset):
    """Lazy windows with target-time-based chronological splitting."""

    def __init__(self, data: np.ndarray, input_len: int, pred_len: int, target_start: int, target_end: int,
                 target_feature: int, stride: int = 1) -> None:
        if data.ndim != 3 or input_len < 1 or pred_len < 1 or stride < 1:
            raise ValueError("Invalid data shape or window configuration")
        self.data = torch.from_numpy(data)
        self.input_len, self.pred_len, self.target_feature = input_len, pred_len, target_feature
        first = max(0, target_start - input_len)
        last = target_end - input_len - pred_len
        self.starts = [s for s in range(first, last + 1, stride)
                       if s + input_len >= target_start and s + input_len + pred_len <= target_end]
        if not self.starts:
            raise ValueError(f"No windows for target range [{target_start}, {target_end})")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        s = self.starts[index]
        x = self.data[s:s + self.input_len]
        y = self.data[s + self.input_len:s + self.input_len + self.pred_len, :, self.target_feature:self.target_feature + 1]
        return x, y


def load_raw_data(cfg: Config) -> np.ndarray:
    data = np.load(cfg.data_path)["data"].astype(np.float32)
    if data.ndim != 3 or data.shape[-1] < cfg.raw_feature_dim or not np.isfinite(data).all():
        raise ValueError(f"Invalid traffic data: shape={data.shape}")
    cfg.num_nodes = data.shape[1]
    return data[..., :cfg.raw_feature_dim]


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    adjacency = torch.clamp(adjacency, min=0.0)
    adjacency = adjacency + torch.eye(adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype)
    inv_sqrt = adjacency.sum(dim=-1).clamp_min(1e-8).pow(-0.5)
    return (inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]).contiguous()


def load_graph_bundle(cfg: Config) -> Dict[str, torch.Tensor]:
    path = Path(cfg.graph_dir) / "graph_bundle.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python preprocess.py` first.")
    bundle = np.load(path)
    required = ("A_en_road", "A_func", "A_steiner", "A_semantic", "cluster_membership",
                "outlier_soft_assignment", "outlier_mask", "steiner_node_mask")
    missing = [key for key in required if key not in bundle]
    if missing:
        raise KeyError(f"graph_bundle.npz is missing: {missing}")

    device = cfg.device
    graphs = {
        "a_en_road": torch.as_tensor(bundle["A_en_road"], dtype=torch.float32, device=device),
        "a_func": torch.as_tensor(bundle["A_func"], dtype=torch.float32, device=device),
        "a_steiner": torch.as_tensor(bundle["A_steiner"], dtype=torch.float32, device=device),
        "a_semantic": torch.as_tensor(bundle["A_semantic"], dtype=torch.float32, device=device),
        "normal_membership": torch.as_tensor(bundle["cluster_membership"], dtype=torch.float32, device=device),
        "outlier_soft_assignment": torch.as_tensor(bundle["outlier_soft_assignment"], dtype=torch.float32, device=device),
        "outlier_mask": torch.as_tensor(bundle["outlier_mask"], dtype=torch.bool, device=device),
        "steiner_node_mask": torch.as_tensor(bundle["steiner_node_mask"], dtype=torch.bool, device=device),
    }
    for key in ("a_en_road", "a_func", "a_steiner", "a_semantic"):
        graphs[f"{key}_norm"] = normalize_adjacency(graphs[key])
    cfg.normal_cluster_count = graphs["normal_membership"].shape[1]
    return graphs


def make_loaders(data: np.ndarray, cfg: Config) -> Dict[str, DataLoader]:
    total = data.shape[0]
    train_end = int(total * cfg.train_ratio)
    val_end = int(total * (cfg.train_ratio + cfg.val_ratio))
    ranges = {"train": (cfg.input_len, train_end), "val": (train_end, val_end), "test": (val_end, total)}
    loaders = {}
    for split, (start, end) in ranges.items():
        dataset = TrafficWindowDataset(data, cfg.input_len, cfg.pred_len, start, end, cfg.target_feature, cfg.stride)
        loaders[split] = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=split == "train",
                                    num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
        print(f"{split}: {len(dataset)} windows, {len(loaders[split])} batches")
    return loaders


def cosine_schedule_with_warmup(optimizer: optim.Optimizer, warmup_epochs: int, total_epochs: int,
                                min_lr_ratio: float = 0.01) -> LambdaLR:
    def multiplier(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = float(epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, multiplier)


class MetricAccumulator:
    """Element-weighted MAE and RMSE accumulator."""

    def __init__(self, threshold: Optional[float] = None) -> None:
        self.threshold = threshold
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> None:
        mask = torch.isfinite(prediction) & torch.isfinite(target)
        mask &= target.abs() > 1e-6 if self.threshold is None else target >= self.threshold
        if node_mask is not None:
            mask &= node_mask.view(1, 1, -1, 1)
        if not mask.any():
            return
        error = prediction[mask] - target[mask]
        self.abs_sum += error.abs().sum().item()
        self.sq_sum += error.square().sum().item()
        self.count += int(mask.sum().item())

    def compute(self) -> Tuple[float, float]:
        if self.count == 0:
            return 0.0, 0.0
        return self.abs_sum / self.count, math.sqrt(self.sq_sum / self.count)


def forward_model(model: HCGLMamba, x: torch.Tensor, g: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(x, g["a_en_road_norm"], g["a_func_norm"], g["a_steiner_norm"], g["a_semantic_norm"],
                 g["normal_membership"], g["outlier_soft_assignment"], g["outlier_mask"], g["steiner_node_mask"])


def weighted_huber_loss(raw_loss: torch.Tensor, target: torch.Tensor, outlier_mask: torch.Tensor,
                        outlier_weight: float) -> torch.Tensor:
    if outlier_weight <= 0:
        raise ValueError("outlier_loss_weight must be positive")
    weights = torch.ones_like(raw_loss)
    if outlier_weight != 1.0:
        weights = torch.where(outlier_mask.view(1, 1, -1, 1), torch.full_like(weights, outlier_weight), weights)
    weights *= torch.isfinite(target).to(weights.dtype)
    return (raw_loss * weights).sum() / weights.sum().clamp_min(1.0)


def train_one_epoch(model: HCGLMamba, loader: DataLoader, graphs: Dict[str, torch.Tensor], optimizer,
                    amp_scaler, criterion: nn.Module, cfg: Config) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(cfg.device, non_blocking=True), y.to(cfg.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            prediction = forward_model(model, x, graphs)
            loss = weighted_huber_loss(criterion(prediction, y), y, graphs["outlier_mask"], cfg.outlier_loss_weight)
        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        amp_scaler.step(optimizer)
        amp_scaler.update()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def make_metric_groups(graphs: Dict[str, torch.Tensor], cfg: Config, detailed: bool):
    if not detailed:
        return {"OverallNonzero": (MetricAccumulator(), None)}
    normal, outlier, threshold = ~graphs["outlier_mask"], graphs["outlier_mask"], cfg.metric_threshold
    return {
        "OverallNonzero": (MetricAccumulator(), None),
        f"OverallThreshold{threshold:g}": (MetricAccumulator(threshold), None),
        "NormalNonzero": (MetricAccumulator(), normal),
        f"NormalThreshold{threshold:g}": (MetricAccumulator(threshold), normal),
        "OutlierNonzero": (MetricAccumulator(), outlier),
        f"OutlierThreshold{threshold:g}": (MetricAccumulator(threshold), outlier),
    }


def update_groups(groups, prediction: torch.Tensor, target: torch.Tensor) -> None:
    for accumulator, node_mask in groups.values():
        accumulator.update(prediction, target, node_mask)


def groups_to_metrics(groups) -> Dict[str, float]:
    metrics = {}
    for name, (accumulator, _) in groups.items():
        mae, rmse = accumulator.compute()
        metrics[f"{name}_MAE"] = mae
        metrics[f"{name}_RMSE"] = rmse
    return metrics


@torch.no_grad()
def evaluate(model: HCGLMamba, loader: DataLoader, graphs: Dict[str, torch.Tensor], scaler: FeatureStandardScaler,
             cfg: Config, detailed: bool = False, horizon_steps: Tuple[int, ...] = ()):
    """Evaluate without retaining prediction arrays in memory or on disk."""
    model.eval()
    aggregate = make_metric_groups(graphs, cfg, detailed)
    horizon_groups = {int(step): make_metric_groups(graphs, cfg, detailed) for step in horizon_steps}

    for x, y in loader:
        x, y = x.to(cfg.device, non_blocking=True), y.to(cfg.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=True):
            prediction = forward_model(model, x, graphs)
        prediction = scaler.inverse_target(prediction.float(), cfg.target_feature)
        target = scaler.inverse_target(y.float(), cfg.target_feature)
        update_groups(aggregate, prediction, target)
        for step, groups in horizon_groups.items():
            if step < 1 or step > prediction.shape[1]:
                raise ValueError(f"Invalid evaluation horizon {step} for pred_len={prediction.shape[1]}")
            update_groups(groups, prediction[:, step - 1:step], target[:, step - 1:step])

    return groups_to_metrics(aggregate), {step: groups_to_metrics(groups) for step, groups in horizon_groups.items()}


def select_device(cfg: Config) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("HCGL-Mamba requires a CUDA-enabled PyTorch/Mamba installation")
    if not 0 <= cfg.gpu_id < torch.cuda.device_count():
        raise ValueError(f"gpu_id={cfg.gpu_id} is invalid; visible GPUs={torch.cuda.device_count()}")
    cfg.device = torch.device(f"cuda:{cfg.gpu_id}")
    torch.cuda.set_device(cfg.device)
    print(f"Device: {cfg.device} | {torch.cuda.get_device_name(cfg.device)}")
    return cfg.device


def native_mamba_preflight(cfg: Config) -> None:
    if not cfg.native_mamba_preflight:
        return
    try:
        probe = Mamba(d_model=cfg.mamba_d_model, d_state=cfg.mamba_d_state, d_conv=cfg.mamba_d_conv,
                      expand=cfg.mamba_expand).to(cfg.device)
        x = torch.randn(2, cfg.input_len, cfg.mamba_d_model, device=cfg.device, requires_grad=True)
        with torch.cuda.amp.autocast(enabled=True):
            probe(x).float().square().mean().backward()
        torch.cuda.synchronize(cfg.device)
        print("Native Mamba CUDA preflight: PASS")
    except RuntimeError as exc:
        raise RuntimeError("Native Mamba CUDA preflight failed. Check PyTorch/CUDA, mamba-ssm, and causal-conv1d compatibility.") from exc
    finally:
        if "probe" in locals():
            del probe
        if "x" in locals():
            del x
        torch.cuda.empty_cache()


def save_test_metrics(aggregate: Dict[str, float], horizons: Dict[int, Dict[str, float]], cfg: Config) -> Path:
    rows = [{"HorizonStep": "all", "HorizonMinutes": "all", **aggregate}]
    for step in cfg.evaluation_horizon_steps:
        step = int(step)
        rows.append({"HorizonStep": str(step), "HorizonMinutes": str(step * cfg.sampling_interval_minutes), **horizons[step]})
    path = Path(cfg.output_dir) / "test_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    cfg = Config()
    cfg.prepare_dirs()
    set_seed(cfg.seed)
    select_device(cfg)
    native_mamba_preflight(cfg)

    raw_data = load_raw_data(cfg)
    train_end = int(raw_data.shape[0] * cfg.train_ratio)
    scaler = FeatureStandardScaler()
    scaler.fit(raw_data[:train_end])
    loaders = make_loaders(scaler.transform(raw_data), cfg)
    graphs = load_graph_bundle(cfg)
    if graphs["a_en_road"].shape[0] != raw_data.shape[1]:
        raise ValueError("Graph node count does not match traffic data")

    model = HCGLMamba(cfg).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = cosine_schedule_with_warmup(optimizer, max(5, int(cfg.epochs * 0.1)), cfg.epochs)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.HuberLoss(delta=1.0, reduction="none")
    checkpoint = Path(cfg.output_dir) / "best_model.pt"

    best_mae, patience = float("inf"), 0
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, loaders["train"], graphs, optimizer, amp_scaler, criterion, cfg)
        val_metrics, _ = evaluate(model, loaders["val"], graphs, scaler, cfg)
        val_mae, val_rmse = val_metrics["OverallNonzero_MAE"], val_metrics["OverallNonzero_RMSE"]
        print(f"Epoch {epoch:03d} | Loss {train_loss:.6f} | Val MAE {val_mae:.4f} | RMSE {val_rmse:.4f} | "
              f"LR {optimizer.param_groups[0]['lr']:.6g} | {time.perf_counter() - epoch_start:.1f}s")

        if val_mae < best_mae - cfg.min_delta:
            best_mae, patience = val_mae, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            patience += 1
            if patience >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        scheduler.step()

    if not checkpoint.exists():
        raise RuntimeError("No validation checkpoint was saved")
    model.load_state_dict(torch.load(checkpoint, map_location=cfg.device))
    aggregate, horizons = evaluate(model, loaders["test"], graphs, scaler, cfg, detailed=True,
                                   horizon_steps=cfg.evaluation_horizon_steps)
    metrics_path = save_test_metrics(aggregate, horizons, cfg)

    print(f"\nTraining time: {time.perf_counter() - start:.1f}s")
    print("\n=== Test metrics: all forecast steps ===")
    for key, value in aggregate.items():
        print(f"{key}: {value:.6f}")
    print("\n=== Test metrics: exact forecast steps ===")
    print(pd.read_csv(metrics_path).to_string(index=False))
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
