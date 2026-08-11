"""Training-only graph construction for HCGL-Mamba.

All clustering, outlier detection, and graph priors are derived from the training
split only. Clustering acts as a structural prior; the original road graph is
never partitioned.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from networkx.algorithms.approximation import steiner_tree
from sklearn_extra.cluster import KMedoids
from tslearn.metrics import cdist_dtw

from config import Config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(data_path: str, adj_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(data_path)["data"].astype(np.float32)
    adjacency = pd.read_csv(adj_path, index_col=0).values.astype(np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected traffic data (T,N,F), got {data.shape}")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1] or adjacency.shape[0] != data.shape[1]:
        raise ValueError(f"Adjacency shape {adjacency.shape} is incompatible with data shape {data.shape}")
    if not np.isfinite(data).all() or not np.isfinite(adjacency).all():
        raise ValueError("Traffic data or adjacency contains NaN/Inf")
    return data, adjacency


def sample_training_prototype(train_data: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Average randomly sampled complete training weeks and temporally downsample."""
    steps_per_week = cfg.steps_per_day * 7
    complete_weeks = train_data.shape[0] // steps_per_week
    if complete_weeks < 1:
        raise ValueError("The training split contains no complete week")

    count = min(cfg.sample_weeks, complete_weeks)
    rng = np.random.default_rng(cfg.cluster_seed)
    selected = np.sort(rng.choice(complete_weeks, size=count, replace=False))
    weeks = np.stack([train_data[w * steps_per_week:(w + 1) * steps_per_week] for w in selected])
    prototype = weeks.mean(axis=0)

    factor = int(cfg.downsample_factor)
    if factor < 1:
        raise ValueError("downsample_factor must be >= 1")
    usable = (prototype.shape[0] // factor) * factor
    prototype = prototype[:usable].reshape(usable // factor, factor, prototype.shape[1], prototype.shape[2]).mean(axis=1)
    prototype = np.transpose(prototype, (1, 0, 2))

    if cfg.cluster_standardize:
        mean = prototype.mean(axis=(0, 1), keepdims=True)
        std = prototype.std(axis=(0, 1), keepdims=True)
        prototype = (prototype - mean) / np.maximum(std, 1e-6)
    return prototype.astype(np.float32), selected.astype(np.int64)


def distance_to_similarity(distance: np.ndarray, diagonal_value: float = 1.0) -> np.ndarray:
    positive = distance[distance > 1e-8]
    sigma = max(float(np.median(positive)) if positive.size else 1.0, 1e-6)
    similarity = np.exp(-(distance ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    np.fill_diagonal(similarity, diagonal_value)
    return similarity


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)


def cluster_and_detect_outliers(raw_dist: np.ndarray, cfg: Config):
    """K-Medoids clustering followed by global-percentile within-cluster deviation detection."""
    model = KMedoids(n_clusters=cfg.normal_cluster_count, metric="precomputed", init="k-medoids++",
                     random_state=cfg.cluster_seed)
    labels = model.fit_predict(raw_dist).astype(np.int64)
    scores = np.zeros(raw_dist.shape[0], dtype=np.float32)

    for cluster_id in range(cfg.normal_cluster_count):
        members = np.where(labels == cluster_id)[0]
        if members.size > 1:
            intra = raw_dist[np.ix_(members, members)]
            scores[members] = (intra.sum(axis=1) / float(members.size - 1)).astype(np.float32)

    threshold = float(np.percentile(scores, cfg.outlier_percentile))
    return labels, scores, threshold, scores > threshold


def select_normal_representatives(raw_dist: np.ndarray, labels: np.ndarray, outlier_mask: np.ndarray, k: int) -> np.ndarray:
    """Choose one retained normal medoid-like representative per cluster."""
    representatives = []
    for cluster_id in range(k):
        members = np.where((labels == cluster_id) & (~outlier_mask))[0]
        if not members.size:
            raise ValueError(f"Normal cluster {cluster_id} has no retained node")
        if members.size == 1:
            representatives.append(int(members[0]))
        else:
            sub = raw_dist[np.ix_(members, members)]
            representatives.append(int(members[np.argmin(sub.sum(axis=1))]))
    return np.asarray(representatives, dtype=np.int64)


def build_membership(labels: np.ndarray, outlier_mask: np.ndarray, k: int) -> np.ndarray:
    membership = np.zeros((labels.size, k), dtype=np.float32)
    normal = np.where(~outlier_mask)[0]
    membership[normal, labels[normal]] = 1.0
    return membership


def build_outlier_assignment(raw_dist: np.ndarray, outlier_mask: np.ndarray, representatives: np.ndarray,
                             temperature_multiplier: float) -> np.ndarray:
    assignment = np.zeros((raw_dist.shape[0], representatives.size), dtype=np.float32)
    outliers = np.where(outlier_mask)[0]
    if not outliers.size:
        return assignment
    distances = raw_dist[np.ix_(outliers, representatives)]
    positive = distances[distances > 1e-8]
    scale = float(np.median(positive)) if positive.size else 1.0
    temperature = max(scale * float(temperature_multiplier), 1e-6)
    assignment[outliers] = stable_softmax(-distances / temperature, axis=1).astype(np.float32)
    return assignment


def symmetric_topk(similarity: np.ndarray, k: int) -> np.ndarray:
    n = similarity.shape[0]
    graph = np.zeros_like(similarity, dtype=np.float32)
    if n <= 1:
        return graph
    k = max(1, min(int(k), n - 1))
    work = similarity.copy()
    np.fill_diagonal(work, -np.inf)
    indices = np.argpartition(work, kth=n - k, axis=1)[:, -k:]
    rows = np.arange(n)[:, None]
    graph[rows, indices] = similarity[rows, indices]
    graph = np.maximum(graph, graph.T)
    np.fill_diagonal(graph, 0.0)
    return graph.astype(np.float32)


def build_functional_graph(similarity: np.ndarray, labels: np.ndarray, outlier_mask: np.ndarray,
                           adjacency: np.ndarray, cfg: Config) -> np.ndarray:
    graph = np.zeros_like(similarity, dtype=np.float32)
    for cluster_id in range(cfg.normal_cluster_count):
        members = np.where((labels == cluster_id) & (~outlier_mask))[0]
        if members.size <= 1:
            continue
        sub = symmetric_topk(similarity[np.ix_(members, members)], cfg.functional_topk)
        if cfg.functional_nonphysical_only:
            sub[adjacency[np.ix_(members, members)] > 0] = 0.0
        graph[np.ix_(members, members)] = sub
    np.fill_diagonal(graph, 0.0)
    return graph


def build_enhanced_road(adjacency: np.ndarray, delta_similarity: np.ndarray, labels: np.ndarray,
                        outlier_mask: np.ndarray, enhance_lambda: float) -> np.ndarray:
    if enhance_lambda < 0:
        raise ValueError("road_enhance_lambda must be nonnegative")
    enhanced = adjacency.astype(np.float32).copy()
    same_cluster = labels[:, None] == labels[None, :]
    normal_pair = (~outlier_mask)[:, None] & (~outlier_mask)[None, :]
    mask = (adjacency > 0) & same_cluster & normal_pair
    enhanced[mask] = adjacency[mask] * (1.0 + enhance_lambda * delta_similarity[mask])
    np.fill_diagonal(enhanced, 0.0)
    return enhanced


def build_semantic_graph(prototype: np.ndarray, labels: np.ndarray, outlier_mask: np.ndarray, cfg: Config) -> np.ndarray:
    cluster_prototypes = []
    for cluster_id in range(cfg.normal_cluster_count):
        members = np.where((labels == cluster_id) & (~outlier_mask))[0]
        if not members.size:
            raise ValueError(f"Normal cluster {cluster_id} has no retained node")
        cluster_prototypes.append(prototype[members].mean(axis=0))

    cluster_prototypes = np.stack(cluster_prototypes).astype(np.float32)
    similarity = distance_to_similarity(cdist_dtw(cluster_prototypes).astype(np.float32), diagonal_value=0.0)
    temperature = max(float(cfg.semantic_temperature), 1e-6)
    adjacency = np.power(np.maximum(similarity, 0.0), 1.0 / temperature).astype(np.float32)
    adjacency = 0.5 * (adjacency + adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def build_steiner_graph(adjacency: np.ndarray, enhanced_road: np.ndarray, outlier_mask: np.ndarray,
                        use_trend_cost: bool) -> Tuple[np.ndarray, np.ndarray]:
    n = adjacency.shape[0]
    outliers = np.where(outlier_mask)[0]
    result = np.zeros((n, n), dtype=np.float32)
    if outliers.size < 2:
        return result, np.zeros(n, dtype=np.bool_)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    ii, jj = np.where(np.triu((adjacency > 0) | (adjacency.T > 0), k=1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        strength = max(float(enhanced_road[i, j]), float(enhanced_road[j, i]), 1e-6)
        graph.add_edge(i, j, cost=1.0 / strength if use_trend_cost else 1.0, affinity=strength)

    outlier_set = set(map(int, outliers.tolist()))
    for component in nx.connected_components(graph):
        terminals = sorted(outlier_set.intersection(component))
        if len(terminals) < 2:
            continue
        tree = steiner_tree(graph.subgraph(component).copy(), terminals, weight="cost", method="kou")
        for u, v in tree.edges():
            strength = max(float(enhanced_road[u, v]), float(enhanced_road[v, u]), 1e-6)
            result[u, v] = result[v, u] = strength

    node_mask = (result > 0).any(axis=1) | (result > 0).any(axis=0)
    return result, node_mask.astype(np.bool_)


def validate_graphs(adjacency: np.ndarray, a_road: np.ndarray, a_func: np.ndarray, a_steiner: np.ndarray,
                    a_semantic: np.ndarray, membership: np.ndarray, assignment: np.ndarray,
                    labels: np.ndarray, outlier_mask: np.ndarray, cfg: Config) -> None:
    n, k = adjacency.shape[0], cfg.normal_cluster_count
    for name, matrix in {"A_en_road": a_road, "A_func": a_func, "A_steiner": a_steiner}.items():
        if matrix.shape != (n, n) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid {name}: shape={matrix.shape}")
    if a_semantic.shape != (k, k) or membership.shape != (n, k) or assignment.shape != (n, k):
        raise ValueError("Semantic graph or assignment matrix has an invalid shape")
    if np.any((adjacency > 0) & (a_road <= 0)):
        raise ValueError("Trend-Enhanced Road Graph removed an original physical edge")

    normal = ~outlier_mask
    if not np.allclose(membership[normal].sum(axis=1), 1.0) or not np.allclose(membership[outlier_mask].sum(axis=1), 0.0):
        raise ValueError("Invalid hard cluster membership")
    if outlier_mask.any() and not np.allclose(assignment[outlier_mask].sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Invalid outlier soft assignment")
    if not np.allclose(assignment[normal].sum(axis=1), 0.0):
        raise ValueError("Normal nodes must have zero outlier soft assignment")

    i, j = np.where(a_func > 0)
    if i.size and np.any(outlier_mask[i] | outlier_mask[j] | (labels[i] != labels[j])):
        raise ValueError("Intra-Cluster Functional Graph contains an invalid edge")
    i, j = np.where(a_steiner > 0)
    if i.size and np.any(~((adjacency > 0) | (adjacency.T > 0))[i, j]):
        raise ValueError("Steiner Subgraph contains a non-physical edge")


def main() -> None:
    cfg = Config()
    cfg.prepare_dirs()
    set_seed(cfg.cluster_seed)

    data, adjacency = load_data(cfg.data_path, cfg.adj_path)
    cfg.num_nodes = data.shape[1]
    train_end = int(data.shape[0] * cfg.train_ratio)
    train_data = data[:train_end, :, :cfg.raw_feature_dim]

    prototype, selected_weeks = sample_training_prototype(train_data, cfg)
    raw_dist = cdist_dtw(prototype).astype(np.float32)
    raw_similarity = distance_to_similarity(raw_dist)
    labels, outlier_scores, outlier_threshold, outlier_mask = cluster_and_detect_outliers(raw_dist, cfg)
    representatives = select_normal_representatives(raw_dist, labels, outlier_mask, cfg.normal_cluster_count)
    membership = build_membership(labels, outlier_mask, cfg.normal_cluster_count)
    assignment = build_outlier_assignment(raw_dist, outlier_mask, representatives, cfg.outlier_assignment_temperature)

    delta_flow = np.diff(prototype[..., cfg.target_feature], axis=1)
    delta_similarity = distance_to_similarity(cdist_dtw(delta_flow).astype(np.float32))
    a_en_road = build_enhanced_road(adjacency, delta_similarity, labels, outlier_mask, cfg.road_enhance_lambda)
    a_func = build_functional_graph(raw_similarity, labels, outlier_mask, adjacency, cfg)
    a_semantic = build_semantic_graph(prototype, labels, outlier_mask, cfg)
    a_steiner, steiner_node_mask = build_steiner_graph(adjacency, a_en_road, outlier_mask, cfg.steiner_use_trend_cost)

    validate_graphs(adjacency, a_en_road, a_func, a_steiner, a_semantic, membership, assignment, labels,
                    outlier_mask, cfg)

    output_dir = Path(cfg.graph_dir)
    np.savez_compressed(output_dir / "graph_bundle.npz", A_en_road=a_en_road, A_func=a_func,
                        A_steiner=a_steiner, A_semantic=a_semantic, cluster_membership=membership,
                        outlier_soft_assignment=assignment, outlier_mask=outlier_mask.astype(np.bool_),
                        steiner_node_mask=steiner_node_mask)

    summary = {
        "data_shape": list(data.shape), "training_end_index": train_end,
        "selected_training_weeks": selected_weeks.tolist(), "prototype_shape": list(prototype.shape),
        "normal_cluster_count": cfg.normal_cluster_count, "outlier_percentile": cfg.outlier_percentile,
        "outlier_threshold": outlier_threshold, "outlier_count": int(outlier_mask.sum()),
        "outlier_rate": float(outlier_mask.mean()), "enhanced_road_edges": int((a_en_road > 0).sum()),
        "functional_edges": int((a_func > 0).sum()), "steiner_edges": int((a_steiner > 0).sum()),
        "valid_steiner_nodes": int(steiner_node_mask.sum())
    }
    with open(output_dir / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved graph bundle: {output_dir / 'graph_bundle.npz'}")


if __name__ == "__main__":
    main()
