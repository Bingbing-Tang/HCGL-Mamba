"""HCGL-Mamba model.

Normal nodes use the Trend-Enhanced Road Graph, Intra-Cluster Functional Graph,
and Inter-Cluster Semantic Graph. Behavior-deviating nodes replace the functional
branch with the Steiner Subgraph. The complete road topology is preserved.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from mamba_ssm import Mamba
except ImportError:
    from mamba_ssm.modules.mamba_simple import Mamba


class ResidualGraphLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        if x.ndim != 4 or adjacency.ndim != 2 or adjacency.shape[0] != x.shape[2]:
            raise ValueError(f"Invalid graph input: x={tuple(x.shape)}, adjacency={tuple(adjacency.shape)}")
        message = F.gelu(torch.matmul(adjacency, self.proj(x)) + self.bias)
        return self.norm(x + self.dropout(message))


class ResidualGraphChannel(nn.Module):
    def __init__(self, d_model: int, dropout: float, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.layers = nn.ModuleList(ResidualGraphLayer(d_model, dropout) for _ in range(num_layers))

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, adjacency)
        return x


class StaticClusterSemanticChannel(nn.Module):
    """Propagate dynamic cluster features through a static inter-cluster graph."""

    def __init__(self, d_model: int, dropout: float, num_layers: int) -> None:
        super().__init__()
        self.cluster_channel = ResidualGraphChannel(d_model, dropout, num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, adjacency: Tensor, membership: Tensor, outlier_assignment: Tensor) -> Tensor:
        n = x.shape[2]
        if membership.ndim != 2 or membership.shape[0] != n or outlier_assignment.shape != membership.shape:
            raise ValueError("Invalid cluster membership or outlier assignment shape")

        membership = membership.to(x.dtype)
        outlier_assignment = outlier_assignment.to(x.dtype)
        counts = membership.sum(dim=0).clamp_min(1.0)
        clusters = torch.einsum("nk,btnd->btkd", membership, x) / counts.view(1, 1, -1, 1)
        clusters = self.cluster_channel(clusters, adjacency)
        return self.norm(torch.einsum("nk,btkd->btnd", membership + outlier_assignment, clusters))


class HeterogeneousSpatialEncoder(nn.Module):
    """Channel-wise graph convolution with node-adaptive branch fusion."""

    def __init__(self, config, input_dim: int) -> None:
        super().__init__()
        d, p, layers = config.spatial_dim, config.spatial_dropout, config.graph_layers
        self.input_proj = nn.Linear(input_dim, d)
        self.input_norm = nn.LayerNorm(d)
        self.road = ResidualGraphChannel(d, p, layers)
        self.functional = ResidualGraphChannel(d, p, layers)
        self.steiner = ResidualGraphChannel(d, p, layers)
        self.semantic = StaticClusterSemanticChannel(d, p, layers)
        self.normal_gate = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 3))
        self.outlier_gate = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 3))
        self.output_norm = nn.LayerNorm(d)

        self._init_gate(self.normal_gate, [config.normal_gate_init_road, config.normal_gate_init_func,
                                           config.normal_gate_init_semantic])
        self._init_gate(self.outlier_gate, [config.outlier_gate_init_road, config.outlier_gate_init_steiner,
                                            config.outlier_gate_init_semantic])

    @staticmethod
    def _init_gate(gate: nn.Sequential, bias) -> None:
        nn.init.zeros_(gate[-1].weight)
        with torch.no_grad():
            gate[-1].bias.copy_(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x: Tensor, a_road: Tensor, a_func: Tensor, a_steiner: Tensor, a_semantic: Tensor,
                membership: Tensor, outlier_assignment: Tensor, outlier_mask: Tensor,
                steiner_node_mask: Tensor) -> Tensor:
        h0 = self.input_norm(self.input_proj(x))
        n, k = h0.shape[2], membership.shape[1]
        if any(a.shape != (n, n) for a in (a_road, a_func, a_steiner)):
            raise ValueError("Node-level graph shape does not match the input node count")
        if a_semantic.shape != (k, k) or outlier_mask.shape != (n,) or steiner_node_mask.shape != (n,):
            raise ValueError("Semantic graph or node mask shape is inconsistent")

        h_road = self.road(h0, a_road)
        h_func = self.functional(h0, a_func)
        h_steiner = self.steiner(h0, a_steiner)
        h_sem = self.semantic(h0, a_semantic, membership, outlier_assignment)

        normal_gate = torch.softmax(self.normal_gate(torch.cat((h_road, h_func, h_sem), dim=-1)), dim=-1)
        h_normal = normal_gate[..., :1] * h_road + normal_gate[..., 1:2] * h_func + normal_gate[..., 2:] * h_sem

        logits = self.outlier_gate(torch.cat((h_road, h_steiner, h_sem), dim=-1))
        valid = steiner_node_mask.bool().view(1, 1, n)
        steiner_logits = torch.where(valid, logits[..., 1], torch.full_like(logits[..., 1], torch.finfo(logits.dtype).min))
        logits = torch.stack((logits[..., 0], steiner_logits, logits[..., 2]), dim=-1)
        outlier_gate = torch.softmax(logits, dim=-1)
        h_outlier = outlier_gate[..., :1] * h_road + outlier_gate[..., 1:2] * h_steiner + outlier_gate[..., 2:] * h_sem

        route = outlier_mask.bool().view(1, 1, n, 1)
        return self.output_norm(torch.where(route, h_outlier, h_normal) + h0)


class MambaTemporalEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        d = config.mamba_d_model
        kwargs = dict(d_model=d, d_state=config.mamba_d_state, d_conv=config.mamba_d_conv, expand=config.mamba_expand)
        self.bidirectional = bool(config.bidirectional_mamba)
        self.forward_mamba = Mamba(**kwargs)
        self.backward_mamba = Mamba(**kwargs) if self.bidirectional else None
        self.bi_proj = nn.Linear(2 * d, d) if self.bidirectional else None
        self.dropout = nn.Dropout(config.temporal_dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Temporal input must be (B*N,T,D), got {tuple(x.shape)}")
        forward = self.forward_mamba(x)
        if not self.bidirectional:
            return self.norm(x + self.dropout(forward))
        backward = torch.flip(self.backward_mamba(torch.flip(x, dims=(1,))), dims=(1,))
        fused = F.gelu(self.bi_proj(torch.cat((forward, backward), dim=-1)))
        return self.norm(x + self.dropout(fused))


class DirectMultiStepForecastHead(nn.Module):
    def __init__(self, d_model: int, input_len: int, pred_len: int) -> None:
        super().__init__()
        self.input_len = input_len
        self.time_proj = nn.Linear(input_len, pred_len)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got {x.shape[1]}")
        return self.output_proj(self.time_proj(x.transpose(1, 2)).transpose(1, 2))


class HCGLMamba(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.spatial = HeterogeneousSpatialEncoder(config, config.raw_feature_dim)
        self.spatial_to_temporal = nn.Identity() if config.spatial_dim == config.mamba_d_model else nn.Linear(
            config.spatial_dim, config.mamba_d_model)
        self.temporal = MambaTemporalEncoder(config)
        self.head = DirectMultiStepForecastHead(config.mamba_d_model, config.input_len, config.pred_len)

    def forward(self, x: Tensor, a_en_road: Tensor, a_func: Tensor, a_steiner: Tensor, a_semantic: Tensor,
                normal_membership: Tensor, outlier_soft_assignment: Tensor, outlier_mask: Tensor,
                steiner_node_mask: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be (B,T,N,F), got {tuple(x.shape)}")
        b, t, n, f = x.shape
        if t != self.config.input_len or f != self.config.raw_feature_dim:
            raise ValueError(f"Expected T={self.config.input_len}, F={self.config.raw_feature_dim}; got T={t}, F={f}")

        h = self.spatial(x, a_en_road, a_func, a_steiner, a_semantic, normal_membership,
                         outlier_soft_assignment, outlier_mask, steiner_node_mask)
        h = self.spatial_to_temporal(h).permute(0, 2, 1, 3).contiguous().view(b * n, t, self.config.mamba_d_model)
        y = self.head(self.temporal(h)).view(b, n, self.config.pred_len, 1)
        return y.permute(0, 2, 1, 3).contiguous()
