"""
Substrate-agnostic Attention Schema Theory (AST) measurement.

AST (Graziano 2013): consciousness is the brain's model of its own
attention. The substrate-agnostic test: can the activity of one set of
nodes be predicted from the activity of upstream nodes? If yes, the
upstream nodes maintain an internal model of the downstream behavior.

Substrate-symmetric: in transformers we ask whether early-layer activity
predicts later-layer attention patterns; in fMRI we ask whether
parietal/temporal regions predict frontal regions' activation patterns.

Three sub-scores:

  1. **Upstream Predictability (UP)**: CCA-style predictability from
     earlier groups to later groups. High = the system has internal
     models of later-group behavior present in earlier groups.

  2. **Schema Compression (SC)**: spectral entropy of each group's
     activity matrix. Low entropy = compressed (low-rank) representation
     = the system abstracts over the dimensions of activity.

  3. **Schema Coherence (SCoh)**: Frobenius similarity between
     normalized group-correlation matrices at successive group depths.
     High coherence = stable schema across group hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from ..substrate.base import Substrate


@dataclass
class ASTResult:
    ast_value: float
    confidence: float
    components: Dict[str, float] = field(default_factory=dict)
    raw: Dict[str, float] = field(default_factory=dict)
    substrate_type: str = ""


def _canonical_correlation(a: np.ndarray, b: np.ndarray, k: int = 8) -> float:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return 0.0
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    a_std = a.std(axis=0) + 1e-8
    b_std = b.std(axis=0) + 1e-8
    a = a / a_std
    b = b / b_std
    c = (a.T @ b) / a.shape[0]
    try:
        sv = np.linalg.svd(c, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0
    if sv.size == 0:
        return 0.0
    return float(np.clip(sv[:min(k, len(sv))].mean(), 0.0, 1.0))


def compute_ast(
    substrate: Substrate,
    weight_up: float = 0.45,
    weight_sc: float = 0.30,
    weight_scoh: float = 0.25,
    max_features: int = 128,
) -> ASTResult:
    """Compute substrate-agnostic AST score."""
    weight_sum = weight_up + weight_sc + weight_scoh
    if not (0.999 <= weight_sum <= 1.001):
        raise ValueError(f"Weights must sum to 1.0; got {weight_sum}")

    activity = substrate.activity
    groups = substrate.node_groups
    if not groups or len(groups) < 2 or activity.shape[0] < 6:
        return ASTResult(
            ast_value=0.0, confidence=0.0,
            components={"UP": 0.0, "SC": 0.0, "SCoh": 0.0},
            raw={"note": "insufficient_groups_or_timesteps"},
            substrate_type=substrate.substrate_type,
        )

    # Build per-group activity, downsample if too many features
    group_arrs = []
    for g_idx in groups:
        if len(g_idx) < 2:
            continue
        ga = activity[:, g_idx]
        if ga.shape[1] > max_features:
            stride = max(1, ga.shape[1] // max_features)
            ga = ga[:, ::stride][:, :max_features]
        group_arrs.append(ga)
    n_groups = len(group_arrs)
    if n_groups < 2:
        return ASTResult(
            ast_value=0.0, confidence=0.0,
            components={"UP": 0.0, "SC": 0.0, "SCoh": 0.0},
            raw={"note": "fewer_than_2_groups"},
            substrate_type=substrate.substrate_type,
        )

    # ----- UP: predictability from upstream to downstream groups -----
    up_corrs = []
    for L in range(n_groups - 1):
        cc = _canonical_correlation(group_arrs[L], group_arrs[L + 1])
        up_corrs.append(cc)
    if up_corrs:
        up_score = float(np.mean(up_corrs)) * 100.0
    else:
        up_score = 0.0

    # ----- SC: spectral entropy of each group's correlation matrix -----
    sc_per_group = []
    for ga in group_arrs:
        n_t = ga.shape[0]
        if n_t < 2:
            continue
        ga_c = ga - ga.mean(axis=0, keepdims=True)
        # ROI-correlation matrix
        sv = np.linalg.svd(ga_c, compute_uv=False)
        sv = sv / (sv.sum() + 1e-8)
        sp = sv.clip(min=1e-12)
        h = -(sp * np.log(sp)).sum()
        max_h = float(np.log(len(sv))) if len(sv) > 1 else 1.0
        compression = 1.0 - h / max_h
        sc_per_group.append(float(compression))
    if sc_per_group:
        sc_raw = float(np.mean(sc_per_group))
        sc_score = float(np.clip(sc_raw * 200.0, 0.0, 100.0))
    else:
        sc_score = 0.0

    # ----- SCoh: Frobenius similarity of adjacent-group correlation matrices -----
    scoh_sims = []
    prev = None
    for ga in group_arrs:
        if ga.shape[1] < 2:
            prev = None
            continue
        ga_c = ga - ga.mean(axis=0, keepdims=True)
        s = ga_c.std(axis=0) + 1e-8
        ga_n = ga_c / s
        if ga_n.shape[0] < 2:
            prev = None
            continue
        corr = (ga_n.T @ ga_n) / ga_n.shape[0]
        if prev is not None and prev.shape == corr.shape:
            num = (corr * prev).sum()
            denom = (np.linalg.norm(corr) * np.linalg.norm(prev)) + 1e-8
            scoh_sims.append(float(num / denom))
        prev = corr
    if scoh_sims:
        scoh_raw = float(np.mean(scoh_sims))
        scoh_score = float(np.clip(scoh_raw * 100.0, 0.0, 100.0))
    else:
        scoh_score = 0.0

    # ----- Aggregation -----
    ast_raw = (
        weight_up * up_score
        + weight_sc * sc_score
        + weight_scoh * scoh_score
    )
    ast_score = float(np.clip(ast_raw, 0.0, 100.0))

    sub = np.array([up_score, sc_score, scoh_score])
    if sub.mean() > 0:
        cv = sub.std() / sub.mean()
        confidence = float(np.clip(1.0 - cv, 0.0, 1.0))
    else:
        confidence = 0.0

    return ASTResult(
        ast_value=ast_score,
        confidence=confidence,
        components={
            "UP": float(up_score),
            "SC": float(sc_score),
            "SCoh": float(scoh_score),
        },
        raw={
            "n_groups_used": float(n_groups),
            "mean_up_corr": float(np.mean(up_corrs)) if up_corrs else 0.0,
            "mean_compression": float(np.mean(sc_per_group)) if sc_per_group else 0.0,
        },
        substrate_type=substrate.substrate_type,
    )
