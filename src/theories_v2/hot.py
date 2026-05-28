"""
Substrate-agnostic Higher-Order Thought (HOT) measurement.

HOT (Rosenthal 2005): a mental state is conscious iff it is the object
of a suitable higher-order thought. Substrate-agnostic operationalization:
do later (or higher-order) node groups encode information about earlier
(lower-order) node groups?

Three sub-scores:

  1. **Self-Representation Capacity (SRC)**: mean canonical correlation
     between pairs of node groups (G_i, G_j) where i < j. High = later
     groups encode earlier-group information.

  2. **Recursive Depth (RD)**: how slowly CCA decays as group-distance
     grows. Slow decay = chains of representational nesting.

  3. **Meta-Representation Stability (MRS)**: do the cross-group CCA
     patterns hold across temporal sub-windows? Stable = the system's
     meta-representation is robust to content; unstable = idiosyncratic
     per stimulus.

Substrate-symmetric: transformer (groups = layers) and fMRI (groups =
networks in hierarchy: sensory → frontal) both work with the same math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from ..substrate.base import Substrate


@dataclass
class HOTResult:
    hot_value: float
    confidence: float
    components: Dict[str, float] = field(default_factory=dict)
    raw: Dict[str, float] = field(default_factory=dict)
    substrate_type: str = ""


def _canonical_correlation(a: np.ndarray, b: np.ndarray, k: int = 8) -> float:
    """Mean of top-k singular values of normalized cross-correlation matrix."""
    if a.shape[0] < 2 or b.shape[0] < 2:
        return 0.0
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    a_std = a.std(axis=0) + 1e-8
    b_std = b.std(axis=0) + 1e-8
    a = a / a_std
    b = b / b_std
    if a.shape[0] < 2:
        return 0.0
    c = (a.T @ b) / a.shape[0]
    try:
        sv = np.linalg.svd(c, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0
    if sv.size == 0:
        return 0.0
    return float(np.clip(sv[:min(k, len(sv))].mean(), 0.0, 1.0))


def compute_hot(
    substrate: Substrate,
    weight_src: float = 0.45,
    weight_rd: float = 0.30,
    weight_mrs: float = 0.25,
    max_features: int = 256,
) -> HOTResult:
    """Compute substrate-agnostic HOT score."""
    weight_sum = weight_src + weight_rd + weight_mrs
    if not (0.999 <= weight_sum <= 1.001):
        raise ValueError(f"Weights must sum to 1.0; got {weight_sum}")

    activity = substrate.activity
    groups = substrate.node_groups
    if not groups or len(groups) < 3 or activity.shape[0] < 8:
        return HOTResult(
            hot_value=0.0, confidence=0.0,
            components={"SRC": 0.0, "RD": 0.0, "MRS": 0.0},
            raw={"note": "insufficient_groups_or_timesteps"},
            substrate_type=substrate.substrate_type,
        )

    # Build per-group activity matrices, downsampled if too wide
    group_arrs = []
    for g_idx in groups:
        if len(g_idx) < 2:
            continue
        ga = activity[:, g_idx]
        if ga.shape[1] > max_features:
            stride = max(1, ga.shape[1] // max_features)
            ga = ga[:, ::stride][:, :max_features]
        group_arrs.append(ga)

    if len(group_arrs) < 3:
        return HOTResult(
            hot_value=0.0, confidence=0.0,
            components={"SRC": 0.0, "RD": 0.0, "MRS": 0.0},
            raw={"note": "fewer_than_3_groups_with_2plus_nodes"},
            substrate_type=substrate.substrate_type,
        )

    n_groups = len(group_arrs)

    # ----- SRC: mean CCA between distant group pairs -----
    src_corrs = []
    for i in range(n_groups):
        for k in (1, 2, 3, 4):
            j = i + k
            if j >= n_groups:
                continue
            cc = _canonical_correlation(group_arrs[i], group_arrs[j])
            src_corrs.append(cc)
    if src_corrs:
        src_raw = float(np.mean(src_corrs))
        src_score = src_raw * 100.0
    else:
        src_score = 0.0

    # ----- RD: slope of CCA decay with group distance -----
    decay_slopes = []
    for i in range(n_groups):
        ks = []
        ccs = []
        for k in range(1, min(5, n_groups - i)):
            cc = _canonical_correlation(group_arrs[i], group_arrs[i + k])
            ks.append(k); ccs.append(cc)
        if len(ks) >= 3:
            ks_arr = np.array(ks, dtype=np.float64)
            ccs_arr = np.array(ccs, dtype=np.float64)
            if ccs_arr.std() > 1e-8:
                slope = float(np.polyfit(ks_arr, ccs_arr, 1)[0])
                decay_slopes.append(slope)
    if decay_slopes:
        mean_slope = float(np.mean(decay_slopes))
        # Slope ∈ [-0.3, 0] maps to [0, 100]; less-negative slope = more depth
        rd_score = float(np.clip(100.0 * (1.0 + mean_slope / 0.3), 0.0, 100.0))
    else:
        rd_score = 0.0

    # ----- MRS: cross-window stability of (L, L+1) CCAs -----
    n_t = activity.shape[0]
    if n_t < 6:
        mrs_score = 0.0
    else:
        mid = n_t // 2
        mrs_divs = []
        for i in range(n_groups - 1):
            a1, b1 = group_arrs[i][:mid], group_arrs[i + 1][:mid]
            a2, b2 = group_arrs[i][mid:], group_arrs[i + 1][mid:]
            cc1 = _canonical_correlation(a1, b1)
            cc2 = _canonical_correlation(a2, b2)
            mrs_divs.append(abs(cc1 - cc2))
        if mrs_divs:
            mean_div = float(np.mean(mrs_divs))
            mrs_score = float(np.clip(100.0 * (1.0 - mean_div / 0.3), 0.0, 100.0))
        else:
            mrs_score = 0.0

    # ----- Aggregation -----
    hot_raw = (
        weight_src * src_score
        + weight_rd * rd_score
        + weight_mrs * mrs_score
    )
    hot_score = float(np.clip(hot_raw, 0.0, 100.0))

    sub = np.array([src_score, rd_score, mrs_score])
    if sub.mean() > 0:
        cv = sub.std() / sub.mean()
        confidence = float(np.clip(1.0 - cv, 0.0, 1.0))
    else:
        confidence = 0.0

    return HOTResult(
        hot_value=hot_score,
        confidence=confidence,
        components={
            "SRC": float(src_score),
            "RD": float(rd_score),
            "MRS": float(mrs_score),
        },
        raw={
            "n_groups_used": float(n_groups),
            "mean_decay_slope": float(np.mean(decay_slopes)) if decay_slopes else 0.0,
        },
        substrate_type=substrate.substrate_type,
    )
