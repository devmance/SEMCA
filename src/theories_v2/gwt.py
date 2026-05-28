"""
Substrate-agnostic Global Workspace Theory (GWT) measurement.

GWT (Baars 1988, Dehaene & Naccache 2001): consciousness arises from global
information broadcast across many specialized processors. The substrate-
agnostic signature is *synchronized cross-group activation* — moments when
many node-groups are simultaneously highly active, an "ignition event".

Three sub-scores:

  1. **Cross-Group Activation Spread (CGAS)**: per-timestep, count groups
     in their top quantile of activity. Higher mean = more frequent
     broadcast events. Substrate-symmetric between transformer-layer
     cascades and fMRI-network ignitions.

  2. **Node-Diversity Engagement (NDE)**: per-timestep, count nodes
     (across all groups) above a global quantile. Wide engagement = many
     processors active simultaneously.

  3. **Inter-Group Coupling (IGC)**: mean pairwise correlation between
     group-level activity vectors. High coupling = groups co-activate
     during broadcast events.

The cosine-similarity-graph mathematics is identical to substrate-
agnostic IIT; what differs is the aggregation (cross-group spread
rather than partition-search).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..substrate.base import Substrate


@dataclass
class GWTResult:
    gwt_value: float
    confidence: float
    components: Dict[str, float] = field(default_factory=dict)
    raw: Dict[str, float] = field(default_factory=dict)
    substrate_type: str = ""


def compute_gwt(
    substrate: Substrate,
    weight_cgas: float = 0.40,
    weight_nde: float = 0.30,
    weight_igc: float = 0.30,
    quantile: float = 0.85,
) -> GWTResult:
    """Compute substrate-agnostic GWT score."""
    weight_sum = weight_cgas + weight_nde + weight_igc
    if not (0.999 <= weight_sum <= 1.001):
        raise ValueError(f"Weights must sum to 1.0; got {weight_sum}")

    activity = substrate.activity
    if activity.shape[0] < 4 or activity.shape[1] < 3:
        return GWTResult(
            gwt_value=0.0, confidence=0.0,
            components={"CGAS": 0.0, "NDE": 0.0, "IGC": 0.0},
            raw={"note": "substrate_too_small"},
            substrate_type=substrate.substrate_type,
        )

    # Take absolute activity then per-node normalize so different node
    # scales don't dominate
    abs_act = np.abs(activity)
    per_node_max = abs_act.max(axis=0, keepdims=True) + 1e-8
    normed = abs_act / per_node_max  # (T, N), each column in [0, 1]

    groups = substrate.node_groups
    if not groups:
        groups = [list(range(activity.shape[1]))]

    # Per-group per-timestep mean activity → (T, G)
    group_activity = np.stack(
        [normed[:, g_idx].mean(axis=1) for g_idx in groups],
        axis=1,
    )

    # ----- CGAS: per-timestep count of groups above quantile -----
    thresh_g = np.quantile(group_activity, quantile, axis=0, keepdims=True)
    above_g = (group_activity >= thresh_g).astype(np.float64)  # (T, G)
    # Per-timestep fraction of groups simultaneously above threshold
    cgas_per_t = above_g.mean(axis=1)
    # Mean fraction. Empirically broadcast events are 5-15% of timesteps;
    # scale 0.10 → 50, 0.20 → 100
    cgas_raw = float(cgas_per_t.mean())
    cgas_score = float(np.clip(cgas_raw * 500.0, 0.0, 100.0))

    # ----- NDE: per-timestep count of all nodes above global quantile -----
    thresh_n = np.quantile(normed, quantile, axis=0, keepdims=True)
    above_n = (normed >= thresh_n).astype(np.float64)
    nde_per_t = above_n.mean(axis=1)
    nde_raw = float(nde_per_t.mean())
    nde_score = float(np.clip(nde_raw * 500.0, 0.0, 100.0))

    # ----- IGC: mean pairwise correlation between group-level vectors -----
    if group_activity.shape[1] < 2:
        igc_score = 0.0
        igc_raw = 0.0
    else:
        ga = group_activity - group_activity.mean(axis=0, keepdims=True)
        std = ga.std(axis=0, keepdims=True) + 1e-8
        ga_n = ga / std
        if ga_n.shape[0] < 2:
            igc_raw = 0.0
        else:
            corr = (ga_n.T @ ga_n) / max(1, ga_n.shape[0] - 1)
            np.fill_diagonal(corr, 0.0)
            # Mean of upper triangle (each pair counted once)
            iu = np.triu_indices(corr.shape[0], k=1)
            igc_raw = float(corr[iu].mean())
        # Map [-1, 1] correlation to [0, 100] (negative coupling → 0)
        igc_score = float(np.clip((igc_raw + 1.0) * 50.0, 0.0, 100.0))

    # ----- Aggregation -----
    gwt_raw = (
        weight_cgas * cgas_score
        + weight_nde * nde_score
        + weight_igc * igc_score
    )
    gwt_score = float(np.clip(gwt_raw, 0.0, 100.0))

    sub = np.array([cgas_score, nde_score, igc_score])
    if sub.mean() > 0:
        cv = sub.std() / sub.mean()
        confidence = float(np.clip(1.0 - cv, 0.0, 1.0))
    else:
        confidence = 0.0

    return GWTResult(
        gwt_value=gwt_score,
        confidence=confidence,
        components={
            "CGAS": float(cgas_score),
            "NDE": float(nde_score),
            "IGC": float(igc_score),
        },
        raw={
            "cgas_fraction": float(cgas_raw),
            "nde_fraction": float(nde_raw),
            "igc_correlation": float(igc_raw),
            "n_groups": float(len(groups)),
            "n_nodes": float(activity.shape[1]),
            "n_timesteps": float(activity.shape[0]),
        },
        substrate_type=substrate.substrate_type,
    )
