"""
Substrate-agnostic Free Energy Principle (FEP) measurement.

FEP (Friston 2010, 2019): adaptive systems minimize variational free
energy F = D_KL[q(s) || p(s|o)] - log p(o). Operationally for a system
producing time-series of activity:

  VFE_t ≈ -log p(observed_{t+1} | predicted state at t)

This is exactly the cross-entropy series from a PredictionAdapter.
Substrate-agnostic by construction.

Three sub-scores:

  1. **Variational Free Energy (VFE)**: mean cross-entropy. Low = good
     model fit = effective free-energy minimization.

  2. **Layer-/Group-wise Free Energy Decay (LFED)**: correlation between
     group-index and group-level free-energy proxy. Hierarchical
     generative models should show monotonic FE decay across depth.

  3. **Active-Inference Information Gain (AIIG)**: correlation between
     per-step prediction entropy and per-step prediction error. Active
     inference predicts positive correlation: high-uncertainty steps
     should also be high-error steps (the system is exploring its
     uncertainty).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from ..substrate.base import PredictionAdapter, Substrate


@dataclass
class FEPResult:
    fep_value: float
    confidence: float
    components: Dict[str, float] = field(default_factory=dict)
    raw: Dict[str, float] = field(default_factory=dict)
    substrate_type: str = ""


def compute_fep(
    substrate: Substrate,
    prediction_adapter: PredictionAdapter,
    weight_vfe: float = 0.40,
    weight_lfed: float = 0.30,
    weight_aiig: float = 0.30,
) -> FEPResult:
    weight_sum = weight_vfe + weight_lfed + weight_aiig
    if not (0.999 <= weight_sum <= 1.001):
        raise ValueError(f"Weights must sum to 1.0; got {weight_sum}")

    if prediction_adapter.n_predictable_steps < 3:
        return FEPResult(
            fep_value=0.0, confidence=0.0,
            components={"VFE": 0.0, "LFED": 0.0, "AIIG": 0.0},
            raw={"note": "too_few_predictable_steps"},
            substrate_type=substrate.substrate_type,
        )

    h_series = prediction_adapter.prediction_entropy_series()
    ce_series = prediction_adapter.cross_entropy_series()

    # ----- VFE: mean cross-entropy, inverted and normalized -----
    mean_ce = float(ce_series.mean())
    max_ce = max(float(ce_series.max()) + 1e-8, 1.0)
    norm_ce = min(1.0, mean_ce / max_ce)
    vfe_score = (1.0 - norm_ce) * 100.0

    # ----- LFED: group-wise FE proxy decay -----
    activity = substrate.activity
    groups = substrate.node_groups
    if groups and len(groups) >= 3:
        per_group_recon_error = []
        for g_idx in groups:
            if len(g_idx) < 2:
                continue
            ga = activity[:, g_idx]
            ga_c = ga - ga.mean(axis=0, keepdims=True)
            if ga_c.shape[0] < 2:
                continue
            try:
                U, S, Vt = np.linalg.svd(ga_c, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            if S.size == 0:
                continue
            recon = U[:, :1] * S[0:1] @ Vt[:1, :]
            err = float(np.linalg.norm(ga_c - recon) / (np.linalg.norm(ga_c) + 1e-8))
            per_group_recon_error.append(err)
        if len(per_group_recon_error) >= 3:
            arr = np.array(per_group_recon_error, dtype=np.float64)
            idx = np.arange(len(arr), dtype=np.float64)
            if arr.std() > 1e-8:
                corr = float(np.corrcoef(idx, arr)[0, 1])
            else:
                corr = 0.0
            lfed_raw = (-corr + 1.0) / 2.0
            lfed_score = lfed_raw * 100.0
        else:
            lfed_score = 0.0
            corr = 0.0
    else:
        lfed_score = 0.0
        corr = 0.0

    # ----- AIIG: correlation between entropy and CE per step -----
    n = min(len(h_series), len(ce_series))
    if n >= 3:
        h_arr = h_series[:n]
        ce_arr = ce_series[:n]
        if h_arr.std() > 1e-8 and ce_arr.std() > 1e-8:
            aiig_corr = float(np.corrcoef(h_arr, ce_arr)[0, 1])
        else:
            aiig_corr = 0.0
        aiig_raw = (aiig_corr + 1.0) / 2.0
        aiig_score = aiig_raw * 100.0
    else:
        aiig_corr = 0.0
        aiig_score = 0.0

    # ----- Aggregation -----
    fep_raw = (
        weight_vfe * vfe_score
        + weight_lfed * lfed_score
        + weight_aiig * aiig_score
    )
    fep_score = float(np.clip(fep_raw, 0.0, 100.0))

    sub = np.array([vfe_score, lfed_score, aiig_score])
    if sub.mean() > 0:
        cv = sub.std() / sub.mean()
        confidence = float(np.clip(1.0 - cv, 0.0, 1.0))
    else:
        confidence = 0.0

    return FEPResult(
        fep_value=fep_score,
        confidence=confidence,
        components={
            "VFE": float(vfe_score),
            "LFED": float(lfed_score),
            "AIIG": float(aiig_score),
        },
        raw={
            "mean_ce": float(mean_ce),
            "norm_ce": float(norm_ce),
            "lfed_corr": float(corr),
            "aiig_corr": float(aiig_corr),
        },
        substrate_type=substrate.substrate_type,
    )
