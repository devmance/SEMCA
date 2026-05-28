"""
Substrate-agnostic Predictive Processing Theory (PPT) measurement.

PPT (Clark 2013, Hohwy 2013): adaptive systems are hierarchical generative
models minimizing prediction error. Substrate-agnostic operationalization
requires a `PredictionAdapter` exposing per-step prediction entropy and
cross-entropy.

  - Transformer adapter: logit-based prediction (native).
  - fMRI adapter: lagged regression forward model over ROIs.

The PPT calculation operates only on the adapter output, so the
downstream math is identical on both substrates.

Three sub-scores:

  1. **Prediction-Distribution Entropy (PDE)**: mean per-step entropy of
     the predicted-next-state distribution. Low entropy = sharp prediction
     = high confidence at this hierarchical level.

  2. **Hierarchical Prediction-Error Trajectory (HPET)**: correlation
     between group-index and group-level prediction error. Substrates
     with multiple node-groups (transformer layers, fMRI networks) should
     show prediction error DECREASING from earlier to later groups under
     PPT.

  3. **Precision-Weighted Cross-Entropy (PWCE)**: cross-entropy weighted
     by precision (inverse of prediction entropy). Friston's exact
     formulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from ..substrate.base import PredictionAdapter, Substrate


@dataclass
class PPTResult:
    ppt_value: float
    confidence: float
    components: Dict[str, float] = field(default_factory=dict)
    raw: Dict[str, float] = field(default_factory=dict)
    substrate_type: str = ""


def compute_ppt(
    substrate: Substrate,
    prediction_adapter: PredictionAdapter,
    weight_pde: float = 0.35,
    weight_hpet: float = 0.35,
    weight_pwce: float = 0.30,
) -> PPTResult:
    """Compute substrate-agnostic PPT score given an adapter for predictions."""
    weight_sum = weight_pde + weight_hpet + weight_pwce
    if not (0.999 <= weight_sum <= 1.001):
        raise ValueError(f"Weights must sum to 1.0; got {weight_sum}")

    if prediction_adapter.n_predictable_steps < 3:
        return PPTResult(
            ppt_value=0.0, confidence=0.0,
            components={"PDE": 0.0, "HPET": 0.0, "PWCE": 0.0},
            raw={"note": "too_few_predictable_steps"},
            substrate_type=substrate.substrate_type,
        )

    h_series = prediction_adapter.prediction_entropy_series()
    ce_series = prediction_adapter.cross_entropy_series()

    # ----- PDE: mean prediction entropy, normalized -----
    mean_h = float(h_series.mean())
    # Normalize: for transformer, max ≈ log(vocab); for fMRI, max ≈ log(diag_cov)
    # We use observed max as the normalization reference (within-substrate)
    max_h = float(h_series.max()) + 1e-8
    norm_h = float(np.clip(mean_h / max_h, 0.0, 1.0))
    # PPT: low entropy = high consciousness signature. Invert.
    pde_score = (1.0 - norm_h) * 100.0

    # ----- HPET: hierarchical prediction-error trajectory across groups -----
    # Use the substrate's activity matrix and group structure to compute
    # group-wise "prediction error" as the deviation of each group's
    # activity from a smooth (low-pass) reconstruction. Better-predicting
    # groups have lower deviation.
    activity = substrate.activity
    groups = substrate.node_groups
    if groups and len(groups) >= 3:
        group_dev = []
        for g_idx in groups:
            if len(g_idx) < 2:
                continue
            ga = activity[:, g_idx]
            # Smooth reconstruction = first-PC projection
            ga_c = ga - ga.mean(axis=0, keepdims=True)
            if ga_c.shape[0] < 2:
                continue
            U, S, Vt = np.linalg.svd(ga_c, full_matrices=False)
            if S.size == 0:
                continue
            # Reconstruct from top-1 component
            recon = U[:, :1] * S[0:1] @ Vt[:1, :]
            dev = float(np.linalg.norm(ga_c - recon) / (np.linalg.norm(ga_c) + 1e-8))
            group_dev.append(dev)
        if len(group_dev) >= 3:
            arr = np.array(group_dev, dtype=np.float64)
            idx = np.arange(len(arr), dtype=np.float64)
            if arr.std() > 1e-8:
                corr = float(np.corrcoef(idx, arr)[0, 1])
            else:
                corr = 0.0
            # Negative correlation (deviation decreases with depth) = good PPT
            hpet_raw = (-corr + 1.0) / 2.0
            hpet_score = hpet_raw * 100.0
        else:
            hpet_score = 0.0
            corr = 0.0
    else:
        hpet_score = 0.0
        corr = 0.0

    # ----- PWCE: precision-weighted cross-entropy -----
    # Precision = 1 / entropy at each step
    precision = 1.0 / (h_series + 1e-6)
    pwce_weighted = float((ce_series * precision[:len(ce_series)]).mean())
    # Lower weighted CE = better PWCE signature. Map via 1/(1+x).
    pwce_normalized = 1.0 / (1.0 + pwce_weighted)
    pwce_score = pwce_normalized * 100.0

    # ----- Aggregation -----
    ppt_raw = (
        weight_pde * pde_score
        + weight_hpet * hpet_score
        + weight_pwce * pwce_score
    )
    ppt_score = float(np.clip(ppt_raw, 0.0, 100.0))

    sub = np.array([pde_score, hpet_score, pwce_score])
    if sub.mean() > 0:
        cv = sub.std() / sub.mean()
        confidence = float(np.clip(1.0 - cv, 0.0, 1.0))
    else:
        confidence = 0.0

    return PPTResult(
        ppt_value=ppt_score,
        confidence=confidence,
        components={
            "PDE": float(pde_score),
            "HPET": float(hpet_score),
            "PWCE": float(pwce_score),
        },
        raw={
            "mean_entropy": float(mean_h),
            "norm_entropy": float(norm_h),
            "hpet_corr": float(corr),
            "pwce_weighted": float(pwce_weighted),
        },
        substrate_type=substrate.substrate_type,
    )
