"""
Substrate-agnostic Quantum-Information-Theory-inspired (QIT) measurement.

Classical neural networks and brains both produce classical observables.
The QIT score here is an analogue — patterns that QIT proponents argue
distinguish a quantum-information-producing system from a classical
one, computed identically on transformer activations and fMRI signals.

Three sub-scores:

  1. **Long-Range Mutual Information (LRMI)**: between node activity
     vectors at distant positions in the activity matrix. For fMRI:
     across distant trials/timepoints. For transformer: across distant
     token positions.

  2. **Cross-Node Coherence (CNC)**: pairwise mutual information
     between different nodes' activity vectors. High CNC = many nodes
     "share information" beyond independent noise.

  3. **Bell-CHSH-style Correlation (BCHSH)**: for selected node pairs
     and "measurement settings" (node groups), compute the CHSH
     correlation value S. Classical bound ≤ 2; quantum bound ≤ 2√2.
     We report S normalized to [0, 100] across the [0, 2√2] range.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..substrate.base import Substrate


@dataclass
class QITResult:
    qit_value: float
    confidence: float
    components: Dict[str, float] = field(default_factory=dict)
    raw: Dict[str, float] = field(default_factory=dict)
    substrate_type: str = ""


def _mutual_information_1d(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    if len(x) < 3 or len(y) < 3:
        return 0.0
    try:
        xb = np.digitize(x, np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
        yb = np.digitize(y, np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1]))
    except Exception:
        return 0.0
    joint, _, _ = np.histogram2d(xb, yb, bins=n_bins)
    if joint.sum() == 0:
        return 0.0
    p_xy = joint / joint.sum()
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p_xy / (p_x * p_y + 1e-12)
        log_ratio = np.log(np.maximum(ratio, 1e-12))
    mi = (p_xy * log_ratio).sum()
    return float(max(0.0, mi))


def compute_qit(
    substrate: Substrate,
    weight_lrmi: float = 0.40,
    weight_cnc: float = 0.30,
    weight_bchsh: float = 0.30,
    max_pairs: int = 50,
    seed: int = 42,
) -> QITResult:
    """Compute substrate-agnostic QIT analogue score."""
    weight_sum = weight_lrmi + weight_cnc + weight_bchsh
    if not (0.999 <= weight_sum <= 1.001):
        raise ValueError(f"Weights must sum to 1.0; got {weight_sum}")

    activity = substrate.activity
    T, N = activity.shape
    if T < 8 or N < 4:
        return QITResult(
            qit_value=0.0, confidence=0.0,
            components={"LRMI": 0.0, "CNC": 0.0, "BCHSH": 0.0},
            raw={"note": "substrate_too_small"},
            substrate_type=substrate.substrate_type,
        )

    rng = np.random.default_rng(seed)

    # ----- LRMI: MI between node activity at distant timesteps -----
    # For each node, MI(activity[:-d], activity[d:]) for d in distance set
    distances = [d for d in (1, 3, 5, 10) if d < T]
    if not distances:
        lrami_raw = 0.0
    else:
        per_node_lrmi = []
        sample_nodes = rng.choice(N, size=min(N, 50), replace=False)
        for n_idx in sample_nodes:
            col = activity[:, n_idx]
            if col.std() < 1e-8:
                continue
            mis = []
            for d in distances:
                mi = _mutual_information_1d(col[:-d], col[d:])
                mis.append(mi)
            if mis:
                per_node_lrmi.append(float(np.mean(mis)))
        if per_node_lrmi:
            lrami_raw = float(np.mean(per_node_lrmi))
        else:
            lrami_raw = 0.0
    # Map typical 0.0-0.5 MI to 0-100
    lrami_score = float(np.clip(lrami_raw / 0.3 * 100.0, 0.0, 100.0))

    # ----- CNC: pairwise MI between different nodes -----
    sample_nodes = rng.choice(N, size=min(N, 30), replace=False)
    pair_mis: List[float] = []
    pairs_evaluated = 0
    for i in range(len(sample_nodes)):
        if pairs_evaluated >= max_pairs:
            break
        for j in range(i + 1, len(sample_nodes)):
            if pairs_evaluated >= max_pairs:
                break
            x = activity[:, sample_nodes[i]]
            y = activity[:, sample_nodes[j]]
            if x.std() < 1e-8 or y.std() < 1e-8:
                continue
            pair_mis.append(_mutual_information_1d(x, y))
            pairs_evaluated += 1
    if pair_mis:
        cnc_raw = float(np.mean(pair_mis))
        cnc_score = float(np.clip(cnc_raw / 0.3 * 100.0, 0.0, 100.0))
    else:
        cnc_raw = 0.0
        cnc_score = 0.0

    # ----- BCHSH: CHSH-style correlation -----
    # Pick 2 positions A, B (far apart timesteps) and 4 "measurement
    # settings" (nodes a1, a2 at position A; b1, b2 at position B).
    # Binarize each via sign(value - median).
    groups = substrate.node_groups
    if groups and len(groups) >= 4 and T >= 6 and N >= 8:
        A_t, B_t = T // 3, 2 * T // 3
        # Use the first node of each of 4 different groups
        chosen_groups = groups[:4]
        a1, a2 = chosen_groups[0][0], chosen_groups[1][0]
        b1, b2 = chosen_groups[2][0], chosen_groups[3][0]
        # Build outcome vectors via binarizing each (node, position-window) view
        win = max(3, T // 10)
        ws_a, we_a = max(0, A_t - win), min(T, A_t + win + 1)
        ws_b, we_b = max(0, B_t - win), min(T, B_t + win + 1)

        def outcome(node, ws, we):
            sub = activity[ws:we, node]
            med = np.median(sub)
            return np.sign(sub - med)

        outcomes = {
            (a1, "A"): outcome(a1, ws_a, we_a),
            (a2, "A"): outcome(a2, ws_a, we_a),
            (b1, "B"): outcome(b1, ws_b, we_b),
            (b2, "B"): outcome(b2, ws_b, we_b),
        }
        # Ensure all vectors same length for correlation
        L = min(len(v) for v in outcomes.values())
        outcomes = {k: v[:L] for k, v in outcomes.items()}

        def E(h1, side1, h2, side2):
            v1 = outcomes[(h1, side1)]
            v2 = outcomes[(h2, side2)]
            if len(v1) == 0:
                return 0.0
            return float((v1 * v2).mean())

        S = abs(
            E(a1, "A", b1, "B")
            + E(a1, "A", b2, "B")
            + E(a2, "A", b1, "B")
            - E(a2, "A", b2, "B")
        )
        S_clamped = min(S, 2 * np.sqrt(2))
        bchsh_score = float(np.clip(S_clamped / (2 * np.sqrt(2)) * 100.0, 0.0, 100.0))
    else:
        bchsh_score = 0.0
        S = 0.0

    # ----- Aggregation -----
    qit_raw = (
        weight_lrmi * lrami_score
        + weight_cnc * cnc_score
        + weight_bchsh * bchsh_score
    )
    qit_score = float(np.clip(qit_raw, 0.0, 100.0))

    sub = np.array([lrami_score, cnc_score, bchsh_score])
    if sub.mean() > 0:
        cv = sub.std() / sub.mean()
        confidence = float(np.clip(1.0 - cv, 0.0, 1.0))
    else:
        confidence = 0.0

    return QITResult(
        qit_value=qit_score,
        confidence=confidence,
        components={
            "LRMI": float(lrami_score),
            "CNC": float(cnc_score),
            "BCHSH": float(bchsh_score),
        },
        raw={
            "mean_lrmi": float(lrami_raw),
            "mean_cnc": float(cnc_raw),
            "chsh_s_value": float(S),
            "n_pairs_evaluated": float(pairs_evaluated),
        },
        substrate_type=substrate.substrate_type,
    )
