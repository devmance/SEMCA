"""
Substrate-agnostic Integrated Information Theory (IIT) Φ measurement.

The substrate-agnostic operationalization:

  1. Build a node-similarity graph W from substrate.activity:
       W[i, j] = cosine(activity[:, i], activity[:, j])
       (correlation of node i's activity time-series with node j's)

  2. For each node-group G in substrate.node_groups, find the
     Minimum Information Partition (MIP) of W restricted to G via
     Normalized Cut (Shi & Malik 2000):

       Ncut(A, B) = cut(A,B) / vol(A) + cut(A,B) / vol(B)
       Φ_G = min_{(A,B) partitions G} Ncut(A,B) / 2

     The Ncut formulation prevents trivial partitions where one side
     is a single isolated node.

  3. For the system as a whole (treating each group as a single
     super-node), compute cross-group Φ analogously.

  4. Aggregate:
       PLG_Φ  = mean Φ_G across groups               (within-group integration)
       CGI_Φ  = Φ over the group-level graph        (between-group integration)
       PV     = std(Φ_G) across groups              (Φ variance signature)

  5. IIT_b = w_plg · PLG_Φ + w_cgi · CGI_Φ + w_pv · PV   ∈ [0, 100]

This formulation is identical on transformer substrates (where groups =
layers, nodes within group = attention heads) and on fMRI substrates
(where groups = brain networks, nodes within group = ROIs in that
network). Same math, both substrates.

Reference: Tononi, Boly, Massimini, Koch (2016), "Integrated information
theory: from consciousness to its physical substrate", Nat Rev Neurosci.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..substrate.base import Substrate


@dataclass
class IITPhiResult:
    """Result of substrate-agnostic IIT Φ measurement."""
    iit_value: float                       # final [0,100] aggregated score
    confidence: float                      # [0,1] from sub-score CV
    n_groups_scored: int
    components: Dict[str, float] = field(default_factory=dict)
    raw: Dict[str, float] = field(default_factory=dict)
    per_group_phi: Optional[List[float]] = None
    substrate_type: str = ""


def _node_similarity_matrix(activity: np.ndarray) -> np.ndarray:
    """
    Compute (N, N) node similarity matrix via cosine of activity columns.

    Each column of `activity` is one node's activity over time. Cosine
    similarity captures how similarly two nodes activate together —
    the substrate-agnostic analogue of "causal connection strength".
    """
    if activity.shape[0] < 2 or activity.shape[1] < 2:
        return np.zeros((activity.shape[1], activity.shape[1]))
    a = activity - activity.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(a, axis=0) + 1e-8
    a_norm = a / norms
    W = a_norm.T @ a_norm                                # (N, N)
    # Cosine can be negative due to anti-correlation; clip to non-negative
    # similarity to interpret as "shared variance" strength
    W = np.clip(W, 0.0, None)
    np.fill_diagonal(W, 0.0)
    return W


def _ncut(W: np.ndarray, a_idx: Tuple[int, ...], b_idx: Tuple[int, ...]) -> float:
    """Shi & Malik normalized cut of subset (A, B) on graph W."""
    cut = W[np.ix_(a_idx, b_idx)].sum()
    vol_a = W[a_idx, :].sum()
    vol_b = W[b_idx, :].sum()
    if vol_a == 0 or vol_b == 0:
        return float("inf")
    return cut * (1.0 / vol_a + 1.0 / vol_b)


def _min_ncut(
    W: np.ndarray,
    indices: List[int],
    max_full_search: int = 8,
    sample_per_size: int = 30,
    seed: int = 42,
) -> Tuple[float, int]:
    """
    Find the minimum Normalized Cut bipartition of the induced subgraph on
    `indices`. Returns (min_ncut, n_partitions_evaluated).

    For |indices| <= max_full_search: exhaustive enumeration with symmetry
    breaking (node 0 always in A).

    For |indices| > max_full_search: direct stratified sampling per
    partition size — never materialize C(n,k) for large n.
    """
    n = len(indices)
    if n < 2:
        return 0.0, 0
    if n == 2:
        a = (indices[0],); b = (indices[1],)
        return _ncut(W, a, b) / 2.0, 1

    # Anchor on indices[0] in A to break A↔B symmetry
    anchor = indices[0]
    others = indices[1:]
    n_other = len(others)
    rng = np.random.default_rng(seed)

    best = float("inf")
    n_eval = 0

    def eval_partition(a_idx: Tuple[int, ...], b_idx: Tuple[int, ...]) -> None:
        nonlocal best, n_eval
        v = _ncut(W, a_idx, b_idx)
        n_eval += 1
        if v < best:
            best = v

    if n <= max_full_search:
        # Exhaustive: iterate over all subsets-of-others to join anchor in A
        for k in range(0, n_other + 1):
            for combo_other in combinations(others, k):
                a_idx = (anchor,) + combo_other
                b_set = set(combo_other)
                b_idx = tuple(i for i in others if i not in b_set)
                if len(b_idx) == 0:
                    continue
                eval_partition(a_idx, b_idx)
    else:
        # Stratified sampling per size_a
        for k in range(0, n_other + 1):
            total = comb(n_other, k)
            if total == 0:
                continue
            if total <= sample_per_size:
                # Few enough — enumerate exactly
                for combo_other in combinations(others, k):
                    a_idx = (anchor,) + combo_other
                    b_set = set(combo_other)
                    b_idx = tuple(i for i in others if i not in b_set)
                    if len(b_idx) == 0:
                        continue
                    eval_partition(a_idx, b_idx)
            else:
                seen = set()
                target = sample_per_size
                attempts = 0
                while len(seen) < target and attempts < target * 10:
                    attempts += 1
                    picked = tuple(sorted(
                        rng.choice(n_other, size=k, replace=False).tolist()
                    ))
                    if picked in seen:
                        continue
                    seen.add(picked)
                    combo_other = tuple(others[i] for i in picked)
                    a_idx = (anchor,) + combo_other
                    b_set = set(combo_other)
                    b_idx = tuple(i for i in others if i not in b_set)
                    if len(b_idx) == 0:
                        continue
                    eval_partition(a_idx, b_idx)

    if best == float("inf"):
        return 0.0, n_eval
    return best / 2.0, n_eval


def compute_iit_phi(
    substrate: Substrate,
    weight_plg: float = 0.50,
    weight_cgi: float = 0.30,
    weight_pv: float = 0.20,
    calibration_alpha: float = 100.0,
    calibration_beta: float = 0.0,
    max_full_search: int = 8,
    sample_per_size: int = 30,
    seed: int = 42,
) -> IITPhiResult:
    """
    Compute substrate-agnostic IIT Φ on any Substrate.

    Three sub-scores:
      - Per-Layer/Group Φ (PLG):    mean Φ within each node group
      - Cross-Group Integration (CGI):  Φ over the group-level graph
      - Φ Variance (PV):            std of per-group Φ values

    All three are scaled to [0, 100] via the same empirical mapping
    constant (alpha = 100). This is symmetric across substrates: the
    same scaling is applied to transformer-attention-head Φ and to
    fMRI-ROI Φ, making the resulting scores directly comparable.
    """
    weight_sum = weight_plg + weight_cgi + weight_pv
    if not (0.999 <= weight_sum <= 1.001):
        raise ValueError(f"Weights must sum to 1.0; got {weight_sum}")

    activity = substrate.activity
    if activity.shape[0] < 2 or activity.shape[1] < 3:
        return IITPhiResult(
            iit_value=0.0, confidence=0.0, n_groups_scored=0,
            components={"PLG": 0.0, "CGI": 0.0, "PV": 0.0},
            raw={"note": "substrate_too_small"},
            substrate_type=substrate.substrate_type,
        )

    W = _node_similarity_matrix(activity)
    groups = substrate.node_groups
    if not groups:
        # If no natural grouping, treat the whole node set as one group
        # and skip CGI/PV computation
        all_idx = list(range(activity.shape[1]))
        phi_raw, _ = _min_ncut(W, all_idx, max_full_search, sample_per_size, seed)
        score = float(np.clip(calibration_alpha * phi_raw + calibration_beta, 0.0, 100.0))
        return IITPhiResult(
            iit_value=score, confidence=0.5, n_groups_scored=1,
            components={"PLG": score, "CGI": 0.0, "PV": 0.0},
            raw={"phi_raw": float(phi_raw), "note": "no_groups_available"},
            per_group_phi=[float(phi_raw)],
            substrate_type=substrate.substrate_type,
        )

    # ----- Per-group Φ (PLG) -----
    per_group_phi: List[float] = []
    for g_idx in groups:
        if len(g_idx) < 2:
            continue
        phi_raw, _ = _min_ncut(W, list(g_idx), max_full_search, sample_per_size, seed)
        per_group_phi.append(float(phi_raw))
    if not per_group_phi:
        return IITPhiResult(
            iit_value=0.0, confidence=0.0, n_groups_scored=0,
            components={"PLG": 0.0, "CGI": 0.0, "PV": 0.0},
            raw={"note": "no_group_with_2plus_nodes"},
            substrate_type=substrate.substrate_type,
        )
    plg_raw = float(np.mean(per_group_phi))
    plg_score = float(np.clip(calibration_alpha * plg_raw + calibration_beta, 0.0, 100.0))

    # ----- Cross-group integration (CGI) -----
    # Build group-level activity matrix: each group's representative is
    # the mean activity vector of its nodes
    if len(groups) >= 2:
        group_activity = np.stack([
            activity[:, g_idx].mean(axis=1) for g_idx in groups if len(g_idx) >= 1
        ], axis=1)                                       # (T, n_groups)
        W_g = _node_similarity_matrix(group_activity)
        cgi_raw, _ = _min_ncut(
            W_g, list(range(W_g.shape[0])),
            max_full_search, sample_per_size, seed,
        )
        cgi_score = float(np.clip(calibration_alpha * cgi_raw + calibration_beta, 0.0, 100.0))
    else:
        cgi_raw = 0.0
        cgi_score = 0.0

    # ----- Φ Variance (PV) -----
    if len(per_group_phi) >= 2:
        pv_raw = float(np.std(per_group_phi))
        # Empirically the per-group Φ std is typically 0.0-0.1
        pv_score = float(np.clip(pv_raw * 1000.0, 0.0, 100.0))
    else:
        pv_raw = 0.0
        pv_score = 0.0

    # ----- Aggregation -----
    iit_raw = (
        weight_plg * plg_score
        + weight_cgi * cgi_score
        + weight_pv * pv_score
    )
    iit_score = float(np.clip(iit_raw, 0.0, 100.0))

    sub = np.array([plg_score, cgi_score, pv_score])
    if sub.mean() > 0:
        cv = sub.std() / sub.mean()
        confidence = float(np.clip(1.0 - cv, 0.0, 1.0))
    else:
        confidence = 0.0

    return IITPhiResult(
        iit_value=iit_score,
        confidence=confidence,
        n_groups_scored=len(per_group_phi),
        components={
            "PLG": float(plg_score),
            "CGI": float(cgi_score),
            "PV": float(pv_score),
        },
        raw={
            "plg_raw": float(plg_raw),
            "cgi_raw": float(cgi_raw),
            "pv_raw": float(pv_raw),
            "n_nodes": float(activity.shape[1]),
            "n_timesteps": float(activity.shape[0]),
            "n_groups_total": float(len(groups)),
        },
        per_group_phi=per_group_phi,
        substrate_type=substrate.substrate_type,
    )
