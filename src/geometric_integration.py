"""
Information-geometric integration of seven mathematical consciousness theories.

Replaces the naive arithmetic mean of 7 theory scores with a Riemannian mean
on a Fisher-information manifold. The Fisher-Rao metric tensor is constructed
from the empirical covariance of theory-score coordinates. Geodesic distances
in this metric, combined with a literature-based theoretical-compatibility
matrix, determine dynamic theoretical weights.

Produces five geometric integration outputs per (substrate, stimulus):

  - unified_score          : Riemannian-mean unified consciousness score [0,100]
  - theoretical_consensus  : how aligned the 7 theories are [0,1]
  - geometric_coherence    : structural quality of the manifold [0,1]
  - manifold_curvature     : scalar curvature signed in [-1,1]
  - theoretical_weights    : dict of dynamic weights summing to 1

API: `geometric_integration(theory_scores)` accepts dict[str, float] keyed by
theory name (IIT, GWT, AST, HOT, PPT, QIT, FEP). Returns an immutable
dataclass result. Designed for batch use across thousands of cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import scipy.linalg as linalg


THEORIES = ["IIT", "GWT", "AST", "HOT", "PPT", "QIT", "FEP"]


# Literature-based theoretical-compatibility prior. Each cell is a soft estimate
# of how compatible two theories' claims are at the conceptual level (e.g.,
# IIT-FEP both center on information integration/minimization → 0.88; QIT-AST
# share less mathematical structure → 0.46). The matrix is symmetric. Used
# only as a prior on the interaction tensor; the per-cell outputs are
# dominated by the empirical Fisher-information metric tensor.
THEORY_COMPAT = np.array([
    # IIT   GWT   AST   HOT   PPT   QIT   FEP
    [1.00, 0.85, 0.72, 0.78, 0.91, 0.63, 0.88],  # IIT
    [0.85, 1.00, 0.93, 0.69, 0.74, 0.52, 0.67],  # GWT
    [0.72, 0.93, 1.00, 0.84, 0.61, 0.46, 0.58],  # AST
    [0.78, 0.69, 0.84, 1.00, 0.65, 0.51, 0.59],  # HOT
    [0.91, 0.74, 0.61, 0.65, 1.00, 0.68, 0.92],  # PPT
    [0.63, 0.52, 0.46, 0.51, 0.68, 1.00, 0.79],  # QIT
    [0.88, 0.67, 0.58, 0.59, 0.92, 0.79, 1.00],  # FEP
])


@dataclass(frozen=True)
class GeometricResult:
    """Result of one (substrate, stimulus) geometric integration."""
    unified_score: float
    theoretical_consensus: float
    geometric_coherence: float
    manifold_curvature: float
    integration_confidence: float
    theoretical_weights: Dict[str, float] = field(default_factory=dict)


def _score_to_coordinate(score: float) -> float:
    """Map 0-100 score → [-1, 1] manifold coordinate via tanh."""
    return float(np.tanh(2.0 * (score / 100.0) - 1.0))


def _score_to_probability(score: float) -> float:
    """Sigmoid transform: 0-100 score → (0,1) probability."""
    return 1.0 / (1.0 + np.exp(-(score - 50.0) / 15.0))


def _probability_to_score(p: float) -> float:
    """Inverse sigmoid: probability → 0-100 score, clipped."""
    p_clipped = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    score = 50.0 + 15.0 * np.log(p_clipped / (1.0 - p_clipped))
    return float(np.clip(score, 0.0, 100.0))


def _build_manifold_coordinates(
    theory_scores: Dict[str, float],
    n_dim: int = 7,
) -> np.ndarray:
    """For each theory, produce a 7-dim coordinate vector on the consciousness manifold.

    Returns (n_theories, n_dim) matrix.
    """
    coords = np.zeros((len(theory_scores), n_dim))
    for i, theory in enumerate(theory_scores.keys()):
        primary = _score_to_coordinate(theory_scores[theory])
        coords[i, i] = primary
        # Add interaction-weighted off-diagonal influences
        for j in range(n_dim):
            if j != i:
                coords[i, j] = THEORY_COMPAT[i, j] * primary * 0.3
    return coords


def _fisher_metric_tensor(
    coordinates: np.ndarray,
    regularization: float = 1e-4,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Fisher information metric tensor from coordinate covariance.

    Inverse of empirical covariance, regularized + projected onto positive-
    definite cone via eigenvalue clipping.
    """
    cov = np.cov(coordinates, rowvar=False)
    cov = cov + regularization * np.eye(coordinates.shape[1])
    try:
        metric = linalg.inv(cov)
    except linalg.LinAlgError:
        metric = linalg.pinv(cov)
    eigvals, eigvecs = linalg.eigh(metric)
    eigvals = np.maximum(eigvals, tolerance)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _geodesic_distances(
    coordinates: np.ndarray, metric: np.ndarray,
) -> np.ndarray:
    """Pairwise geodesic distances in the Fisher-Rao metric.

    Returns (n_theories, n_theories) symmetric matrix.
    """
    n = coordinates.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = coordinates[i] - coordinates[j]
            D[i, j] = D[j, i] = float(np.sqrt(diff @ metric @ diff))
    return D


def _scalar_curvature(metric: np.ndarray, n_dim: int) -> float:
    """Scalar curvature approximation from log-det of metric tensor."""
    det = float(linalg.det(metric))
    if det <= 0:
        return 0.0
    kappa = -np.log(det) / n_dim
    return float(np.tanh(kappa))


def _interaction_tensor(
    coordinates: np.ndarray, theory_indices: np.ndarray,
) -> np.ndarray:
    """Per-pair interaction strength = theoretical compatibility × proximity.

    Returns (n, n) matrix; row sums approximate per-theory total interaction.
    """
    n = coordinates.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 1.0
            else:
                compat = THEORY_COMPAT[theory_indices[i], theory_indices[j]]
                dist = float(np.linalg.norm(coordinates[i] - coordinates[j]))
                M[i, j] = compat * np.exp(-dist)
    return M


def _theoretical_alignment(interaction: np.ndarray) -> float:
    """Eigenvalue-based alignment: low spectrum variance = high alignment."""
    eigvals = linalg.eigvalsh(interaction)
    var = float(np.var(eigvals))
    max_var = (len(eigvals) - 1) / 4.0
    if max_var <= 0:
        return 1.0
    return float(np.clip(1.0 - var / max_var, 0.0, 1.0))


def _dynamic_weights(
    probabilities: np.ndarray, geodesic: np.ndarray,
    interaction: np.ndarray, theory_names: list,
) -> Dict[str, float]:
    """Per-theory dynamic weights from probability × interaction × centrality."""
    n = len(probabilities)
    base = probabilities
    interaction_factor = interaction.mean(axis=1)
    # Centrality: exp(-mean geodesic distance to all others)
    np.fill_diagonal(geodesic, 0)
    mean_dist = geodesic.sum(axis=1) / max(n - 1, 1)
    centrality = np.exp(-mean_dist)
    combined = base * interaction_factor * centrality
    total = combined.sum()
    if total > 0:
        normed = combined / total
    else:
        normed = np.ones(n) / n
    return dict(zip(theory_names, normed.astype(float)))


def _theoretical_consensus(
    geodesic: np.ndarray, theoretical_alignment: float,
    probabilities: np.ndarray,
) -> float:
    """Three-factor consensus mean: distance coherence, alignment, prob variance."""
    factors = []
    nonzero = geodesic[geodesic > 0]
    if nonzero.size > 0:
        avg, mx = nonzero.mean(), nonzero.max()
        factors.append(1.0 - avg / mx if mx > 0 else 1.0)
    factors.append(theoretical_alignment)
    prob_var = float(np.var(probabilities))
    factors.append(1.0 - min(1.0, prob_var / 0.25))
    return float(np.clip(np.mean(factors), 0.0, 1.0))


def _geometric_coherence(
    geodesic: np.ndarray, curvature: float, coordinates: np.ndarray,
) -> float:
    """Three-factor coherence: distance dispersion, curvature, coord conditioning."""
    factors = []
    nonzero = geodesic[geodesic > 0]
    if nonzero.size > 0:
        m, s = nonzero.mean(), nonzero.std()
        factors.append(1.0 / (1.0 + s / m) if m > 0 else 1.0)
    factors.append(float(np.exp(-abs(curvature))))
    try:
        cond = float(np.linalg.cond(coordinates))
        factors.append(1.0 / (1.0 + np.log(max(cond, 1e-9))))
    except Exception:
        factors.append(0.5)
    return float(np.clip(np.mean(factors), 0.0, 1.0))


def _riemannian_mean_score(
    probabilities: np.ndarray, weights: np.ndarray,
) -> float:
    """Weighted average probability → logit → 0-100 score."""
    p = float(np.sum(weights * probabilities))
    return _probability_to_score(p)


def geometric_integration(
    theory_scores: Dict[str, float],
    return_weights: bool = True,
) -> GeometricResult:
    """Integrate 7 consciousness theory scores into geometric outputs.

    Args:
        theory_scores: dict keyed by theory name (must contain all 7 theories;
            order is taken from the dict, not THEORIES constant — for batched
            use, pass in canonical order).
        return_weights: include per-theory dynamic weights in the result.

    Returns:
        GeometricResult with five scalar fields and a weights dict.
    """
    names = list(theory_scores.keys())
    # Indices in the global THEORY_COMPAT matrix
    try:
        theory_indices = np.array([THEORIES.index(n) for n in names])
    except ValueError as e:
        raise ValueError(f"Unknown theory name in {names}; expected from {THEORIES}") from e

    coordinates = _build_manifold_coordinates(theory_scores)
    metric = _fisher_metric_tensor(coordinates)
    geodesic = _geodesic_distances(coordinates, metric)
    curvature = _scalar_curvature(metric, n_dim=coordinates.shape[1])
    interaction = _interaction_tensor(coordinates, theory_indices)
    alignment = _theoretical_alignment(interaction)

    probabilities = np.array([_score_to_probability(theory_scores[n]) for n in names])
    weights = _dynamic_weights(probabilities, geodesic.copy(), interaction, names)
    weight_vec = np.array([weights[n] for n in names])

    unified = _riemannian_mean_score(probabilities, weight_vec)
    consensus = _theoretical_consensus(geodesic, alignment, probabilities)
    coherence = _geometric_coherence(geodesic, curvature, coordinates)

    # Integration confidence: mean of consensus, coherence, theory-count factor, structure quality
    theory_factor = min(1.0, len(theory_scores) / 7.0)
    structure = float(np.exp(-abs(curvature)))
    confidence = float(np.clip(np.mean([consensus, coherence, theory_factor, structure]),
                              0.0, 1.0))

    return GeometricResult(
        unified_score=unified,
        theoretical_consensus=consensus,
        geometric_coherence=coherence,
        manifold_curvature=curvature,
        integration_confidence=confidence,
        theoretical_weights=weights if return_weights else {},
    )


def naive_unified_score(theory_scores: Dict[str, float]) -> float:
    """Arithmetic mean across the 7 theory scores."""
    return float(np.mean(list(theory_scores.values())))


def batch_geometric_integration(
    theory_scores_batch: list,
) -> list:
    """Apply geometric_integration to many cells.

    Args:
        theory_scores_batch: list of dict[theory_name, score]

    Returns:
        list of GeometricResult
    """
    return [geometric_integration(ts) for ts in theory_scores_batch]
