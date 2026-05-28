"""
Build an FMRISubstrate from a LeBel et al. 2023 subject × story.

Provides one helper, `build_lebel_substrate`, that ties together:
  1. TextGrid → Sentence list (with timing)
  2. HDF5 BOLD → (n_TRs, n_voxels)
  3. Per-sentence evoked extraction → (n_sentences, n_voxels)
  4. Voxel → functional parcel clustering (K-means) for tractability
  5. FMRISubstrate construction with parcel-level node_groups

Returns:
  - FMRISubstrate ready for any of the 7 substrate-agnostic theory
    calculators
  - The corresponding sentence-text list (for AI substrate alignment)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..substrate.fmri import FMRISubstrate
from .bold_loader import (
    extract_per_sentence_evoked,
    load_subject_story_bold,
)
from .textgrid_loader import Sentence, parse_textgrid


def _kmeans_numpy(
    X: np.ndarray,
    k: int,
    n_iter: int = 30,
    seed: int = 42,
) -> np.ndarray:
    """
    Pure-numpy K-means with random initialization (fast for large data).

    No sklearn dependency. Deterministic given seed. Designed to be
    fast on large voxel sets (~80K voxels × ~100 features).

    Args:
        X: (n_samples, n_features) data matrix.
        k: number of clusters.
        n_iter: max Lloyd's-iteration steps.
        seed: RNG seed.

    Returns:
        labels: (n_samples,) integer cluster assignments in [0, k).
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    k = min(k, n)

    # Random init: pick k random samples as initial centers
    init_idx = rng.choice(n, size=k, replace=False)
    centers = X[init_idx].copy()

    # Lloyd's iteration with vectorized distance computation via
    # ||a-b||² = ||a||² + ||b||² - 2 a·b, which is O(n*k*d) but with
    # GEMM-friendly matrix product
    X_norm = (X ** 2).sum(axis=1, keepdims=True)        # (n, 1)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        C_norm = (centers ** 2).sum(axis=1, keepdims=True).T  # (1, k)
        # Cross-term: X @ centers.T → (n, k)
        cross = X @ centers.T
        dists = X_norm + C_norm - 2.0 * cross                  # (n, k)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = X[mask].mean(axis=0)
    return labels


def _kmeans_cluster_voxels(
    evoked: np.ndarray,
    n_clusters: int = 200,
    n_networks: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Cluster voxels into K functional parcels via numpy K-means on each
    voxel's evoked-activity profile. Super-cluster parcels into N
    functional networks.

    Returns:
        (parcel_activity, parcel_cluster_ids, network_labels):
          - parcel_activity: (n_sentences, n_clusters) cluster mean activity
          - parcel_cluster_ids: list of cluster IDs (0..n_clusters-1)
          - network_labels: each parcel's super-cluster network ("net_0", "net_1", …)
    """
    n_sentences, n_voxels = evoked.shape
    voxel_features = evoked.T  # (n_voxels, n_sentences)
    n_clusters = min(n_clusters, n_voxels)

    # For large voxel sets, downsample features before K-means to keep
    # the pairwise-distance computation manageable
    max_features_for_kmeans = 200
    if n_sentences > max_features_for_kmeans:
        # Sub-sample sentence-features for clustering (still uses all when computing means)
        stride = max(1, n_sentences // max_features_for_kmeans)
        cluster_features = voxel_features[:, ::stride][:, :max_features_for_kmeans]
    else:
        cluster_features = voxel_features

    voxel_labels = _kmeans_numpy(cluster_features, n_clusters, seed=seed)

    parcel_activity = np.zeros((n_sentences, n_clusters), dtype=np.float64)
    for cid in range(n_clusters):
        mask = voxel_labels == cid
        if mask.any():
            parcel_activity[:, cid] = voxel_features[mask].mean(axis=0)

    # Super-cluster parcels into networks
    n_networks = min(n_networks, n_clusters)
    parcel_features = parcel_activity.T  # (n_clusters, n_sentences)
    if n_sentences > max_features_for_kmeans:
        stride = max(1, n_sentences // max_features_for_kmeans)
        parcel_cluster_features = parcel_features[:, ::stride][:, :max_features_for_kmeans]
    else:
        parcel_cluster_features = parcel_features
    network_assignments = _kmeans_numpy(parcel_cluster_features, n_networks, seed=seed)
    network_labels = [f"net_{n}" for n in network_assignments]

    return parcel_activity, list(range(n_clusters)), network_labels


def build_lebel_substrate(
    dataset_root: str,
    subject_id: str,
    story_name: str,
    tr_seconds: float = 2.0,
    hrf_lag_sec: float = 5.0,
    n_parcels: int = 200,
    n_networks: int = 10,
    seed: int = 42,
) -> Tuple[FMRISubstrate, List[str]]:
    """
    Build one FMRISubstrate from a LeBel et al. 2023 subject × story.

    Voxels are clustered into `n_parcels` functional parcels via K-means
    on each voxel's evoked-activity profile. The parcels are then
    super-clustered into `n_networks` functional networks, providing the
    `node_groups` structure that IIT and GWT need for tractable analysis.

    With default n_parcels=200, n_networks=10:
      - activity shape: (n_sentences, 200)
      - node_groups: 10 networks of ~20 parcels each
      - IIT bipartition search is fast (10 groups × 20 nodes)

    Args:
        dataset_root: root of the ds003020 download.
        subject_id: e.g., 'UTS01'.
        story_name: e.g., 'adollshouse'.
        tr_seconds: BOLD sampling interval (LeBel dataset uses 2.0).
        hrf_lag_sec: hemodynamic response delay for evoked extraction.
        n_parcels: how many functional parcels to cluster voxels into.
        n_networks: how many super-clusters of parcels.
        seed: K-means RNG seed for reproducibility.

    Returns:
        (FMRISubstrate, sentence_texts).
    """
    root = Path(dataset_root)
    h5_path = root / "derivatives" / "preprocessed_data" / subject_id / f"{story_name}.hf5"
    tg_path = root / "derivatives" / "TextGrids" / f"{story_name}.TextGrid"

    sentences = parse_textgrid(str(tg_path))
    if not sentences:
        raise RuntimeError(f"No sentences extracted from {tg_path}")

    bold, tr = load_subject_story_bold(str(h5_path), tr_seconds=tr_seconds)
    evoked = extract_per_sentence_evoked(
        bold, sentences, tr_seconds=tr, hrf_lag_sec=hrf_lag_sec,
    )

    # Cluster voxels into parcels for tractable theory computation
    parcel_activity, parcel_ids, network_labels = _kmeans_cluster_voxels(
        evoked, n_clusters=n_parcels, n_networks=n_networks, seed=seed,
    )

    substrate = FMRISubstrate(
        activity_matrix=parcel_activity,
        mode="evoked",
        atlas=f"lebel-kmeans-{n_parcels}p-{n_networks}n",
        subject_id=subject_id,
        roi_names=[f"parcel_{i}" for i in parcel_ids],
        roi_networks=network_labels,
        tr=tr,
        stimuli=[s.text for s in sentences],
    )
    return substrate, [s.text for s in sentences]
