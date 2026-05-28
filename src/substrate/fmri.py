"""
FMRISubstrate — wraps fMRI BOLD signal as a Substrate.

Two operating modes:

  - 'evoked': activity[t, n] = beta weight for ROI n at trial/sentence t.
    Use this for sentence-level comparison to transformer per-prompt
    activations. Matches the Pereira et al. 2018 release format.

  - 'timeseries': activity[t, n] = BOLD signal in ROI n at TR t.
    Use this for within-stimulus temporal dynamics. Requires interpolating
    the slow hemodynamic response into a comparable timescale.

For cross-substrate analysis, 'evoked' mode is the operating choice:
one activation pattern per stimulus per subject, comparable to one
activation pattern per prompt for the transformer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import PredictionAdapter, Substrate


class FMRISubstrate(Substrate):
    """
    Substrate adapter from preprocessed fMRI data.

    Args:
        activity_matrix: (T, N) array.
          - 'evoked' mode: T = number of stimuli (e.g., 243 sentences),
            N = number of ROIs (e.g., 400 Schaefer parcels).
          - 'timeseries' mode: T = number of TRs (timepoints), N = ROIs.

        mode: 'evoked' | 'timeseries'
        atlas: name of the ROI parcellation (for metadata).
        subject_id: identifier for the subject.
        roi_names: optional list of N strings labeling each ROI.
        roi_networks: optional list of N strings giving each ROI's
          functional-network membership (DMN, SAL, FPN, etc.). Used for
          node_groups.
        tr: TR in seconds (only relevant for 'timeseries' mode).
        connectivity: optional (N, N) functional or structural connectivity.
        stimuli: optional list of T strings labeling each row (sentence
          text, condition name, etc.) for traceability.
    """

    def __init__(
        self,
        activity_matrix: np.ndarray,
        mode: str = "evoked",
        atlas: str = "schaefer-400-7networks",
        subject_id: str = "unknown",
        roi_names: Optional[List[str]] = None,
        roi_networks: Optional[List[str]] = None,
        tr: Optional[float] = None,
        connectivity: Optional[np.ndarray] = None,
        stimuli: Optional[List[str]] = None,
    ):
        if mode not in ("evoked", "timeseries"):
            raise ValueError(f"mode must be 'evoked' or 'timeseries'; got {mode}")
        if activity_matrix.ndim != 2:
            raise ValueError(f"activity_matrix must be 2D; got {activity_matrix.shape}")
        self._activity = activity_matrix.astype(np.float64)
        self.mode = mode
        self.atlas = atlas
        self.subject_id = subject_id
        self.roi_names = roi_names
        self.roi_networks = roi_networks
        self.tr = tr
        self._connectivity = connectivity
        self.stimuli = stimuli
        self._node_groups: Optional[List[List[int]]] = None

    @property
    def substrate_type(self) -> str:
        return "fmri"

    @property
    def activity(self) -> np.ndarray:
        return self._activity

    @property
    def node_groups(self) -> Optional[List[List[int]]]:
        if self._node_groups is not None:
            return self._node_groups
        if self.roi_networks is None:
            return None
        # Group ROIs by their functional network
        groups: Dict[str, List[int]] = {}
        for idx, net in enumerate(self.roi_networks):
            groups.setdefault(net, []).append(idx)
        self._node_groups = list(groups.values())
        return self._node_groups

    @property
    def adjacency(self) -> Optional[np.ndarray]:
        if self._connectivity is not None:
            return self._connectivity
        # Default: empirical correlation between ROI activity vectors
        if self.n_timesteps < 3:
            return None
        ax = self._activity - self._activity.mean(axis=0, keepdims=True)
        std = ax.std(axis=0, keepdims=True) + 1e-8
        ax = ax / std
        adj = (ax.T @ ax) / max(1, ax.shape[0] - 1)
        np.fill_diagonal(adj, 0.0)
        return adj

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "atlas": self.atlas,
            "subject_id": self.subject_id,
            "tr": self.tr,
            "n_rois": self.n_nodes,
            "n_trials_or_timepoints": self.n_timesteps,
            "has_roi_networks": self.roi_networks is not None,
            "has_connectivity": self._connectivity is not None,
        }


class FMRIPredictionAdapter(PredictionAdapter):
    """
    Prediction adapter for fMRI via lagged linear regression.

    For each ROI we train a forward model predicting next-step BOLD from
    the current activity pattern of all ROIs (a simple AR(1) over the
    ROI vector). This produces a predicted next-step distribution we
    can compute entropy and cross-entropy against the observed BOLD.

    This is the symmetric counterpart to TransformerPredictionAdapter's
    logit-based prediction — both give us a "what does the system predict
    next?" signal, but derived from substrate-appropriate machinery.
    """

    def __init__(self, activity_matrix: np.ndarray, lag: int = 1):
        if activity_matrix.shape[0] <= lag + 1:
            raise ValueError(
                f"activity_matrix has only {activity_matrix.shape[0]} timesteps; "
                f"need > {lag + 1} for AR({lag}) prediction"
            )
        self.activity = activity_matrix.astype(np.float64)
        self.lag = lag
        # Fit forward model: A[t+1] ≈ A[t] @ W
        # Closed-form linear regression
        X = self.activity[:-lag]                       # (T-lag, N)
        Y = self.activity[lag:]                        # (T-lag, N)
        # Solve W = (X^T X)^-1 X^T Y with regularization for numerical stability
        n = X.shape[1]
        reg = 1e-3 * np.eye(n)
        try:
            self.W = np.linalg.solve(X.T @ X + reg, X.T @ Y)
        except np.linalg.LinAlgError:
            self.W = np.linalg.lstsq(X, Y, rcond=None)[0]

        # Predicted next-step values
        self.predicted = X @ self.W                    # (T-lag, N)
        self.residual = Y - self.predicted             # (T-lag, N)
        # Empirical residual std per ROI (for Gaussian likelihood entropy)
        self.residual_std = self.residual.std(axis=0) + 1e-6

    @property
    def n_predictable_steps(self) -> int:
        return self.predicted.shape[0]

    def prediction_entropy(self, t: int) -> float:
        """
        Per-step prediction uncertainty proxy.

        Under a fixed-variance Gaussian likelihood the entropy would be
        constant. To produce a per-step uncertainty signal comparable
        across substrates, we use **leverage-adjusted prediction
        uncertainty**: the magnitude of the predicted ROI vector
        relative to its own residual std. High = the model is making a
        bold prediction (deviation from baseline); low = model is
        predicting near-mean (low information).

        We additionally normalize per ROI by the substrate-wide median
        leverage, so the resulting per-step value lives in approximately
        the same numerical range across substrates.
        """
        if not hasattr(self, "_entropy_series"):
            # Compute once and cache
            lev = np.abs(self.predicted) / (self.residual_std + 1e-8)
            # Per-step mean leverage in nats-ish range
            self._entropy_series = np.log1p(lev).mean(axis=1)
            # Normalize to make typical range comparable across substrates
            ref = float(np.median(self._entropy_series)) + 1e-8
            self._entropy_series = self._entropy_series / ref
        return float(self._entropy_series[t])

    def cross_entropy(self, t: int) -> float:
        """
        Cross-entropy of observed next-step vs predicted, in a substrate-
        comparable form.

        We compute the per-step squared-residual norm normalized by the
        substrate's typical residual scale:

          CE_t = ||observed_{t+1} - predicted_t||_W / median_residual_norm

        where W is the inverse residual covariance (diagonal). The
        normalization makes the magnitudes comparable to the
        transformer adapter's per-token log-prob range.
        """
        if not hasattr(self, "_ce_series"):
            diff = self.activity[self.lag:] - self.predicted    # (T-lag, N)
            weighted = (diff ** 2) / (self.residual_std ** 2 + 1e-8)
            per_step = weighted.sum(axis=1)                      # (T-lag,)
            ref = float(np.median(per_step)) + 1e-8
            self._ce_series = per_step / ref
        return float(self._ce_series[t])
