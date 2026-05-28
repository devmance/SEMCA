"""
Substrate base class — the abstraction every consciousness theory calculator
operates on.

A Substrate represents any dynamical system as an activity matrix (T, N).
Theory calculators see only this abstraction, not the underlying transformer
or fMRI structure. This is the methodological move that makes the same
mathematical operationalization comparable across AI and human substrates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


class Substrate(ABC):
    """
    A substrate-agnostic representation of a dynamical system.

    The core observable is `activity`: a (T, N) array where
      T = number of time/sequence steps
      N = number of nodes (attention heads, brain ROIs, electrodes, etc.)

    Theory calculators read activity and optionally node_groups / adjacency.
    They never inspect substrate_type; that field is for logging and
    metadata-aware analysis only.

    Implementations included:
      - TransformerSubstrate: from transformer InternalsRecord
      - FMRISubstrate: from preprocessed fMRI BOLD signal

    Adding a new substrate type (e.g., EEG, spiking-network, MEG) requires
    only subclassing Substrate and implementing the abstract methods.
    """

    @property
    @abstractmethod
    def activity(self) -> np.ndarray:
        """The (T, N) activity matrix. Core observable."""

    @property
    @abstractmethod
    def substrate_type(self) -> str:
        """Identifier string: 'transformer', 'fmri', 'eeg', etc."""

    @property
    def n_timesteps(self) -> int:
        return self.activity.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.activity.shape[1]

    @property
    def node_groups(self) -> Optional[List[List[int]]]:
        """
        Optional grouping of nodes. Each inner list contains node indices
        belonging to one group.

        Examples:
          - Transformer 'head' unit: groups = layers (each contains the heads
            within that layer)
          - fMRI: groups = brain networks (each contains the ROIs in that
            network — DMN, salience, frontoparietal, etc.)

        Returns None if no natural grouping exists.
        """
        return None

    @property
    def adjacency(self) -> Optional[np.ndarray]:
        """
        Optional (N, N) connectivity matrix between nodes.

        Examples:
          - Transformer: averaged attention pattern across layers
          - fMRI: structural connectivity from DTI, or functional connectivity
            from resting-state correlation

        Returns None if no adjacency information is available.
        """
        return None

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Substrate-specific information: model name, atlas, sampling rate,
        subject ID, stimulus identifier, etc. Used for logging and
        downstream analysis grouping.
        """
        return {}

    def summary(self) -> str:
        """One-line summary for logs."""
        return (
            f"{self.__class__.__name__}("
            f"type={self.substrate_type}, "
            f"T={self.n_timesteps}, N={self.n_nodes}, "
            f"groups={'yes' if self.node_groups else 'no'}, "
            f"adjacency={'yes' if self.adjacency is not None else 'no'})"
        )


class PredictionAdapter(ABC):
    """
    Substrate-specific prediction interface required by PPT and FEP.

    PPT and FEP both measure prediction error. Transformers expose this
    directly via logit distributions. fMRI and EEG require building a
    forward model (e.g., lagged regression). This adapter encapsulates
    that substrate-specific prediction machinery.

    Returns per-position (or per-timestep) prediction quality measures:
      - prediction_entropy(t): entropy of the predicted distribution at t
      - cross_entropy(t): −log p(observed_{t+1} | predicted_{t+1})
    """

    @abstractmethod
    def prediction_entropy(self, t: int) -> float:
        """Per-position prediction-distribution entropy."""

    @abstractmethod
    def cross_entropy(self, t: int) -> float:
        """Per-position cross-entropy of predicted vs observed."""

    @property
    @abstractmethod
    def n_predictable_steps(self) -> int:
        """Number of timesteps for which predictions are available."""

    def prediction_entropy_series(self) -> np.ndarray:
        return np.array([
            self.prediction_entropy(t) for t in range(self.n_predictable_steps)
        ])

    def cross_entropy_series(self) -> np.ndarray:
        return np.array([
            self.cross_entropy(t) for t in range(self.n_predictable_steps)
        ])
