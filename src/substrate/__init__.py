"""
Substrate abstraction for substrate-agnostic consciousness theory measurement.

A Substrate is any observable dynamical system expressed as:
  - activity: (T, N) array — T timesteps × N nodes
  - node_groups (optional): list of node-index lists (layers, brain networks, etc.)
  - adjacency (optional): (N, N) connectivity matrix
  - metadata: substrate-specific information

Consciousness theory calculators operate on Substrates, not on
substrate-specific structures, allowing the same mathematical
operationalization to be applied to transformer activations, fMRI BOLD,
EEG signals, or any other dynamical system.

See PHASE3_DESIGN.md for the architectural rationale.
"""

from .base import Substrate, PredictionAdapter
from .transformer import TransformerSubstrate, TransformerPredictionAdapter
from .fmri import FMRISubstrate, FMRIPredictionAdapter

__all__ = [
    "Substrate",
    "PredictionAdapter",
    "TransformerSubstrate",
    "TransformerPredictionAdapter",
    "FMRISubstrate",
    "FMRIPredictionAdapter",
]
