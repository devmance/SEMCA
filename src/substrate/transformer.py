"""
TransformerSubstrate — wraps a transformer InternalsRecord as a Substrate.

Three unit choices for the activity matrix:
  - 'head'    : (seq_len, n_layers × n_heads) of attention-received per head
  - 'layer'   : (seq_len, n_layers) of hidden-state L2 norms per layer
  - 'token'   : (seq_len, hidden_dim) of final-layer hidden states

The 'head' unit is the closest analogue to brain ROIs: each head is a
semi-autonomous information channel, and stacking heads gives a node set
comparable to fMRI ROIs (both are O(100-1000) nodes).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..model_loader import InternalsRecord
from .base import PredictionAdapter, Substrate


class TransformerSubstrate(Substrate):
    """
    Substrate adapter from a transformer InternalsRecord.

    Args:
        record: InternalsRecord from model_loader
        unit: 'head' | 'layer' | 'token'
          - 'head': activity[t, l*H+h] = total attention received by head h
            in layer l at position t (summed over source positions).
            Most directly comparable to fMRI ROI activity per timepoint.
          - 'layer': activity[t, l] = L2 norm of hidden state at position t,
            layer l. Coarser; comparable to fMRI network-level activity.
          - 'token': activity[t, d] = final-layer hidden state at position t,
            dimension d. Useful for QIT/HOT analyses on representation space.
    """

    def __init__(self, record: InternalsRecord, unit: str = "head"):
        if unit not in ("head", "layer", "token"):
            raise ValueError(f"unit must be 'head', 'layer', or 'token'; got {unit}")
        self.record = record
        self.unit = unit
        self._activity: Optional[np.ndarray] = None
        self._node_groups: Optional[List[List[int]]] = None
        self._adjacency: Optional[np.ndarray] = None

    @property
    def substrate_type(self) -> str:
        return "transformer"

    @property
    def activity(self) -> np.ndarray:
        if self._activity is None:
            self._activity = self._compute_activity()
        return self._activity

    def _compute_activity(self) -> np.ndarray:
        rec = self.record
        if self.unit == "head":
            # (seq, n_layers * n_heads): attention received by each head
            per_layer_received = []
            for layer_attn in rec.attentions:
                # (1, n_heads, seq, seq) → sum source axis → (n_heads, seq)
                received = layer_attn[0].float().sum(dim=-2)
                per_layer_received.append(received.detach().cpu().numpy())
            # stack: (n_layers, n_heads, seq) → reshape to (seq, n_layers*n_heads)
            stacked = np.stack(per_layer_received)  # (L, H, T)
            L, H, T = stacked.shape
            return stacked.transpose(2, 0, 1).reshape(T, L * H)

        if self.unit == "layer":
            per_layer_norm = []
            for h in rec.hidden_states:
                norms = h[0].float().norm(dim=-1)         # (seq,)
                per_layer_norm.append(norms.detach().cpu().numpy())
            return np.stack(per_layer_norm).T              # (seq, n_layers+1)

        # unit == "token"
        return rec.hidden_states[-1][0].float().detach().cpu().numpy()  # (seq, hidden_dim)

    @property
    def node_groups(self) -> Optional[List[List[int]]]:
        if self._node_groups is not None:
            return self._node_groups
        if self.unit == "head":
            # Each layer is a group; group g contains heads g*H..g*H+H-1
            n_heads = self.record.n_heads
            n_layers = self.record.n_layers
            self._node_groups = [
                list(range(l * n_heads, (l + 1) * n_heads))
                for l in range(n_layers)
            ]
            return self._node_groups
        return None

    @property
    def adjacency(self) -> Optional[np.ndarray]:
        if self._adjacency is not None:
            return self._adjacency
        if self.unit == "head":
            # Build a head-to-head adjacency via attention-pattern similarity
            # (Pearson correlation between heads' received-attention vectors)
            act = self.activity                            # (T, N)
            ax = act - act.mean(axis=0, keepdims=True)
            std = ax.std(axis=0, keepdims=True) + 1e-8
            ax = ax / std
            adj = (ax.T @ ax) / max(1, ax.shape[0] - 1)    # (N, N)
            np.fill_diagonal(adj, 0.0)
            self._adjacency = adj
            return adj
        return None

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.record.model_name,
            "prompt": self.record.prompt,
            "unit": self.unit,
            "n_layers": self.record.n_layers,
            "n_heads": self.record.n_heads,
            "seq_len": self.record.seq_len,
            "hidden_dim": self.record.hidden_dim,
        }


class TransformerPredictionAdapter(PredictionAdapter):
    """Prediction adapter using transformer logits (PPT, FEP)."""

    def __init__(self, record: InternalsRecord):
        self.record = record
        # Pre-compute logit distributions
        self.log_probs = F.log_softmax(record.logits[0].float(), dim=-1)
        self.probs = self.log_probs.exp()
        # Per-position prediction entropy
        self._entropy = -(self.probs * self.log_probs).sum(dim=-1).cpu().numpy()
        # Cross-entropy: target is next token
        input_ids = record.input_ids[0]
        if input_ids.shape[0] >= 2:
            targets = input_ids[1:]
            target_lp = self.log_probs[:-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            self._ce = -target_lp.cpu().numpy()
        else:
            self._ce = np.array([])

    @property
    def n_predictable_steps(self) -> int:
        return len(self._ce)

    def prediction_entropy(self, t: int) -> float:
        return float(self._entropy[t])

    def cross_entropy(self, t: int) -> float:
        return float(self._ce[t])
