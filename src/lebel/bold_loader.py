"""
HDF5 BOLD loader for LeBel et al. 2023.

Each subject × story is stored as a single HDF5 file (~150MB - 1.4GB)
under `derivatives/preprocessed_data/UTS<NN>/<story>.hf5`.

The file structure (per LeBel et al. 2023):
  - 'data': (n_TRs, n_voxels) preprocessed BOLD signal
            TR ≈ 2 seconds (story-dependent)
  - Other keys may include subject metadata, masks, etc.

This module:
  1. Loads BOLD time-series from one HDF5 file
  2. Extracts per-sentence evoked patterns by averaging BOLD during each
     sentence's time window, with hemodynamic-response-function lag
     compensation (~4-6 sec)
  3. Outputs (n_sentences, n_voxels) matrix → input to FMRISubstrate
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from .textgrid_loader import Sentence


@dataclass
class LebelSubstrate:
    """One subject × story BOLD substrate ready for analysis."""
    subject_id: str
    story_name: str
    activity: np.ndarray            # (n_sentences, n_voxels)
    sentences: List[Sentence]
    n_voxels: int
    n_sentences: int
    tr_seconds: float


def load_subject_story_bold(
    h5_path: str,
    tr_seconds: float = 2.0,
) -> Tuple[np.ndarray, float]:
    """
    Load BOLD time-series from a LeBel preprocessed_data HDF5 file.

    Returns:
        (bold, tr): bold is (n_TRs, n_voxels); tr is sampling interval in seconds.

    The HDF5 key is typically 'data' but a fallback scan tries the most
    common variants.
    """
    if h5py is None:
        raise ImportError("h5py is required. `pip install h5py`")
    p = Path(h5_path)
    if not p.exists():
        raise FileNotFoundError(f"HDF5 file not found: {p}")

    with h5py.File(str(p), "r") as f:
        # Try common keys
        for key in ("data", "bold", "responses", "preprocessed", "Y"):
            if key in f:
                bold = np.asarray(f[key], dtype=np.float64)
                # Heuristic: (n_TRs, n_voxels) — TRs typically << voxels
                if bold.ndim == 2 and bold.shape[0] < bold.shape[1]:
                    return bold, tr_seconds
                # If it looks like (n_voxels, n_TRs), transpose
                if bold.ndim == 2 and bold.shape[1] < bold.shape[0]:
                    return bold.T, tr_seconds
                # 1D — single voxel, unusual
                return bold, tr_seconds
        # Fall back: take the first 2-D dataset we find
        for key in f:
            obj = f[key]
            if hasattr(obj, "shape") and len(obj.shape) == 2:
                bold = np.asarray(obj, dtype=np.float64)
                if bold.shape[0] < bold.shape[1]:
                    return bold, tr_seconds
                return bold.T, tr_seconds
    raise RuntimeError(f"Could not find BOLD data in {p}; inspect HDF5 keys.")


def extract_per_sentence_evoked(
    bold: np.ndarray,
    sentences: List[Sentence],
    tr_seconds: float = 2.0,
    hrf_lag_sec: float = 5.0,
    window_post_offset_sec: float = 4.0,
) -> np.ndarray:
    """
    Extract per-sentence evoked activation patterns from a continuous-BOLD
    time-series + sentence-timing annotations.

    For each sentence with onset t_on and offset t_off, average BOLD over
    the time window:

        [t_on + hrf_lag, t_off + hrf_lag + window_post_offset]

    This approximates the hemodynamic-response-shifted signal-of-interest
    window. With TR=2s and HRF peak ~5s, this captures the BOLD response
    elicited by the sentence.

    Returns:
        evoked: (n_sentences, n_voxels) matrix of per-sentence mean BOLD.
    """
    n_trs, n_voxels = bold.shape
    n_sentences = len(sentences)
    evoked = np.zeros((n_sentences, n_voxels), dtype=np.float64)

    for i, s in enumerate(sentences):
        win_start_sec = s.start_sec + hrf_lag_sec
        win_end_sec = s.end_sec + hrf_lag_sec + window_post_offset_sec
        # Convert to TR indices
        tr_start = int(np.floor(win_start_sec / tr_seconds))
        tr_end = int(np.ceil(win_end_sec / tr_seconds))
        tr_start = max(0, tr_start)
        tr_end = min(n_trs, tr_end)
        if tr_end <= tr_start:
            # Window completely outside recorded BOLD — fill with zeros
            continue
        window = bold[tr_start:tr_end]
        evoked[i] = window.mean(axis=0)
    return evoked
