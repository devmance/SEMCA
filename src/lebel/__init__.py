"""
LeBel et al. 2023 fMRI dataset loader for SEMCA 7 cross-substrate analysis.

Reference:
  LeBel, A., Wagner, L., Jain, S., et al. (2023). A natural language
  fMRI dataset for voxelwise encoding models. Scientific Data 10, 555.
  DOI: 10.1038/s41597-023-02437-z

OpenNeuro: https://openneuro.org/datasets/ds003020 (license: CC0)

Tang et al. 2023 ("Semantic reconstruction of continuous language from
non-invasive brain recordings", Nat Neurosci) uses the same dataset.

9 subjects (UTS01-UTS09) listening to ~84 narrative stories each (~5-15
hours of fMRI per subject). Each story is one HDF5 file of preprocessed
BOLD signal. Word-level alignment in companion TextGrid files lets us
extract per-sentence evoked patterns.

This is our chosen fallback when Pereira et al. 2018 BOLD data is
unavailable. The advantages over Pereira:
  - 9 subjects vs 16 (slightly fewer)
  - CC0 license, fully public, no application required
  - Much larger sentence count (~5000+ sentences across all stories)
  - Naturalistic stimuli (stories) - ecologically valid
  - Pre-processed data + alignments ready to use

The methodology:
  1. Parse TextGrid → list of (sentence_text, start_sec, end_sec)
  2. For each subject + story: load HDF5 BOLD, extract per-sentence
     evoked pattern by averaging BOLD during sentence time window
     (with HRF lag adjustment)
  3. Build FMRISubstrate (n_sentences, n_voxels_or_ROIs) per subject
  4. Run substrate-agnostic theories
"""

from .textgrid_loader import parse_textgrid, Sentence
from .bold_loader import (
    load_subject_story_bold,
    extract_per_sentence_evoked,
    LebelSubstrate,
)
from .substrate_adapter import build_lebel_substrate

__all__ = [
    "parse_textgrid",
    "Sentence",
    "load_subject_story_bold",
    "extract_per_sentence_evoked",
    "LebelSubstrate",
    "build_lebel_substrate",
]
