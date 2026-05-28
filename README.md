# SEMCA-7 — Substrate-Agnostic Cross-Substrate Consciousness Theory Comparison

Reproducible code and data for the paper:

> **Architectural Variance Dominates Stimulus Variance in Six of Seven Substrate-Agnostic Consciousness Operationalizations**
>
> *A cross-substrate comparison with human fMRI BOLD reveals coincidental magnitude alignment of fundamentally different variance sources*
>
> Nate Travis · Devmance Labs

LaTeX source: [`paper/semca7.tex`](paper/semca7.tex) (bibliography: [`paper/refs.bib`](paper/refs.bib)).

## What's in this repository

A complete substrate-agnostic operationalization of seven mathematical theories of consciousness — IIT, GWT, AST, HOT, PPT, QIT, FEP — applied identically to:

- **AI substrates**: attention activations from four open-weight transformer language models reading narrative stories
- **Human substrates**: fMRI BOLD signals from three subjects listening to the same stories (LeBel et al. 2023, ds003020)

The seven theories' operationalizations are implemented as functions of a single abstract `Substrate` type with a `(T × N)` activity matrix. The same Python code runs on transformer attention activations and on K-means-parcellated fMRI BOLD signals.

## Findings

1. **Population magnitudes overlap** across AI and human substrates for four of seven (plus unified) — unified-score gap −1.7 on a 0–100 scale; three theories (IIT, GWT, FEP) under 5-point gaps. Four theories (AST, HOT, PPT, QIT) show substantial cross-substrate divergence.
2. **Per-stimulus rankings show no cross-substrate correlation for six of seven theories** (Pearson r ∈ [−0.17, +0.18] after noise averaging). Global Workspace Theory is the exception (r = +0.365, p = 0.001), though its cross-substrate signal is heterogeneous across AI architectures.
3. **AI-side per-story variance is dominated by architecture**, not stimulus content. The substrate-agnostic mathematics produces stimulus-insensitive outputs on transformer activations for six of seven theories. The apparent population magnitude alignment is consistent with coincidental overlap of architecturally-driven AI variance and stimulus-driven human variance.

Fisher-Rao information-geometric integration (Riemannian-mean unified score) preserves the null cross-substrate correlation (r = −0.185, p = 0.11), stable across alternative compatibility-prior matrices (range [−0.21, −0.16] over uniform, identity, and 20 random perturbations).

The substrate-independence claim of contemporary consciousness theory, as standardly operationalized, is not testable on transformer substrates with naturalistic narrative stimuli using these operationalizations: no AI-side stimulus-driven measurement of sufficient signal-to-noise is available to compare against the human-side measurement.

## Repository structure

```
paper/
  semca7.tex             — paper source (LaTeX + natbib)
  refs.bib               — bibliography
data/                    — pre-computed substrate scores + analysis outputs
  lebel_ai_*.json        — AI substrate scores (4 models × 76 LeBel stories)
  fmri_substrate_lebel_per_story.json — human substrate scores (3 subjects × 76 stories)
  cross_substrate_*.json — analysis outputs (§4.1, §4.2, §4.4)
  robustness_perstory_analysis.json — §4.2/§4.3 noise-averaged + variance decomp
  geometric_sensitivity_analysis.json — §4.4 compatibility-prior sensitivity
figures/                 — 6 publication figures (PNG + PDF)
src/
  substrate/             — Substrate abstraction (TransformerSubstrate, FMRISubstrate)
  theories_v2/           — 7 substrate-agnostic theory calculators
  lebel/                 — LeBel ds003020 BOLD + TextGrid loaders, K-means parcellation
  model_loader.py        — transformer activation extractor
  geometric_integration.py — Fisher-Rao information-geometric integration
  population_cross_substrate.py — §4.1 analysis
  perstory_cross_substrate.py   — §4.2 per-pair analysis
  robustness_perstory_analysis.py — §4.2/§4.3 noise-averaged + variance decomp
  cross_substrate_geometric.py    — §4.4 Riemannian unified
  geometric_sensitivity.py        — §4.4 prior-matrix sensitivity
examples/
  run_lebel_ai_substrate.py   — AI substrate on LeBel stories
  run_lebel_fmri_substrate.py — human substrate on LeBel BOLD
```

## Reproducing the analyses

The repository ships pre-computed substrate scores so the cross-substrate analyses can be reproduced without re-running the expensive substrate computations.

### Re-running the analyses on pre-computed data (fast, no GPU)

```bash
pip install -r requirements.txt

# Three cross-substrate analyses
python -m src.population_cross_substrate
python -m src.perstory_cross_substrate
python -m src.robustness_perstory_analysis

# Geometric integration + sensitivity
python -m src.cross_substrate_geometric
python -m src.geometric_sensitivity
```

Each analysis prints a results table matching §4 of the paper and writes a JSON output to `data/`. Total runtime ~1 minute on a laptop.

### Re-running substrate computation from raw data (slow, GPU recommended)

**Human substrate (CPU, ~30 min for 3 subjects × 76 stories):**

```bash
# 1. Download LeBel ds003020 from OpenNeuro (https://openneuro.org/datasets/ds003020/) under CC0
#    You need: derivatives/preprocessed_data/UTS01,UTS02,UTS03/*.hf5 and derivatives/TextGrids/*.TextGrid
# 2. Set the dataset root and run:
LEBEL_DATASET_ROOT=/path/to/ds003020 python -m examples.run_lebel_fmri_substrate
```

**AI substrate (GPU, ~30 min per model on 2 × H100):**

```bash
# 4 models, ~30 min each. Requires HuggingFace login for gated models (Llama).
export HF_TOKEN=your_token
MODEL_NAME='mistralai/Mistral-7B-Instruct-v0.3' python -m examples.run_lebel_ai_substrate
MODEL_NAME='mistralai/Mistral-Nemo-Instruct-2407' python -m examples.run_lebel_ai_substrate
MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'    python -m examples.run_lebel_ai_substrate
MODEL_NAME='microsoft/Phi-3-mini-4k-instruct'    python -m examples.run_lebel_ai_substrate
```

## The substrate abstraction

The mathematical core is in `src/substrate/base.py`:

```python
class Substrate(ABC):
    @property
    @abstractmethod
    def activity(self) -> np.ndarray:    # (T, N) activity matrix
        ...

    @property
    @abstractmethod
    def substrate_type(self) -> str:     # only used to pick prediction adapter
        ...

    @property
    def node_groups(self) -> Optional[List[List[int]]]:
        return None
```

Any dynamical system observable as a `(T × N)` real-valued activity matrix qualifies — molecular dynamics, neural recordings, simulated agents, etc. The seven theory calculators in `src/theories_v2/` accept any `Substrate` and produce a score in `[0, 100]`.

`TransformerSubstrate` and `FMRISubstrate` are concrete implementations; adding a new substrate type (e.g., a spiking-network or recurrent-net substrate) requires only implementing the abstract methods.

## Citation

If you use this work, please cite:

```bibtex
@misc{travis2026semca7,
  title={Architectural Variance Dominates Stimulus Variance in
         Six of Seven Substrate-Agnostic Consciousness Operationalizations},
  author={Travis, Nate},
  year={2026},
  howpublished={Preprint, Devmance Labs},
  url={https://github.com/devmance/SEMCA}
}
```

## License

MIT License — see [`LICENSE`](LICENSE).

The LeBel et al. 2023 dataset (ds003020) is released under CC0 by the original authors and is independently available at [OpenNeuro](https://openneuro.org/datasets/ds003020/). The data files in `data/fmri_substrate_lebel_per_story.json` are pre-computed scores derived from that public dataset.

## Contact

Nate Travis — labs@devmance.com — Devmance Labs
