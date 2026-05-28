"""
Sensitivity analysis for the Fisher-Rao geometric integration's theoretical-
compatibility prior matrix. Tests whether the cross-substrate result is
robust to the choice of compatibility matrix.

Audit concern: the 7×7 theoretical-compatibility matrix is hand-encoded
based on a reading of the consciousness-theory literature. If the matrix
was tuned (consciously or not) while iterating on the implementation,
the cross-substrate r and consensus values could be artifact of that
tuning. To defuse this concern, we re-run the geometric integration
under three alternative priors:

  (i)   Uniform 0.5 matrix (all theory pairs equally compatible)
  (ii)  Identity matrix (theories have no inter-theory influence)
  (iii) Random perturbation of the original matrix (n=20 random priors,
        each cell ~Uniform[0.3, 0.95])

For each prior, we recompute the per-cell geometric integration and the
cross-substrate per-stimulus correlations. If the cross-substrate r is
stable across priors, the original finding is not cherry-picked. If r
varies substantially, the matrix is doing more work than acknowledged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import pearsonr

from .geometric_integration import (
    THEORIES, THEORY_COMPAT, geometric_integration,
)
import src.geometric_integration as gi_module  # noqa: PLW0406  (for mutation)


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_ai() -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for f in sorted(DATA.glob("lebel_ai_*.json")):
        model = f.stem.replace("lebel_ai_", "").replace("__", "/")
        d = json.loads(f.read_text())
        per_story: Dict[str, Dict[str, float]] = {}
        for r in d.get("results", []):
            if "error" in r:
                continue
            row = {t: r["theories"][t]["score"] for t in THEORIES}
            row["unified"] = r["unified"]
            per_story[r["story"]] = row
        out[model] = per_story
    return out


def load_human(min_sentences: int = 8) -> Dict[str, Dict[str, Dict[str, float]]]:
    d = json.loads((DATA / "fmri_substrate_lebel_per_story.json").read_text())
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in d.get("results", []):
        if "error" in r:
            continue
        if r.get("n_sentences", 0) < min_sentences:
            continue
        u = r.get("unified", 0.0)
        if isinstance(u, float) and (math.isnan(u) or u == 0.0):
            continue
        row = {t: r["theories"][t]["score"] for t in THEORIES}
        row["unified"] = u
        out.setdefault(r["subject_id"], {})[r["story"]] = row
    return out


def matched_stories(ai, hum) -> List[str]:
    s = [set(d.keys()) for d in ai.values()] + [set(d.keys()) for d in hum.values()]
    return sorted(set.intersection(*s))


def integrate_with_matrix(matrix: np.ndarray, scores: Dict[str, float]) -> float:
    """Patch the module-level THEORY_COMPAT for this call, then restore."""
    original = gi_module.THEORY_COMPAT
    gi_module.THEORY_COMPAT = matrix
    try:
        result = geometric_integration({t: scores[t] for t in THEORIES},
                                       return_weights=False)
        return result.unified_score
    finally:
        gi_module.THEORY_COMPAT = original


def riemannian_cross_substrate_r(
    ai: Dict, hum: Dict, stories: List[str], matrix: np.ndarray,
) -> dict:
    """Riemannian unified cross-substrate Pearson r under the given prior."""
    ai_vec, hum_vec = [], []
    for s in stories:
        # AI side: average across all models
        ai_cells = [integrate_with_matrix(matrix, ai[m][s])
                    for m in ai if s in ai[m]]
        # Human side: average across all subjects
        hum_cells = [integrate_with_matrix(matrix, hum[subj][s])
                     for subj in hum if s in hum[subj]]
        if not ai_cells or not hum_cells:
            continue
        ai_vec.append(float(np.mean(ai_cells)))
        hum_vec.append(float(np.mean(hum_cells)))

    ai_vec, hum_vec = np.array(ai_vec), np.array(hum_vec)
    if len(ai_vec) < 4 or ai_vec.std() < 1e-9 or hum_vec.std() < 1e-9:
        return {"r": float("nan"), "p": float("nan"),
                "ai_mean": float(ai_vec.mean()) if len(ai_vec) else float("nan"),
                "hum_mean": float(hum_vec.mean()) if len(hum_vec) else float("nan"),
                "ai_sd": float(ai_vec.std()) if len(ai_vec) else float("nan"),
                "hum_sd": float(hum_vec.std()) if len(hum_vec) else float("nan"),
                "n": len(ai_vec)}
    pr = pearsonr(ai_vec, hum_vec)
    return {
        "r": float(pr.statistic), "p": float(pr.pvalue),
        "ai_mean": float(ai_vec.mean()), "hum_mean": float(hum_vec.mean()),
        "ai_sd": float(ai_vec.std()), "hum_sd": float(hum_vec.std()),
        "n": int(len(ai_vec)),
    }


def main():
    ai = load_ai()
    hum = load_human()
    stories = matched_stories(ai, hum)
    print(f"Loaded: {len(ai)} AI models, {len(hum)} subjects, {len(stories)} matched stories\n")

    results = {}

    # Original literature-based matrix
    print("Computing under ORIGINAL literature-based matrix...")
    results["original_literature"] = riemannian_cross_substrate_r(
        ai, hum, stories, THEORY_COMPAT,
    )

    # Uniform 0.5
    print("Computing under UNIFORM 0.5 matrix...")
    uniform = np.full((7, 7), 0.5)
    np.fill_diagonal(uniform, 1.0)
    results["uniform_0.5"] = riemannian_cross_substrate_r(ai, hum, stories, uniform)

    # Identity (no off-diagonal influence)
    print("Computing under IDENTITY matrix (no inter-theory influence)...")
    identity = np.eye(7)
    results["identity"] = riemannian_cross_substrate_r(ai, hum, stories, identity)

    # Random perturbations (n=20)
    print("Computing under 20 RANDOM perturbations of original matrix...")
    rng = np.random.default_rng(42)
    random_rs = []
    for i in range(20):
        # Random matrix with each off-diagonal cell ~ Uniform[0.3, 0.95], symmetric
        rand = np.eye(7)
        upper = rng.uniform(0.3, 0.95, size=(7, 7))
        upper = np.triu(upper, k=1)
        rand = rand + upper + upper.T
        rand = np.clip(rand, 0, 1)
        np.fill_diagonal(rand, 1.0)
        cs = riemannian_cross_substrate_r(ai, hum, stories, rand)
        random_rs.append(cs["r"])
    random_rs = np.array(random_rs)
    results["random_n20"] = {
        "mean_r": float(np.mean(random_rs)),
        "std_r": float(np.std(random_rs)),
        "min_r": float(np.min(random_rs)),
        "max_r": float(np.max(random_rs)),
        "individual_rs": random_rs.tolist(),
    }

    # Print summary
    print(f"\n{'='*80}")
    print("SENSITIVITY ANALYSIS: Riemannian unified cross-substrate Pearson r")
    print('='*80)
    print(f"{'Prior':<28}  {'r':>10}  {'p':>8}  {'AI gap':>10}")
    print("-" * 60)
    for key in ["original_literature", "uniform_0.5", "identity"]:
        r = results[key]
        gap = r["ai_mean"] - r["hum_mean"]
        print(f"  {key:<26}  {r['r']:+.4f}    {r['p']:.4f}    {gap:+.3f}")
    rr = results["random_n20"]
    print(f"  {'random_n20 (mean)':<26}  {rr['mean_r']:+.4f}    {'-':>8}    {'-':>10}")
    print(f"  {'random_n20 (range)':<26}  [{rr['min_r']:+.3f}, {rr['max_r']:+.3f}]")
    print(f"  {'random_n20 (sd)':<26}  {rr['std_r']:.4f}")

    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print('='*80)
    orig_r = results["original_literature"]["r"]
    uniform_r = results["uniform_0.5"]["r"]
    identity_r = results["identity"]["r"]
    if abs(orig_r - uniform_r) < 0.1 and abs(orig_r - identity_r) < 0.1:
        print("  Cross-substrate r is stable across prior choices (Δr < 0.1)")
        print("  → Theoretical-compatibility matrix is NOT doing the work; the null is real")
    else:
        print(f"  Cross-substrate r varies substantially across priors")
        print(f"  (original {orig_r:+.3f}, uniform {uniform_r:+.3f}, identity {identity_r:+.3f})")
        print(f"  → The prior IS affecting the conclusion; report sensitivity range")

    out_path = DATA / "geometric_sensitivity_analysis.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
