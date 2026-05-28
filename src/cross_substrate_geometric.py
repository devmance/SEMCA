"""
Apply Fisher-Rao geometric integration to all (substrate, stimulus) cells.

For each cell — (AI model, story) on the AI side and (subject, story) on the
human side — compute the four geometric integration outputs (Riemannian
unified, consensus, coherence, curvature). Aggregate across substrates and
analyze:

  1. Per-substrate distribution of each geometric output (does AI differ from
     human in consensus / coherence / curvature?)
  2. Cross-substrate per-stimulus correlation of each output
  3. Comparison of Riemannian-mean unified to naive-mean unified
     (robustness of the cross-substrate result to integration method)

Outputs JSON for figure generation and a printed summary table.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import pearsonr, spearmanr, ks_2samp

from .geometric_integration import (
    THEORIES, geometric_integration, naive_unified_score,
)

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
        row = {}
        skip = False
        for t in THEORIES:
            s = r["theories"][t]["score"]
            if isinstance(s, float) and math.isnan(s):
                s = 0.0
            row[t] = s
        row["unified"] = u
        out.setdefault(r["subject_id"], {})[r["story"]] = row
    return out


def integrate_cell(scores: Dict[str, float]) -> Dict[str, float]:
    """Compute geometric integration outputs for one cell."""
    seven = {t: float(scores[t]) for t in THEORIES}
    result = geometric_integration(seven, return_weights=False)
    return {
        "naive_unified": naive_unified_score(seven),
        "riemannian_unified": result.unified_score,
        "consensus": result.theoretical_consensus,
        "coherence": result.geometric_coherence,
        "curvature": result.manifold_curvature,
        "confidence": result.integration_confidence,
    }


def integrate_substrate_batch(
    substrate_data: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Apply integrate_cell to every (substrate_member, story)."""
    out = {}
    for member, per_story in substrate_data.items():
        out[member] = {}
        for story, scores in per_story.items():
            out[member][story] = integrate_cell(scores)
    return out


def collect_per_substrate(
    integrated: Dict[str, Dict[str, Dict[str, float]]],
    field: str,
) -> np.ndarray:
    """Flatten all (member, story) values of one geometric field into one array."""
    return np.array([
        cell[field]
        for per_story in integrated.values()
        for cell in per_story.values()
    ])


def cross_substrate_correlation(
    ai_integrated: Dict, hum_integrated: Dict, field: str, stories: List[str],
) -> Dict[str, float]:
    """Model-and-subject averaged Pearson r on field across matched stories."""
    ai_vec = []
    hum_vec = []
    for s in stories:
        ai_vals = [ai_integrated[m][s][field] for m in ai_integrated if s in ai_integrated[m]]
        hum_vals = [hum_integrated[subj][s][field] for subj in hum_integrated if s in hum_integrated[subj]]
        if not ai_vals or not hum_vals:
            continue
        ai_vec.append(np.mean(ai_vals))
        hum_vec.append(np.mean(hum_vals))
    ai_vec, hum_vec = np.array(ai_vec), np.array(hum_vec)
    if len(ai_vec) < 4 or ai_vec.std() < 1e-9 or hum_vec.std() < 1e-9:
        return {"r": float("nan"), "p": float("nan"), "rho": float("nan"), "n": len(ai_vec)}
    pr = pearsonr(ai_vec, hum_vec)
    sr = spearmanr(ai_vec, hum_vec)
    return {
        "r": float(pr.statistic), "p": float(pr.pvalue),
        "rho": float(sr.statistic), "n": int(len(ai_vec)),
    }


def main():
    ai = load_ai()
    hum = load_human()

    print("Applying Fisher-Rao geometric integration to all cells...")
    ai_integrated = integrate_substrate_batch(ai)
    hum_integrated = integrate_substrate_batch(hum)

    # Matched-corpus design: per-substrate distributions are pooled across
    # the same set of stories on both sides (otherwise AI/human means are
    # not directly comparable). The intersection of stories present in every
    # AI model and every human subject is computed here and used for both
    # the per-substrate distribution and the cross-substrate correlation.
    sets = [set(per_story.keys()) for per_story in ai_integrated.values()]
    sets += [set(per_story.keys()) for per_story in hum_integrated.values()]
    stories = sorted(set.intersection(*sets))
    matched = set(stories)
    ai_integrated = {
        m: {s: cell for s, cell in per_story.items() if s in matched}
        for m, per_story in ai_integrated.items()
    }
    hum_integrated = {
        subj: {s: cell for s, cell in per_story.items() if s in matched}
        for subj, per_story in hum_integrated.items()
    }

    n_ai = sum(len(per_story) for per_story in ai_integrated.values())
    n_hum = sum(len(per_story) for per_story in hum_integrated.values())
    print(f"  Matched-corpus design: {len(stories)} stories present in all "
          f"{len(ai_integrated)} AI models and all {len(hum_integrated)} human subjects.")
    print(f"  AI: {len(ai_integrated)} models × {len(stories)} stories = {n_ai} cells")
    print(f"  Human: {len(hum_integrated)} subjects × {len(stories)} stories = {n_hum} cells")

    # --- Per-substrate distribution comparison ---
    print(f"\n{'='*100}")
    print("PER-SUBSTRATE DISTRIBUTION OF GEOMETRIC INTEGRATION OUTPUTS")
    print('='*100)
    headers = ["Field", "AI mean (sd)", "Human mean (sd)", "Gap", "KS p"]
    rows = []
    summary = {}
    fields = ["naive_unified", "riemannian_unified", "consensus", "coherence",
              "curvature", "confidence"]
    for field in fields:
        ai_arr = collect_per_substrate(ai_integrated, field)
        hum_arr = collect_per_substrate(hum_integrated, field)
        ai_mean, ai_std = float(ai_arr.mean()), float(ai_arr.std())
        h_mean, h_std = float(hum_arr.mean()), float(hum_arr.std())
        try:
            ks_stat, ks_p = ks_2samp(ai_arr, hum_arr)
        except Exception:
            ks_p = float("nan")
        rows.append([
            field,
            f"{ai_mean:.3f} ({ai_std:.3f})",
            f"{h_mean:.3f} ({h_std:.3f})",
            f"{ai_mean - h_mean:+.3f}",
            f"{ks_p:.4f}",
        ])
        summary[f"distribution_{field}"] = {
            "ai_mean": ai_mean, "ai_std": ai_std,
            "human_mean": h_mean, "human_std": h_std,
            "gap": ai_mean - h_mean, "ks_p": float(ks_p),
        }
    widths = [max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))

    # --- Cross-substrate per-stimulus correlation ---
    # (stories already computed above as the matched-corpus intersection)
    print(f"\n{'='*100}")
    print(f"CROSS-SUBSTRATE PER-STIMULUS CORRELATION ({len(stories)} matched stories,")
    print(f"  model-averaged AI × subject-averaged human)")
    print('='*100)
    headers2 = ["Field", "Pearson r", "Pearson p", "Spearman ρ", "n_stories"]
    rows2 = []
    for field in fields:
        cs = cross_substrate_correlation(ai_integrated, hum_integrated, field, stories)
        rows2.append([
            field,
            f"{cs['r']:+.4f}" if not np.isnan(cs["r"]) else "NaN",
            f"{cs['p']:.4f}" if not np.isnan(cs["p"]) else "NaN",
            f"{cs['rho']:+.4f}" if not np.isnan(cs["rho"]) else "NaN",
            f"{cs['n']}",
        ])
        summary[f"cross_substrate_{field}"] = cs
    widths2 = [max(len(headers2[i]), max(len(r[i]) for r in rows2)) for i in range(len(headers2))]
    print("  ".join(h.ljust(w) for h, w in zip(headers2, widths2)))
    print("  ".join("-" * w for w in widths2))
    for row in rows2:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths2)))

    # --- Compare naive vs Riemannian unified for cross-substrate finding ---
    naive_cs = cross_substrate_correlation(ai_integrated, hum_integrated, "naive_unified", stories)
    riem_cs = cross_substrate_correlation(ai_integrated, hum_integrated, "riemannian_unified", stories)
    print(f"\n{'='*100}")
    print("Robustness check: cross-substrate r under naive vs Riemannian integration")
    print('='*100)
    print(f"  Naive arithmetic mean unified:   r = {naive_cs['r']:+.4f}, p = {naive_cs['p']:.4f}")
    print(f"  Riemannian mean unified:         r = {riem_cs['r']:+.4f}, p = {riem_cs['p']:.4f}")
    if abs(naive_cs["r"]) < 0.1 and abs(riem_cs["r"]) < 0.1:
        print("  → Both formulations produce near-zero cross-substrate correlation.")
    else:
        print("  → Cross-substrate r differs between formulations; result sensitive to integration method.")

    # Save full integrated data + summary
    out_path = DATA / "cross_substrate_geometric_analysis.json"
    out_path.write_text(json.dumps({
        "ai_integrated": ai_integrated,
        "human_integrated": hum_integrated,
        "summary": summary,
        "n_matched_stories": len(stories),
    }, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
