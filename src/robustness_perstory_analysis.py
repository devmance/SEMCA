"""
Robustness analyses for the per-stimulus cross-substrate finding.

The original per-stimulus analysis computes 12 separate (model, subject)
Pearson correlations and reports their mean. A reviewer concern: if
within-model AI variance and within-subject human variance are largely
noise rather than stimulus-driven signal, the mean of 12 noisy
correlations could be near zero even if a substrate-shared signature
exists at the *population mean* level.

This script computes the cross-substrate correlation under three
alternative averaging strategies:

  (1) Model-averaged AI × subject-averaged human, per story.
      For each story, average AI scores across all 4 models → one AI
      score per (theory, story). Average human scores across 3
      subjects → one human score per (theory, story). Pearson r over
      76 stories.

  (2) Model-averaged AI × per-subject human.
      Averages AI noise across models but keeps per-subject human
      variance. One r per theory per subject; 3 r-values per theory.

  (3) Per-model AI × subject-averaged human.
      Inverse of (2): averages human noise across subjects but keeps
      per-model AI variance. One r per theory per model; 4 r-values
      per theory.

Additionally:
  (4) Permutation null distribution: shuffle story labels and
      recompute correlation; quantify how often pure noise produces
      the observed r magnitude.
  (5) Split-half stability: compute r on stories 1-38 and 39-76
      separately; if r is consistent across halves, the estimate is
      stable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

THEORIES = ["IIT", "GWT", "AST", "HOT", "PPT", "QIT", "FEP"]
ALL_KEYS = THEORIES + ["unified"]


def load_ai() -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for f in sorted(DATA.glob("lebel_ai_*.json")):
        model = f.stem.replace("lebel_ai_", "").replace("__", "/")
        d = json.loads(f.read_text())
        per_story: Dict[str, Dict[str, float]] = {}
        for r in d.get("results", []):
            if "error" in r:
                continue
            row = {"unified": r["unified"]}
            for t in THEORIES:
                row[t] = r["theories"][t]["score"]
            per_story[r["story"]] = row
        out[model] = per_story
    return out


def load_human(min_sentences: int = 8) -> Dict[str, Dict[str, Dict[str, float]]]:
    import math
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
        row = {"unified": u}
        for t in THEORIES:
            s = r["theories"][t]["score"]
            if isinstance(s, float) and math.isnan(s):
                s = 0.0
            row[t] = s
        out.setdefault(r["subject_id"], {})[r["story"]] = row
    return out


def matched_stories(ai: Dict, hum: Dict) -> List[str]:
    sets = [set(per_story.keys()) for per_story in ai.values()]
    sets += [set(per_story.keys()) for per_story in hum.values()]
    return sorted(set.intersection(*sets))


def per_story_means(ai: Dict, hum: Dict, theory: str, stories: List[str]):
    """For each story, compute mean across models (AI) and mean across subjects (Human)."""
    ai_mean = []
    hum_mean = []
    for s in stories:
        ai_vals = [ai[m][s][theory] for m in ai
                   if not (isinstance(ai[m][s][theory], float) and np.isnan(ai[m][s][theory]))]
        hum_vals = [hum[subj][s][theory] for subj in hum
                    if not (isinstance(hum[subj][s][theory], float) and np.isnan(hum[subj][s][theory]))]
        if not ai_vals or not hum_vals:
            ai_mean.append(np.nan); hum_mean.append(np.nan)
        else:
            ai_mean.append(np.mean(ai_vals))
            hum_mean.append(np.mean(hum_vals))
    return np.array(ai_mean), np.array(hum_mean)


def safe_corr(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 4 or x.std() < 1e-6 or y.std() < 1e-6:
        return np.nan, np.nan, np.nan, np.nan, len(x)
    try:
        pr = pearsonr(x, y)
        sr = spearmanr(x, y)
        return float(pr.statistic), float(pr.pvalue), float(sr.statistic), float(sr.pvalue), len(x)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, len(x)


def permutation_null(x, y, n_perm: int = 10000, seed: int = 42):
    """Return p-value via story-label shuffle."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    obs_r = float(np.corrcoef(x, y)[0, 1])
    null_rs = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        null_rs[i] = np.corrcoef(x, y_perm)[0, 1]
    p_two_sided = float(np.mean(np.abs(null_rs) >= abs(obs_r)))
    return obs_r, p_two_sided


def split_half_stability(x, y, seed: int = 42):
    """Compute r on two random halves; report both."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 8:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    half = len(idx) // 2
    h1 = idx[:half]; h2 = idx[half:]
    r1 = float(np.corrcoef(x[h1], y[h1])[0, 1])
    r2 = float(np.corrcoef(x[h2], y[h2])[0, 1])
    return r1, r2


def main():
    ai = load_ai()
    hum = load_human()
    stories = matched_stories(ai, hum)

    print(f"AI models: {len(ai)}  Subjects: {len(hum)}  Matched stories: {len(stories)}")
    print()

    # --- Analysis 1: model-averaged AI × subject-averaged human ---
    print("=" * 100)
    print("ANALYSIS (1): Model-averaged AI × subject-averaged human, single r per theory across stories")
    print("=" * 100)
    headers = ["Theory", "Pearson r", "Pearson p", "Spearman r", "perm p", "Split r1", "Split r2", "n_stories"]
    rows = []
    summary = {}
    for theory in ALL_KEYS:
        x, y = per_story_means(ai, hum, theory, stories)
        pr, ppr, sr, psr, n = safe_corr(x, y)
        _, p_perm = permutation_null(x, y, n_perm=5000)
        r1, r2 = split_half_stability(x, y)
        rows.append([
            theory,
            f"{pr:+.4f}" if not np.isnan(pr) else "NaN",
            f"{ppr:.4f}" if not np.isnan(ppr) else "NaN",
            f"{sr:+.4f}" if not np.isnan(sr) else "NaN",
            f"{p_perm:.4f}" if not np.isnan(p_perm) else "NaN",
            f"{r1:+.3f}" if not np.isnan(r1) else "NaN",
            f"{r2:+.3f}" if not np.isnan(r2) else "NaN",
            f"{n}",
        ])
        summary[theory] = {
            "pearson_r": pr, "pearson_p": ppr,
            "spearman_r": sr, "spearman_p": psr,
            "permutation_p": p_perm,
            "split_r1": r1, "split_r2": r2,
            "n_stories": n,
        }
    widths = [max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    print()

    # --- Analysis 2: model-averaged AI × per-subject human ---
    print("=" * 100)
    print("ANALYSIS (2): Model-averaged AI × per-subject human")
    print("=" * 100)
    rows2 = []
    headers2 = ["Theory", "UTS01 r", "UTS02 r", "UTS03 r", "Mean r"]
    for theory in ALL_KEYS:
        x = per_story_means(ai, hum, theory, stories)[0]  # model-avg AI
        per_subj = {}
        for subj in hum:
            y = np.array([
                hum[subj][s][theory] if s in hum[subj] else np.nan
                for s in stories
            ])
            r, _, _, _, _ = safe_corr(x, y)
            per_subj[subj] = r
        rs = [v for v in per_subj.values() if not np.isnan(v)]
        rows2.append([
            theory,
            f"{per_subj.get('UTS01', np.nan):+.4f}",
            f"{per_subj.get('UTS02', np.nan):+.4f}",
            f"{per_subj.get('UTS03', np.nan):+.4f}",
            f"{np.mean(rs):+.4f}" if rs else "NaN",
        ])
        summary[theory]["model_avg_per_subject"] = per_subj
    widths2 = [max(len(headers2[i]), max(len(r[i]) for r in rows2)) for i in range(len(headers2))]
    print("  ".join(h.ljust(w) for h, w in zip(headers2, widths2)))
    print("  ".join("-" * w for w in widths2))
    for row in rows2:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths2)))
    print()

    # --- Analysis 3: per-model × subject-averaged human ---
    print("=" * 100)
    print("ANALYSIS (3): Per-model AI × subject-averaged human")
    print("=" * 100)
    rows3 = []
    model_short = {m: m.split("/")[-1].split("-Instruct")[0].replace("Mistral-Nemo", "Nemo") for m in ai}
    headers3 = ["Theory"] + [model_short[m][:18] for m in ai] + ["Mean r"]
    for theory in ALL_KEYS:
        y = per_story_means(ai, hum, theory, stories)[1]
        per_model = {}
        for m in ai:
            x = np.array([
                ai[m][s][theory] if s in ai[m] else np.nan
                for s in stories
            ])
            r, _, _, _, _ = safe_corr(x, y)
            per_model[m] = r
        rs = [v for v in per_model.values() if not np.isnan(v)]
        rows3.append([theory] + [f"{per_model[m]:+.4f}" for m in ai] +
                     [f"{np.mean(rs):+.4f}" if rs else "NaN"])
        summary[theory]["per_model_subject_avg"] = per_model
    widths3 = [max(len(headers3[i]), max(len(r[i]) for r in rows3)) for i in range(len(headers3))]
    print("  ".join(h.ljust(w) for h, w in zip(headers3, widths3)))
    print("  ".join("-" * w for w in widths3))
    for row in rows3:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths3)))
    print()

    # --- Variance audit: how much of AI per-story variance is shared across models? ---
    print("=" * 100)
    print("AI-side variance decomposition: between-story vs within-story (across models)")
    print("=" * 100)
    headers4 = ["Theory", "Between-story sd", "Within-story sd (model-noise)", "ratio", "interpret"]
    rows4 = []
    for theory in ALL_KEYS:
        # For each story, get the 4 AI scores
        per_story_matrix = []
        for s in stories:
            scores = [ai[m][s][theory] for m in ai
                      if not (isinstance(ai[m][s][theory], float) and np.isnan(ai[m][s][theory]))]
            if scores:
                per_story_matrix.append(scores)
        per_story_matrix = np.array(per_story_matrix)  # (n_stories, n_models)
        if per_story_matrix.size == 0:
            rows4.append([theory, "NaN", "NaN", "NaN", "no data"])
            continue
        between_sd = per_story_matrix.mean(axis=1).std()  # SD of per-story means
        within_sd = per_story_matrix.std(axis=1).mean()    # mean within-story SD
        ratio = between_sd / (within_sd + 1e-9)
        interp = "stim-driven" if ratio > 1.0 else "model-noise dominant" if ratio < 0.5 else "mixed"
        rows4.append([theory, f"{between_sd:.4f}", f"{within_sd:.4f}",
                     f"{ratio:.3f}", interp])
        summary[theory]["between_story_sd"] = float(between_sd)
        summary[theory]["within_story_sd"] = float(within_sd)
        summary[theory]["variance_ratio"] = float(ratio)
    widths4 = [max(len(headers4[i]), max(len(r[i]) for r in rows4)) for i in range(len(headers4))]
    print("  ".join(h.ljust(w) for h, w in zip(headers4, widths4)))
    print("  ".join("-" * w for w in widths4))
    for row in rows4:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths4)))
    print()

    out_path = DATA / "robustness_perstory_analysis.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
