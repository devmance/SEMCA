"""
Population-level cross-substrate analysis (matched-corpus design).

Both substrates run on the LeBel et al. 2023 narrative-listening
stimulus set. AI substrates: 4 transformer architectures × 76 stories.
Human substrates: 3 fMRI subjects × 76 stories.

  Does the distribution of theory scores for AI substrates overlap with
  the distribution for human substrates, across the same substrate-
  agnostic mathematical operationalization?

This analyzer:
  1. Loads all LeBel AI substrate JSONs (per-story rows × per-model).
  2. Loads the LeBel fMRI substrate JSON (per-story rows × per-subject).
  3. For each of the 7 theories + unified, computes:
       AI distribution: pooled across all (model, story) cells
       Human distribution: pooled across all (subject, story) cells
       Mean / std / range for each
       Distribution overlap via:
         - Magnitude alignment (means within 1 std?)
         - KS test for distribution equality
         - Pearson correlation of per-model-mean × per-subject-mean
           ordering (architecture-vs-individual variance)
  4. Produces a publication-ready table.

Produces the population-level cross-substrate distribution comparison
reported in §4.1 of the paper. Per-stimulus correlation analyses on the
matched LeBel set are in `perstory_cross_substrate.py` and
`robustness_perstory_analysis.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu


THEORIES = ["IIT", "GWT", "AST", "HOT", "PPT", "QIT", "FEP"]


def load_ai_substrate_results(data_dir: str) -> Dict[str, Dict]:
    """
    Returns {model_name: {story_name: {theory: score, ..., unified: score}}}
    """
    out = {}
    for f in sorted(Path(data_dir).glob("lebel_ai_*.json")):
        model = f.stem.replace("lebel_ai_", "").replace("__", "/")
        d = json.loads(f.read_text())
        model_data = {}
        for r in d.get("results", []):
            if "error" in r:
                continue
            row = {"unified": r["unified"]}
            for theory in THEORIES:
                row[theory] = r["theories"][theory]["score"]
            model_data[r["story"]] = row
        out[model] = model_data
    return out


def load_lebel_substrate_results(
    path: str, min_sentences: int = 8,
) -> Dict[str, List[Dict]]:
    """
    Returns {subject_id: [{story, n_sentences, IIT, GWT, ..., unified}, ...]}

    Skips:
      - stories with errors during scoring
      - stories with fewer than min_sentences sentences (too short for
        stable theory scoring — produces NaN or near-zero scores)
      - stories whose unified score is NaN or exactly 0.0 (degenerate)
    """
    import math
    d = json.loads(Path(path).read_text())
    out: Dict[str, List[Dict]] = {}
    skipped = 0
    for r in d.get("results", []):
        if "error" in r:
            skipped += 1
            continue
        if r.get("n_sentences", 0) < min_sentences:
            skipped += 1
            continue
        u = r.get("unified", 0.0)
        if isinstance(u, float) and (math.isnan(u) or u == 0.0):
            skipped += 1
            continue
        row = {
            "story": r["story"],
            "n_sentences": r["n_sentences"],
            "unified": u,
        }
        for theory in THEORIES:
            s = r["theories"][theory]["score"]
            if isinstance(s, float) and math.isnan(s):
                s = 0.0
            row[theory] = s
        out.setdefault(r["subject_id"], []).append(row)
    if skipped:
        print(f"  (skipped {skipped} degenerate substrates)")
    return out


def pool_distribution(
    data: Dict, kind: str, theory: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Pool theory scores across all (model|subject, sentence|story) cells.

    Returns:
        (pooled_values, per_group_values) where per_group_values is
        {group_id: array} so we can also compute group-level means.
    """
    per_group = {}
    pooled = []
    if kind == "ai":
        # data is {model: {sentence_idx: {theory: ..., unified: ...}}}
        for model, sentences in data.items():
            vals = [s[theory] for s in sentences.values()]
            per_group[model] = np.array(vals)
            pooled.extend(vals)
    elif kind == "human":
        # data is {subject: [{theory: ..., unified: ...}, ...]}
        for subject, stories in data.items():
            vals = [s[theory] for s in stories]
            per_group[subject] = np.array(vals)
            pooled.extend(vals)
    else:
        raise ValueError(f"kind must be 'ai' or 'human'; got {kind}")
    return np.array(pooled), per_group


def matched_stories(ai_data: Dict, human_data: Dict) -> List[str]:
    """Stories present in every AI model AND every human subject (matched-corpus design)."""
    sets = [set(per_story.keys()) for per_story in ai_data.values()]
    sets += [set(row["story"] for row in rows) for rows in human_data.values()]
    return sorted(set.intersection(*sets)) if sets else []


def restrict_to_matched(
    ai_data: Dict, human_data: Dict, stories: List[str],
) -> Tuple[Dict, Dict]:
    """Filter both substrates to the matched-story set."""
    keep = set(stories)
    ai_out = {
        m: {s: row for s, row in per_story.items() if s in keep}
        for m, per_story in ai_data.items()
    }
    human_out = {
        subj: [row for row in rows if row["story"] in keep]
        for subj, rows in human_data.items()
    }
    return ai_out, human_out


def analyze_one_theory(
    ai_data: Dict, human_data: Dict, theory: str,
) -> Dict:
    """
    Compute the cross-substrate distribution comparison for one theory.
    """
    ai_pooled, ai_per_model = pool_distribution(ai_data, "ai", theory)
    human_pooled, human_per_subject = pool_distribution(human_data, "human", theory)

    if len(ai_pooled) < 3 or len(human_pooled) < 3:
        return {"theory": theory, "note": "insufficient_data"}

    # Magnitude alignment
    ai_mean, ai_std = ai_pooled.mean(), ai_pooled.std()
    h_mean, h_std = human_pooled.mean(), human_pooled.std()
    gap = ai_mean - h_mean
    pooled_std = (ai_std + h_std) / 2
    aligned = abs(gap) <= pooled_std

    # KS test for distribution equality (large-N statistic)
    try:
        ks_stat, ks_p = ks_2samp(ai_pooled, human_pooled)
    except Exception:
        ks_stat, ks_p = float("nan"), float("nan")

    # Mann-Whitney U for rank-based shift detection
    try:
        mw_stat, mw_p = mannwhitneyu(
            ai_pooled, human_pooled, alternative="two-sided",
        )
    except Exception:
        mw_stat, mw_p = float("nan"), float("nan")

    # Group-level means (architecture-level and subject-level)
    ai_group_means = {m: float(v.mean()) for m, v in ai_per_model.items()}
    h_group_means = {s: float(v.mean()) for s, v in human_per_subject.items()}
    # Group CV: how tightly do means cluster within each substrate type?
    ai_group_arr = np.array(list(ai_group_means.values()))
    h_group_arr = np.array(list(h_group_means.values()))
    ai_group_cv = float(ai_group_arr.std() / ai_group_arr.mean()) if ai_group_arr.mean() > 0 else 0.0
    h_group_cv = float(h_group_arr.std() / h_group_arr.mean()) if h_group_arr.mean() > 0 else 0.0

    return {
        "theory": theory,
        "n_ai": len(ai_pooled),
        "n_human": len(human_pooled),
        "ai_mean": float(ai_mean),
        "ai_std": float(ai_std),
        "human_mean": float(h_mean),
        "human_std": float(h_std),
        "magnitude_gap": float(gap),
        "magnitude_aligned": bool(aligned),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "mannwhitney_stat": float(mw_stat),
        "mannwhitney_p": float(mw_p),
        "ai_group_means": ai_group_means,
        "human_group_means": h_group_means,
        "ai_group_cv": ai_group_cv,
        "human_group_cv": h_group_cv,
    }


def run_population_analysis(
    ai_dir: str = "data/",
    lebel_path: str = "data/fmri_substrate_lebel_per_story.json",
) -> List[Dict]:
    ai_data = load_ai_substrate_results(ai_dir)
    human_data = load_lebel_substrate_results(lebel_path)
    print(f"AI substrates loaded ({len(ai_data)}):")
    for model, sentences in ai_data.items():
        print(f"  {model}: {len(sentences)} sentences")
    print(f"\nHuman substrates loaded:")
    for subj, stories in human_data.items():
        print(f"  {subj}: {len(stories)} stories")

    matched = matched_stories(ai_data, human_data)
    print(f"\nMatched-corpus design: restricting both substrates to "
          f"{len(matched)} stories present in all AI models and all human subjects.")
    ai_data, human_data = restrict_to_matched(ai_data, human_data, matched)

    print(f"\n{'='*100}")
    print("POPULATION-LEVEL CROSS-SUBSTRATE COMPARISON")
    print(f"{'='*100}\n")

    headers = ["Theory", "AI mean", "AI std", "Hum mean", "Hum std", "Gap", "Aligned",
               "KS p", "MW p", "AI cv", "Hum cv"]
    rows = []
    results = []
    for theory in THEORIES + ["unified"]:
        r = analyze_one_theory(ai_data, human_data, theory)
        if "note" in r:
            continue
        results.append(r)
        rows.append([
            r["theory"],
            f"{r['ai_mean']:.2f}", f"{r['ai_std']:.2f}",
            f"{r['human_mean']:.2f}", f"{r['human_std']:.2f}",
            f"{r['magnitude_gap']:+.2f}",
            "✓" if r["magnitude_aligned"] else "✗",
            f"{r['ks_p']:.4f}", f"{r['mannwhitney_p']:.4f}",
            f"{r['ai_group_cv']:.3f}", f"{r['human_group_cv']:.3f}",
        ])

    # Print formatted table
    widths = [max(len(str(headers[i])), max(len(r[i]) for r in rows)) for i in range(len(headers))]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))

    print()
    n_aligned = sum(1 for r in results if r["magnitude_aligned"])
    n_total = len(results)
    print(f"Magnitude alignment (gap ≤ pooled std): {n_aligned}/{n_total} theories")
    n_indistinguishable = sum(1 for r in results if r["ks_p"] > 0.05)
    print(f"Distribution indistinguishable (KS p > 0.05): {n_indistinguishable}/{n_total} theories")

    return results


if __name__ == "__main__":
    import sys
    ROOT = Path(__file__).resolve().parent.parent
    results = run_population_analysis(
        ai_dir=str(ROOT / "data"),
        lebel_path=str(ROOT / "data" / "fmri_substrate_lebel_per_story.json"),
    )
    output_path = ROOT / "data" / "cross_substrate_population_analysis.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved {output_path}")
