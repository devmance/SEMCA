"""
Per-story cross-substrate correlation analysis.

The companion to `population_cross_substrate.py`. Where the population
analysis compares pooled distributions, this analyzes paired data:
each LeBel story has been processed by both an AI substrate (1500-token
encoding) and a human substrate (per-subject BOLD evoked pattern). For
each (AI model, theory) we compute the Pearson and Spearman correlation
between AI per-story scores and human per-story scores across the
matched set of stories.

Headline statistic: per-theory Pearson r averaged across AI models and
human subjects. r > 0 indicates the theory's substrate-agnostic
operationalization assigns concordant scores across substrates given
the same stimulus.

  Pattern              Interpretation
  -----------------    ---------------------------------------------
  r > 0.4 (multi-model, multi-subj)
                       Strong cross-substrate signature — theory
                       captures shared stimulus-level structure that
                       transfers across AI and human substrates.
  0.1 < r < 0.4        Modest substrate-shared variance; theory
                       partially substrate-portable.
  r ≈ 0                Substrate-agnostic operationalization scores
                       stimuli on substrate-specific criteria;
                       cross-substrate magnitudes only.
  r < 0                Inverse substrate ordering — theories' ranking
                       of stimuli diverges between AI and human.

Outputs publication-ready per-theory correlation tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr


THEORIES = ["IIT", "GWT", "AST", "HOT", "PPT", "QIT", "FEP"]


def load_lebel_ai(data_dir: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns {model: {story: {theory: score, ..., unified: ...}}}
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for f in sorted(Path(data_dir).glob("lebel_ai_*.json")):
        model = f.stem.replace("lebel_ai_", "").replace("__", "/")
        d = json.loads(f.read_text())
        per_story: Dict[str, Dict[str, float]] = {}
        for r in d.get("results", []):
            if "error" in r:
                continue
            row = {"unified": r["unified"]}
            for theory in THEORIES:
                row[theory] = r["theories"][theory]["score"]
            per_story[r["story"]] = row
        out[model] = per_story
    return out


def load_lebel_human(
    path: str, min_sentences: int = 8,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns {subject: {story: {theory: score, ..., unified: ...}}}
    Filters out degenerate (too-short, NaN, zero) substrates.
    """
    import math
    d = json.loads(Path(path).read_text())
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
        for theory in THEORIES:
            s = r["theories"][theory]["score"]
            if isinstance(s, float) and math.isnan(s):
                s = 0.0
            row[theory] = s
        out.setdefault(r["subject_id"], {})[r["story"]] = row
    return out


def matched_stories(ai_data: Dict, hum_data: Dict) -> List[str]:
    """
    Stories present in all AI models AND all subjects.
    """
    if not ai_data or not hum_data:
        return []
    sets = [set(per_story.keys()) for per_story in ai_data.values()]
    sets += [set(per_story.keys()) for per_story in hum_data.values()]
    return sorted(set.intersection(*sets))


def compute_correlations(
    ai_data: Dict, hum_data: Dict, stories: List[str],
) -> Dict:
    """
    For each (model, subject, theory) triple, compute Pearson + Spearman
    correlation across matched stories.

    Returns nested dict:
      {
        "per_pair": [{model, subject, theory, pearson_r, pearson_p,
                      spearman_r, spearman_p, n}, ...],
        "per_theory_summary": {theory: {mean_r, std_r, n_pairs,
                                         frac_positive, frac_significant}},
        "stories": list of matched stories,
        "n_models": ..., "n_subjects": ..., "n_stories": ...,
      }
    """
    per_pair: List[Dict] = []
    for model, ai_per_story in ai_data.items():
        for subject, hum_per_story in hum_data.items():
            for theory in THEORIES + ["unified"]:
                ai_vec = np.array([ai_per_story[s][theory] for s in stories])
                hum_vec = np.array([hum_per_story[s][theory] for s in stories])
                if len(ai_vec) < 3:
                    continue
                try:
                    pr = pearsonr(ai_vec, hum_vec)
                    sr = spearmanr(ai_vec, hum_vec)
                    per_pair.append({
                        "model": model,
                        "subject": subject,
                        "theory": theory,
                        "n": len(ai_vec),
                        "pearson_r": float(pr.statistic),
                        "pearson_p": float(pr.pvalue),
                        "spearman_r": float(sr.statistic),
                        "spearman_p": float(sr.pvalue),
                        "ai_mean": float(ai_vec.mean()),
                        "hum_mean": float(hum_vec.mean()),
                    })
                except Exception as e:
                    per_pair.append({
                        "model": model, "subject": subject,
                        "theory": theory, "error": str(e),
                    })

    # Per-theory aggregation
    per_theory_summary: Dict[str, Dict] = {}
    for theory in THEORIES + ["unified"]:
        rs = [r for r in per_pair if r.get("theory") == theory and "pearson_r" in r]
        if not rs:
            continue
        pearsons = np.array([r["pearson_r"] for r in rs])
        spears = np.array([r["spearman_r"] for r in rs])
        ps = np.array([r["pearson_p"] for r in rs])
        per_theory_summary[theory] = {
            "n_pairs": int(len(rs)),
            "pearson_r_mean": float(pearsons.mean()),
            "pearson_r_std": float(pearsons.std()),
            "pearson_r_min": float(pearsons.min()),
            "pearson_r_max": float(pearsons.max()),
            "spearman_r_mean": float(spears.mean()),
            "spearman_r_std": float(spears.std()),
            "frac_positive": float((pearsons > 0).mean()),
            "frac_significant_p05": float((ps < 0.05).mean()),
        }
    return {
        "per_pair": per_pair,
        "per_theory_summary": per_theory_summary,
        "stories": stories,
        "n_stories_matched": len(stories),
    }


def run(data_dir: str = "data/", lebel_human_path: str = None) -> Dict:
    if lebel_human_path is None:
        lebel_human_path = str(Path(data_dir) / "fmri_substrate_lebel_per_story.json")

    ai_data = load_lebel_ai(data_dir)
    hum_data = load_lebel_human(lebel_human_path)

    print(f"AI LeBel substrates loaded ({len(ai_data)} models):")
    for model, sd in ai_data.items():
        print(f"  {model}: {len(sd)} stories")
    print(f"\nHuman LeBel substrates loaded ({len(hum_data)} subjects):")
    for subj, sd in hum_data.items():
        print(f"  {subj}: {len(sd)} stories")

    stories = matched_stories(ai_data, hum_data)
    print(f"\nMatched stories (in all models and subjects): {len(stories)}")
    if len(stories) < 5:
        print("WARNING: too few matched stories for stable correlation")

    result = compute_correlations(ai_data, hum_data, stories)

    summary = result["per_theory_summary"]
    print(f"\n{'='*88}")
    print(f"PER-STORY CROSS-SUBSTRATE CORRELATION  (n={len(stories)} stories,"
          f" {len(ai_data)} models × {len(hum_data)} subjects = "
          f"{len(ai_data)*len(hum_data)} model-subject pairs per theory)")
    print(f"{'='*88}")
    headers = ["Theory", "mean r", "std r", "[min,max]", "Spearman ρ",
               "frac r>0", "frac p<.05"]
    rows = []
    for theory in THEORIES + ["unified"]:
        if theory not in summary:
            continue
        s = summary[theory]
        rows.append([
            theory,
            f"{s['pearson_r_mean']:+.3f}",
            f"{s['pearson_r_std']:.3f}",
            f"[{s['pearson_r_min']:+.2f}, {s['pearson_r_max']:+.2f}]",
            f"{s['spearman_r_mean']:+.3f}",
            f"{s['frac_positive']:.2f}",
            f"{s['frac_significant_p05']:.2f}",
        ])
    widths = [max(len(headers[i]), max(len(r[i]) for r in rows))
              for i in range(len(headers))]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))

    print("\nInterpretation:")
    pos_theories = [t for t in THEORIES if t in summary
                    and summary[t]["pearson_r_mean"] > 0.2]
    neg_theories = [t for t in THEORIES if t in summary
                    and summary[t]["pearson_r_mean"] < -0.1]
    null_theories = [t for t in THEORIES if t in summary
                     and -0.1 <= summary[t]["pearson_r_mean"] <= 0.2]
    print(f"  Theories with strong substrate-shared ranking (r > 0.2): "
          f"{pos_theories or 'none'}")
    print(f"  Theories with substrate-agnostic noise (|r| < 0.1): "
          f"{null_theories or 'none'}")
    print(f"  Theories with inverse substrate ranking (r < -0.1): "
          f"{neg_theories or 'none'}")

    return result


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    result = run(data_dir=str(ROOT / "data"))
    out_path = ROOT / "data" / "cross_substrate_perstory_analysis.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved {out_path}")
