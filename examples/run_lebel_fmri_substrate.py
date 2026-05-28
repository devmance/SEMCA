"""
Run all 7 substrate-agnostic consciousness theories on the LeBel et al.
2023 fMRI dataset (ds003020).

For each (subject, story) pair: build the K-means-parcellated substrate
from the BOLD HDF5 + TextGrid, then score the 7 substrate-agnostic
theories. Save per-(subject, story) results.

Scores at the story level: one set of seven theory scores per
(subject, story). 84 stories × 3 subjects = 252 scores, of which 76 ×
3 = 228 pass the minimum-sentence-count filter and enter the matched
cross-substrate analysis (paper §3.1).

Usage:
    python -m examples.run_lebel_fmri_substrate
    # Subset of subjects:
    LEBEL_SUBJECTS=UTS01,UTS02 python -m examples.run_lebel_fmri_substrate
    # Subset of stories (debug):
    LEBEL_STORIES=adollshouse,wheretheressmoke python -m examples.run_lebel_fmri_substrate
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.lebel import build_lebel_substrate
from src.substrate.fmri import FMRIPredictionAdapter
from src.theories_v2 import (
    compute_iit_phi, compute_gwt, compute_hot, compute_ast,
    compute_ppt, compute_fep, compute_qit,
)


THEORIES = ["IIT", "GWT", "AST", "HOT", "PPT", "QIT", "FEP"]
KEY_MAP = dict(IIT="iit_value", GWT="gwt_value", HOT="hot_value", AST="ast_value",
               PPT="ppt_value", FEP="fep_value", QIT="qit_value")


def score_substrate(substrate):
    # PPT and FEP require a prediction adapter, which needs ≥3 timesteps
    # for AR(1) regression. Stories with very few sentences fall back to
    # theory-only scoring with PPT/FEP zeroed out.
    if substrate.n_timesteps < 4:
        # Score only the theories that don't need prediction
        out = {}
        for k, fn in [("IIT", compute_iit_phi), ("GWT", compute_gwt),
                      ("AST", compute_ast), ("HOT", compute_hot),
                      ("QIT", compute_qit)]:
            v = fn(substrate)
            out[k] = {
                "score": float(getattr(v, KEY_MAP[k])),
                "components": v.components,
                "confidence": float(v.confidence),
            }
        # PPT/FEP marked as degenerate
        for k in ("PPT", "FEP"):
            out[k] = {"score": 0.0, "components": {}, "confidence": 0.0,
                      "note": "substrate_too_short_for_prediction_adapter"}
        out["unified"] = float(np.mean([out[t]["score"] for t in THEORIES]))
        return out

    pred = FMRIPredictionAdapter(substrate.activity, lag=1)
    r = {
        "IIT": compute_iit_phi(substrate),
        "GWT": compute_gwt(substrate),
        "AST": compute_ast(substrate),
        "HOT": compute_hot(substrate),
        "PPT": compute_ppt(substrate, pred),
        "FEP": compute_fep(substrate, pred),
        "QIT": compute_qit(substrate),
    }
    out = {}
    for k, v in r.items():
        score = float(getattr(v, KEY_MAP[k]))
        out[k] = {
            "score": score,
            "components": v.components,
            "confidence": float(v.confidence),
        }
    out["unified"] = float(np.mean([out[t]["score"] for t in THEORIES]))
    return out


def list_stories(dataset_root: str, subject_id: str):
    p = Path(dataset_root) / "derivatives" / "preprocessed_data" / subject_id
    return sorted([f.stem for f in p.glob("*.hf5")])


def main():
    # Path to the LeBel ds003020 dataset root. Download from
    # https://openneuro.org/datasets/ds003020/ and point LEBEL_DATASET_ROOT
    # at the directory containing derivatives/preprocessed_data/ and
    # derivatives/TextGrids/.
    dataset_root = os.environ.get(
        "LEBEL_DATASET_ROOT",
        str(ROOT / "data" / "ds003020"),
    )
    subjects_env = os.environ.get("LEBEL_SUBJECTS", "UTS01,UTS02,UTS03")
    subjects = [s.strip() for s in subjects_env.split(",") if s.strip()]

    stories_env = os.environ.get("LEBEL_STORIES", "")
    requested_stories = [s.strip() for s in stories_env.split(",") if s.strip()]

    output_path = os.environ.get(
        "OUTPUT_PATH",
        str(ROOT / "data" / "fmri_substrate_lebel_per_story.json"),
    )

    # n_parcels and n_networks controls substrate size + IIT tractability
    n_parcels = int(os.environ.get("LEBEL_N_PARCELS", "200"))
    n_networks = int(os.environ.get("LEBEL_N_NETWORKS", "10"))

    print(f"LeBel dataset root: {dataset_root}")
    print(f"Subjects: {subjects}")
    print(f"Parcels: {n_parcels}, networks: {n_networks}")

    all_results = []
    overall_t0 = time.time()

    for subj in subjects:
        stories = requested_stories or list_stories(dataset_root, subj)
        print(f"\n{'='*90}")
        print(f"Subject {subj}: {len(stories)} stories")
        print(f"{'='*90}")
        for i, story in enumerate(stories, 1):
            t0 = time.time()
            try:
                substrate, sentence_texts = build_lebel_substrate(
                    dataset_root=dataset_root,
                    subject_id=subj,
                    story_name=story,
                    n_parcels=n_parcels,
                    n_networks=n_networks,
                )
            except Exception as e:
                print(f"  [{i:>2}/{len(stories)}] {story}  ERROR: {e}")
                all_results.append({
                    "subject_id": subj, "story": story,
                    "error": str(e),
                })
                continue

            scored = score_substrate(substrate)
            elapsed = time.time() - t0
            print(
                f"  [{i:>2}/{len(stories)}] {story:<40}  "
                f"n_sent={substrate.n_timesteps:>3}  "
                f"unified={scored['unified']:6.2f}  ({elapsed:.1f}s)"
            )
            all_results.append({
                "subject_id": subj,
                "story": story,
                "n_sentences": substrate.n_timesteps,
                "n_parcels": substrate.n_nodes,
                "n_networks": n_networks,
                "theories": {t: scored[t] for t in THEORIES},
                "unified": scored["unified"],
                "duration_seconds": round(elapsed, 1),
            })

            # Incremental save every 10 stories
            if i % 10 == 0 or i == len(stories):
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).write_text(json.dumps({
                    "dataset": "lebel_ds003020",
                    "parcellation": f"kmeans-{n_parcels}p-{n_networks}n",
                    "n_subjects": len(subjects),
                    "subjects": subjects,
                    "n_stories_per_subject": len(stories),
                    "n_completed": len(all_results),
                    "results": all_results,
                }, indent=2, default=str))

    total_elapsed = time.time() - overall_t0
    print(f"\n{'='*90}\nDone. {len(all_results)} (subject, story) substrates "
          f"in {total_elapsed/60:.1f} min\n{'='*90}")

    # Summary statistics
    successes = [r for r in all_results if "error" not in r]
    if successes:
        unified_by_subj = {}
        for r in successes:
            unified_by_subj.setdefault(r["subject_id"], []).append(r["unified"])
        for subj, vals in sorted(unified_by_subj.items()):
            arr = np.array(vals)
            print(f"  {subj}: n_stories={len(vals)}  unified mean={arr.mean():.2f}  "
                  f"std={arr.std():.2f}  range=[{arr.min():.2f}, {arr.max():.2f}]")
        all_vals = np.array([r["unified"] for r in successes])
        print(f"\n  ALL: n={len(all_vals)}  unified mean={all_vals.mean():.2f}  "
              f"std={all_vals.std():.2f}  range=[{all_vals.min():.2f}, {all_vals.max():.2f}]")


if __name__ == "__main__":
    main()
