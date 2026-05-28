"""
Run all 7 substrate-agnostic consciousness theories on the LeBel et al.
2023 *story text* via open-weight AI models.

Pairs with `examples/run_lebel_fmri_substrate.py` (human-substrate side).
For each LeBel story we have:
  - Human substrate: per-subject per-story BOLD evoked patterns
    (already scored — fmri_substrate_lebel_per_story.json)
  - AI substrate (this script): theory scores from running the
    open-weight model on the story's text content

This enables per-story cross-substrate correlation analysis (paper §4.2).

Story text is extracted from each TextGrid's word tier and concatenated.
Stories are truncated to MAX_STORY_TOKENS tokens to keep memory
manageable on 2× H100 80GB.

Usage:
    MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python -m examples.run_lebel_ai_substrate
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
import torch

from src.lebel import parse_textgrid
from src.model_loader import ModelInternalsRunner
from src.substrate import TransformerSubstrate, TransformerPredictionAdapter
from src.theories_v2 import (
    compute_iit_phi, compute_gwt, compute_hot, compute_ast,
    compute_ppt, compute_fep, compute_qit,
)


MAX_STORY_TOKENS = int(os.environ.get("MAX_STORY_TOKENS", "1500"))
# Path to the LeBel ds003020 TextGrid alignments. Download the dataset from
# https://openneuro.org/datasets/ds003020/ and point TEXTGRIDS_DIR at
# .../derivatives/TextGrids/ (env var override).
TEXTGRIDS_DIR = os.environ.get(
    "TEXTGRIDS_DIR",
    str(ROOT / "data" / "ds003020" / "derivatives" / "TextGrids"),
)


def extract_story_text(textgrid_path: str) -> str:
    """Extract the full story transcript by joining all words from
    the TextGrid's word tier."""
    sentences = parse_textgrid(textgrid_path)
    if not sentences:
        return ""
    # Join sentences with periods to mimic natural sentence boundaries
    return ". ".join(s.text for s in sentences) + "."


def list_lebel_stories(textgrids_dir: str) -> list:
    """Return all story names (without .TextGrid extension), sorted."""
    p = Path(textgrids_dir)
    return sorted([f.stem for f in p.glob("*.TextGrid")])


def score_substrate(record, substrate_unit: str = "head") -> dict:
    sub = TransformerSubstrate(record, unit=substrate_unit)
    pred = TransformerPredictionAdapter(record)
    r_iit = compute_iit_phi(sub)
    r_gwt = compute_gwt(sub)
    r_hot = compute_hot(sub)
    r_ast = compute_ast(sub)
    r_ppt = compute_ppt(sub, pred)
    r_fep = compute_fep(sub, pred)
    r_qit = compute_qit(sub)
    vals = [r_iit.iit_value, r_gwt.gwt_value, r_ast.ast_value, r_hot.hot_value,
            r_ppt.ppt_value, r_qit.qit_value, r_fep.fep_value]
    unified = float(np.mean(vals))
    return {
        "IIT": {"score": r_iit.iit_value, "components": r_iit.components,
                "confidence": r_iit.confidence},
        "GWT": {"score": r_gwt.gwt_value, "components": r_gwt.components,
                "confidence": r_gwt.confidence},
        "AST": {"score": r_ast.ast_value, "components": r_ast.components,
                "confidence": r_ast.confidence},
        "HOT": {"score": r_hot.hot_value, "components": r_hot.components,
                "confidence": r_hot.confidence},
        "PPT": {"score": r_ppt.ppt_value, "components": r_ppt.components,
                "confidence": r_ppt.confidence},
        "FEP": {"score": r_fep.fep_value, "components": r_fep.components,
                "confidence": r_fep.confidence},
        "QIT": {"score": r_qit.qit_value, "components": r_qit.components,
                "confidence": r_qit.confidence},
        "unified": unified,
    }


def main():
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    output_path = os.environ.get(
        "OUTPUT_PATH",
        str(ROOT / "data" / f"lebel_ai_{model_name.replace('/', '__')}.json"),
    )

    stories = list_lebel_stories(TEXTGRIDS_DIR)
    print(f"Found {len(stories)} LeBel TextGrid stories at {TEXTGRIDS_DIR}")
    print(f"Model: {model_name}")
    print(f"Output: {output_path}")
    print(f"Max tokens per story: {MAX_STORY_TOKENS}")

    print(f"\nLoading {model_name}...")
    runner = ModelInternalsRunner(model_name, max_new_tokens=0)
    print(f"  Device: {runner.device}, GPUs: {torch.cuda.device_count()}")

    results = []
    overall_t0 = time.time()
    for i, story in enumerate(stories, 1):
        t0 = time.time()
        tg_path = Path(TEXTGRIDS_DIR) / f"{story}.TextGrid"
        text = extract_story_text(str(tg_path))
        if not text:
            print(f"  [{i:>2}/{len(stories)}] {story} SKIP (no text)")
            continue

        # Truncate by tokens: encode, slice to MAX_STORY_TOKENS, decode
        token_ids = runner.tokenizer.encode(text, add_special_tokens=False)
        n_tokens_full = len(token_ids)
        if n_tokens_full > MAX_STORY_TOKENS:
            token_ids = token_ids[:MAX_STORY_TOKENS]
            text = runner.tokenizer.decode(token_ids, skip_special_tokens=True)

        try:
            record = runner.encode_with_internals(text)
        except Exception as e:
            print(f"  [{i:>2}/{len(stories)}] {story} ERROR: {e}")
            results.append({"story": story, "error": str(e)})
            continue

        scored = score_substrate(record)
        elapsed = time.time() - t0
        print(
            f"  [{i:>2}/{len(stories)}] {story:<40}  "
            f"seq={record.seq_len:>4} (orig {n_tokens_full:>5}) "
            f"unified={scored['unified']:6.2f}  ({elapsed:.1f}s)"
        )
        results.append({
            "story": story,
            "model_name": model_name,
            "n_tokens_full": n_tokens_full,
            "n_tokens_encoded": record.seq_len,
            "truncated": n_tokens_full > MAX_STORY_TOKENS,
            "theories": {k: v for k, v in scored.items() if k != "unified"},
            "unified": scored["unified"],
            "duration_seconds": round(elapsed, 1),
        })

        # Incremental save every 5 stories
        if i % 5 == 0 or i == len(stories):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json.dumps({
                "model_name": model_name,
                "stimulus_set": "lebel_ds003020_stories",
                "max_story_tokens": MAX_STORY_TOKENS,
                "n_stories_processed": len(results),
                "results": results,
            }, indent=2, default=str))

    total = time.time() - overall_t0
    print(f"\n{'='*90}\nDone. {len(results)} stories in {total/60:.1f} min")

    successes = [r for r in results if "error" not in r]
    if successes:
        u = np.array([r["unified"] for r in successes])
        print(f"  Unified: n={len(u)}  mean={u.mean():.2f}  std={u.std():.2f}  "
              f"range=[{u.min():.2f}, {u.max():.2f}]")


if __name__ == "__main__":
    main()
