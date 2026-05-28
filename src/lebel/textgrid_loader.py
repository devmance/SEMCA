"""
TextGrid parser for LeBel et al. 2023 dataset.

TextGrid files are Praat-format aligned-transcript files. Each contains
two interval tiers:
  - 'words' — per-word intervals with the spoken word
  - 'phones' (or 'phonemes') — per-phoneme intervals

We aggregate words into sentences using:
  - Sentence-ending punctuation in the word stream
  - Long inter-word pauses as fallback delimiters

This matches how an experimentalist would chunk continuous narrative
into discrete "trials" for evoked-response analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Sentence:
    """One sentence within a story, with word-level timing."""
    text: str                # joined word text, no punctuation
    start_sec: float         # first word onset
    end_sec: float           # last word offset
    words: List["Word"]      # constituent words


@dataclass
class Word:
    text: str
    start_sec: float
    end_sec: float


def _parse_long_textgrid(text: str) -> List[Word]:
    """
    Parse Praat 'long' TextGrid format into a flat list of Word objects.

    The format looks like:

        File type = "ooTextFile"
        Object class = "TextGrid"
        ...
        item [1]:
            class = "IntervalTier"
            name = "words"
            xmin = 0
            xmax = 480
            intervals: size = 4321
            intervals [1]:
                xmin = 0
                xmax = 1.23
                text = "and"
            ...

    We extract only the 'words' tier.
    """
    words: List[Word] = []
    # Find the word tier section. LeBel TextGrids use name = "word" (singular).
    # Pereira / other Praat tools use "words". Support both.
    word_tier_re = re.compile(
        r'item \[\d+\]:.*?name = "words?".*?intervals: size = \d+(.*?)(?=item \[|\Z)',
        flags=re.DOTALL,
    )
    match = word_tier_re.search(text)
    if not match:
        return words
    intervals_text = match.group(1)

    interval_re = re.compile(
        r'intervals \[\d+\]:\s*'
        r'xmin = ([0-9.eE+-]+)\s*'
        r'xmax = ([0-9.eE+-]+)\s*'
        r'text = "([^"]*)"',
    )
    for m in interval_re.finditer(intervals_text):
        start = float(m.group(1))
        end = float(m.group(2))
        word = m.group(3).strip()
        if word and word.lower() not in ("sp", "sil", ""):  # skip silence markers
            words.append(Word(text=word, start_sec=start, end_sec=end))
    return words


def _parse_short_textgrid(text: str) -> List[Word]:
    """
    Parse Praat 'short' TextGrid format into a flat list of Word objects.

    Short format has tier headers then interval entries with no per-line
    labels:
        "ooTextFile"
        "TextGrid"
        0
        480
        <exists>
        2
        "IntervalTier"
        "words"
        0
        480
        4321
        0
        1.23
        "and"
        ...
    """
    lines = [ln.strip() for ln in text.split("\n")]
    if len(lines) < 6 or '"TextGrid"' not in lines[1]:
        return []

    # Find the start of the words tier
    words: List[Word] = []
    i = 0
    while i < len(lines):
        if lines[i] == '"words"':
            # Tier header found at line i; tier xmin/xmax/n_intervals at i+1..i+3
            try:
                # Skip xmin, xmax of the tier
                i += 3
                n_intervals = int(lines[i].strip())
                i += 1
                # Then n_intervals × 3 lines: xmin, xmax, text
                for _ in range(n_intervals):
                    if i + 2 >= len(lines):
                        break
                    start = float(lines[i])
                    end = float(lines[i + 1])
                    word = lines[i + 2].strip().strip('"')
                    if word and word.lower() not in ("sp", "sil", ""):
                        words.append(Word(text=word, start_sec=start, end_sec=end))
                    i += 3
                break
            except (ValueError, IndexError):
                break
        i += 1
    return words


def parse_textgrid(path: str) -> List[Sentence]:
    """
    Parse a Praat TextGrid file and chunk words into sentences.

    Sentence boundaries are determined by:
      - Words ending in '.', '!', '?'  → sentence ends here
      - Pauses > 0.8 sec between consecutive words → sentence boundary
        (fallback for stories without punctuation in alignments)

    Returns a list of Sentence objects with timing.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    # Try long format first, fall back to short
    words = _parse_long_textgrid(text)
    if not words:
        words = _parse_short_textgrid(text)
    if not words:
        return []
    return _group_into_sentences(words)


def _group_into_sentences(
    words: List[Word],
    pause_threshold_sec: float = 0.8,
) -> List[Sentence]:
    """Group a flat word list into sentences by punctuation + pauses."""
    sentences: List[Sentence] = []
    current: List[Word] = []
    end_punct_re = re.compile(r"[.!?]+$")

    for i, w in enumerate(words):
        current.append(w)
        # Decide whether this word ends a sentence
        is_end = False
        # Punctuation in word text
        if end_punct_re.search(w.text):
            is_end = True
        # Long pause before next word
        elif i + 1 < len(words):
            gap = words[i + 1].start_sec - w.end_sec
            if gap > pause_threshold_sec:
                is_end = True
        else:
            # last word in story → finalize
            is_end = True

        if is_end and current:
            text = " ".join(re.sub(r"[.!?,;:]+$", "", x.text) for x in current).strip()
            if text:
                sentences.append(Sentence(
                    text=text,
                    start_sec=current[0].start_sec,
                    end_sec=current[-1].end_sec,
                    words=list(current),
                ))
            current = []
    return sentences
