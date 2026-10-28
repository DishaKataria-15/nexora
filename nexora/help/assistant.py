"""Offline help assistant: matches a typed question to the built-in answers.

Matching uses TF-IDF over the question phrasings (word and character n-grams, so small typos still match), with a
small boost for answers that belong to the page the user is on. Below a confidence threshold the assistant says it
does not know rather than guessing, and offers the closest topics instead.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from .kb import ANY, KB, SUGGESTED

THRESHOLD = 0.26
PAGE_BOOST = 0.06
SYNONYMS = {"colour": "color", "colours": "color", "visualisation": "visualization", "visualise": "visualize", "analyse": "analyze",
            "summarise": "summarize", "optimise": "optimize", "normalise": "normalize", "behaviour": "behavior"}


@dataclass
class Answer:
    found: bool
    entry: dict | None
    score: float
    related: list[dict] = field(default_factory=list)


def _norm(text: str) -> str:
    t = re.sub(r"[^a-z0-9%+\- ]", " ", text.lower())
    return " ".join(SYNONYMS.get(w, w) for w in t.split())


GENERIC = {"a", "an", "the", "is", "are", "was", "be", "am", "i", "me", "my", "we", "our", "you", "your", "it", "its", "this", "that", "these",
           "those", "to", "of", "in", "on", "at", "for", "with", "and", "or", "but", "so", "if", "do", "does", "did", "can", "could", "should",
           "would", "will", "how", "what", "why", "when", "where", "which", "who", "whom", "there", "here", "about", "from", "by", "as", "into",
           "any", "some", "please", "tell", "explain", "know", "want", "need", "help", "hi", "hello", "hey", "ok", "okay", "thanks", "today",
           "now", "get", "got", "use", "using", "make", "made", "one", "all", "just", "also", "very", "not", "no", "yes"}


def _content(text: str) -> set[str]:
    return {w for w in _norm(text).split() if w not in GENERIC and len(w) > 1}


@lru_cache(maxsize=None)
def _vocab(i: int) -> frozenset[str]:
    e = KB[i]
    return frozenset(_content(" ".join(e["q"]) + " " + re.sub(r"[*`]", " ", e["a"])))


def _overlaps(words: set[str], i: int) -> bool:
    """At least one meaningful word of the question appears in the answer's topic (allowing small typos)."""
    vocab = _vocab(i)
    for w in words:
        if w in vocab or any(v.startswith(w) or w.startswith(v) for v in vocab if min(len(v), len(w)) >= 4):
            return True
        if len(w) >= 5 and difflib.get_close_matches(w, vocab, n=1, cutoff=0.8):
            return True
    return False


@lru_cache(maxsize=1)
def _index():
    docs, owner = [], []
    for i, e in enumerate(KB):
        for q in e["q"]:
            docs.append(_norm(q))
            owner.append(i)
        docs.append(_norm(" ".join(e["q"]) + " " + re.sub(r"[*`]", "", e["a"])))   # the answer text helps with unusual wording
        owner.append(i)
    vec = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
                        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))])
    m = vec.fit_transform(docs)
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1
    return vec, m, norms, np.array(owner)


def ask(question: str, page: str | None = None, top: int = 3) -> Answer:
    q = _norm(question or "")
    if len(q) < 2:
        return Answer(False, None, 0.0)
    vec, m, norms, owner = _index()
    qv = vec.transform([q])
    qn = float(np.sqrt(qv.multiply(qv).sum())) or 1.0
    sims = np.asarray((m @ qv.T).todense()).ravel() / (norms * qn)
    best: dict[int, float] = {}
    for s, o in zip(sims, owner):
        if s > best.get(o, -1):
            best[o] = float(s)
    for o in best:
        if page and KB[o]["page"] == page:
            best[o] += PAGE_BOOST
    ranked = sorted(best.items(), key=lambda t: -t[1])
    words = _content(question)
    if not words:
        return Answer(False, None, 0.0, [KB[i] for i, _ in ranked[:top]])
    ranked = [(i, s if _overlaps(words, i) else s * 0.5) for i, s in ranked]        # similar letters alone are not enough
    ranked.sort(key=lambda t: -t[1])
    top_i, top_s = ranked[0]
    related = [KB[i] for i, s in ranked[1:1 + top] if s >= THRESHOLD * 0.6]
    if top_s < THRESHOLD:
        return Answer(False, None, top_s, [KB[i] for i, _ in ranked[:top]])
    return Answer(True, KB[top_i], top_s, related)


def suggestions(page: str | None) -> list[dict]:
    ids = SUGGESTED.get(page or "", []) or ["start", "workflow", "privacy"]
    by_id = {e["id"]: e for e in KB}
    return [by_id[i] for i in ids if i in by_id]


def entry(entry_id: str) -> dict | None:
    return next((e for e in KB if e["id"] == entry_id), None)


def topics() -> dict[str, list[dict]]:
    """All entries grouped by page, for a browsable list."""
    out: dict[str, list[dict]] = {}
    for e in KB:
        out.setdefault("General" if e["page"] == ANY else e["page"], []).append(e)
    return out
