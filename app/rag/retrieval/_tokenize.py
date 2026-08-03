"""Tokenization for BM25 — the sparse side's equivalent of an embedding model.

Dense retrieval matches meaning; BM25 matches strings. So the tokenizer *is* the
sparse retriever's semantics: whether "embeddings" and "embedding" are the same
term is decided here, and nowhere else.

Stemming uses snowballstemmer rather than nltk. nltk 3.10 ships an import guard
that blocks any module resolving inside the current working directory, and a
``.venv/`` inside the project — the ordinary layout — trips it, so ``import
nltk`` fails outright here. snowballstemmer is the same Porter algorithm,
needs no corpus download, and has no such guard.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Kept small and explicit rather than pulled from a corpus download. These are
# the words that appear in nearly every chunk of every paper, so they cost BM25
# time and add nothing: a term present everywhere discriminates nothing.
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can cannot could did do does doing down
    during each few for from further had has have having he her here hers herself
    him himself his how i if in into is it its itself just me more most my myself
    no nor not now of off on once only or other our ours ourselves out over own
    same she should so some such than that the their theirs them themselves then
    there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours
    yourself yourselves
    """.split()
)

# Words and bare numbers. Numbers are kept: in this corpus they carry real
# signal — model sizes, years, equation and figure references.
_WORD = re.compile(r"[a-z0-9]+")

# Single characters are almost always fragments of split maths or list markers.
MIN_TOKEN_LEN = 2


@lru_cache(maxsize=1)
def _stemmer():
    """Built once. Constructing a stemmer per call dominates tokenization time."""
    import snowballstemmer

    return snowballstemmer.stemmer("porter")


def tokenize(text: str, *, stem: bool = True, remove_stopwords: bool = True) -> list[str]:
    """Split ``text`` into BM25 terms.

    ``stem=False`` is the A/B control: it keeps surface forms, which helps for
    exact identifiers ("BM25", "GPT-4") and hurts for ordinary morphology
    ("retrieving" no longer matches "retrieval").
    """
    words = [word for word in _WORD.findall(text.lower()) if len(word) >= MIN_TOKEN_LEN]
    if remove_stopwords:
        words = [word for word in words if word not in STOPWORDS]
    if not stem or not words:
        return words
    return _stemmer().stemWords(words)
