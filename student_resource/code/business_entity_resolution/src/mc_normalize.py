"""Normalization for the multi-channel (mc_*) retrieval architecture.

Kept separate from normalize.py on purpose: the original pipeline (which
produced the 0.767 submission) must remain exactly reproducible, and these
transforms change what its keys and features would be.

Each transform targets a failure class observed in retrieval misses:
  - DBA wrappers ("Wexariawex D.B.A. Baba Exports Limited")
  - honorific / domain noise ("Mr GLOBAL FOOD", "smartit.com", "M/s ...")
  - native-script names that are English words transliterated into
    Devanagari/Gujarati/Kannada/Tamil: handled by a phonetic skeleton that
    maps the Latin rendering of both spellings onto the same consonant frame
  - compound house numbers ("6-3-10/3", "D-2/2A") that the generic cleaner
    shreds into common digits; preserved here as single rare tokens
  - address placeholders ("<NULL>", "null")
"""

import re

from normalize import (
    _ADDR_ABBREV,
    _LEGAL_SUFFIXES,
    basic_clean,
    house_tokens,
    postal_tokens,
)

_NAME_NOISE = {"mr", "mrs", "ms", "dr", "smt", "shri", "the", "www", "com",
               "net", "org", "co", "in"}
_ADDR_NOISE = {"null", "none", "na", "nil"}
_MS_RE = re.compile(r"^\s*m\s*/\s*s\.?\s+", re.I)
_DBA_RE = re.compile(r"\b(?:d\s*b\s*a|doing business as)\b")

# Applied in order. 'ch' -> 'c' -> 'k' is intentional: unidecode renders
# Devanagari च as 'c' while English spells the sound 'ch', so both must land
# on the same symbol. Collapsing c/k costs a little precision in English and
# buys cross-script recall, which is the scarcer thing here.
# Soft g/c come first: English "energy", "engineering", "agency" carry a soft
# g that Indic scripts write as ज (j); "center", "services" a soft c written
# as स (s). Mapping them before the generic c->k keeps cross-script spellings
# of the same word on the same skeleton.
_PHON = [
    ("ge", "je"), ("gi", "ji"), ("gy", "ji"),
    ("ce", "se"), ("ci", "si"), ("cy", "si"),
    ("ph", "f"), ("gh", "g"), ("ck", "k"), ("kh", "k"), ("th", "t"),
    ("dh", "d"), ("bh", "b"), ("sh", "s"), ("ch", "c"), ("qu", "k"),
    ("q", "k"), ("c", "k"), ("x", "ks"), ("z", "s"), ("w", "v"), ("y", "i"),
]
_VOWELS = set("aeiou")
_JOINERS = dict.fromkeys(map(ord, "‌‍​﻿"), None)


def norm_name(raw: str) -> str:
    if not raw:
        return ""
    # Zero-width joiners inside Indic words otherwise become word breaks
    # after transliteration ("ಇನ್‌ಫೋಟೆಕ್" -> "in" + "photek").
    raw = _MS_RE.sub("", raw.translate(_JOINERS))
    t = basic_clean(raw)
    m = _DBA_RE.search(t)
    if m and t[m.end():].strip():
        t = t[m.end():]
    toks = [w for w in t.split() if w not in _LEGAL_SUFFIXES and w not in _NAME_NOISE]
    return " ".join(toks)


def norm_addr(raw: str) -> str:
    """Cleaned address plus compound house numbers re-appended as single
    tokens ('6-3-10/3' -> 'h6x3x10x3') so char n-grams keep them intact."""
    if not raw:
        return ""
    toks = [_ADDR_ABBREV.get(w, w) for w in basic_clean(raw).split()
            if w not in _ADDR_NOISE]
    toks += ["h" + re.sub(r"[-/]", "x", h) for h in sorted(house_tokens(raw))]
    return " ".join(toks)


def phon_word(w: str) -> str:
    t = w.lower()
    for a, b in _PHON:
        t = t.replace(a, b)
    if not t:
        return ""
    out = t[0] + "".join(c for c in t[1:] if c not in _VOWELS)
    dedup = []
    for c in out:
        if not dedup or dedup[-1] != c:
            dedup.append(c)
    return "".join(dedup)


# Skeletons of legal-form words, so a transliterated "प्राइवेट लिमिटेड"
# (-> "prvt lmtd") is stripped just like the English "Private Limited" was at
# the text level. Plus spelled-out forms ("एलएलपी" = "el-el-pi" -> "elp").
# Length >= 3 only: shorter skeletons ("k", "lp") collide with real words.
_PHON_LEGAL = {p for p in (phon_word(s) for s in _LEGAL_SUFFIXES) if len(p) >= 3}
_PHON_LEGAL |= {"elp", "elelp", "pvtltd", "prvtlmtd"}


def phon_string(name_n: str) -> str:
    """Phonetic skeleton of a normalized name, word order preserved."""
    ps = (phon_word(w) for w in name_n.split())
    return " ".join(p for p in ps if p and p not in _PHON_LEGAL)


def phon_compact(name_n: str) -> str:
    """Skeleton with word breaks removed. Char n-grams over this are immune to
    spurious splits from transliteration ("je n inf tk" vs "jn inftk")."""
    return phon_string(name_n).replace(" ", "")


def name_sorted(name_n: str) -> str:
    return " ".join(sorted(name_n.split()))


def house_set(raw: str) -> frozenset:
    return frozenset(house_tokens(raw))


def postal_set(raw: str) -> frozenset:
    return frozenset(postal_tokens(raw))
