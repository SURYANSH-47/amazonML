"""Text normalization for business names and addresses.

Everything here is a deterministic, rule-based transform (Unicode tables,
hand-maintained abbreviation/suffix lists, regex tokenization) with no network
access or external lookup of any kind.

The transforms target the noise patterns the challenge documents:
abbreviation variants (Corp/Corporation, Rd/Road), legal-suffix drift,
punctuation and '&'/'and' differences, word-order transposition, typos, and
non-Latin script variants of the same name.
"""

import re
import unicodedata

from unidecode import unidecode

# True legal-entity forms only (US / India / France patterns seen in the data).
# Stripped from the comparison form of the name so "Acme Corp" and
# "Acme Corporation" collapse together. Deliberately does NOT include words
# like "group"/"holdings"/"enterprises": those are name content, and dropping
# them would collapse genuinely different businesses ("X Holdings" vs
# "X Enterprises") onto the same key.
_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "llp", "ltd", "limited", "pvt", "private", "plc",
    "sarl", "sasu", "sas", "sa", "eurl", "eirl", "snc", "sci",
    "gmbh", "bv", "nv", "ag", "spa", "srl",
}

# Address abbreviation canonicalization — "Rd vs Road", "St vs Street" etc. are
# called out explicitly as a noise pattern. Mapping both directions to one
# canonical form makes address token overlap far more reliable.
_ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "pl": "place", "sq": "square", "hwy": "highway", "pkwy": "parkway",
    "ste": "suite", "apt": "apartment", "bldg": "building", "fl": "floor",
    "flr": "floor", "rm": "room", "dept": "department", "opp": "opposite",
    "nr": "near", "no": "number", "hno": "number", "sec": "sector",
    "ph": "phase", "colo": "colony", "nagar": "nagar", "mkt": "market",
    "ext": "extension", "cross": "cross", "main": "main",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "rue": "rue", "bd": "boulevard", "av.": "avenue",
}

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")
_VOWELS = set("aeiou")

# Consonant-cluster canonicalization for the phonetic skeleton, applied before
# vowel removal. Handles typo/transliteration variants that differ only in how
# a sound was spelled (shakti/shakthi, photo/foto, kwik/quick).
_PHON_SUBS = [
    ("ph", "f"), ("gh", "g"), ("ck", "k"), ("kh", "k"), ("th", "t"),
    ("dh", "d"), ("bh", "b"), ("sh", "s"), ("ch", "c"), ("qu", "k"),
    ("q", "k"), ("x", "ks"), ("z", "s"), ("w", "v"), ("y", "i"),
]


def strip_accents_and_transliterate(text: str) -> str:
    """Fold accented Latin to plain ASCII and transliterate other scripts.

    NFKD strips combining marks (handles French accents generically, not
    hardcoded per country). unidecode is a bundled, deterministic
    script->Latin table (Devanagari/Tamil/Kannada/... -> ASCII) — needed
    because Source 1 names are always Latin while Source 2/3 India records
    are sometimes in native script for the same business.
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_folded = "".join(c for c in nfkd if not unicodedata.combining(c))
    if ascii_folded.isascii():
        return ascii_folded
    return unidecode(ascii_folded)


def basic_clean(text: str) -> str:
    if not text:
        return ""
    text = strip_accents_and_transliterate(text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_name(name: str) -> str:
    """Lowercased, accent/script-folded, punctuation-stripped, legal-suffix-free name."""
    cleaned = basic_clean(name)
    if not cleaned:
        return ""
    tokens = [t for t in cleaned.split(" ") if t and t not in _LEGAL_SUFFIXES]
    return " ".join(tokens)


def normalize_address(address: str) -> str:
    """Cleaned address with common abbreviations expanded to one canonical form."""
    cleaned = basic_clean(address)
    if not cleaned:
        return ""
    tokens = [_ADDR_ABBREV.get(t, t) for t in cleaned.split(" ") if t]
    return " ".join(tokens)


def name_tokens(name_norm: str) -> set:
    if not name_norm:
        return set()
    return {t for t in name_norm.split(" ") if len(t) > 2}


def addr_tokens(addr_norm: str) -> set:
    if not addr_norm:
        return set()
    return {t for t in addr_norm.split(" ") if len(t) > 2}


def digit_tokens(text: str) -> set:
    """Numeric substrings (house/unit/PIN/zip-like numbers) from raw or normalized text."""
    if not text:
        return set()
    return {d for d in _DIGIT_RE.findall(text) if len(d) >= 2}


def postal_tokens(text: str) -> set:
    """Digit runs that look like a postal code: 5-digit (US/France) or 6-digit (India PIN).

    Kept separate from digit_tokens because a postal code is far more
    discriminative than a house number.
    """
    if not text:
        return set()
    return {d for d in _DIGIT_RE.findall(text) if len(d) in (5, 6)}


def name_prefix(name_norm: str, length: int = 4) -> str:
    compact = name_norm.replace(" ", "")
    return compact[:length]


def name_sorted_key(name_norm: str) -> str:
    """Tokens sorted alphabetically — makes word-order transposition a no-op.

    "Chinglepet Private Limited Center" and "Center Chinglepet" collapse to
    the same key once suffixes are stripped and tokens sorted.
    """
    toks = sorted(t for t in name_norm.split(" ") if t)
    return " ".join(toks)


def phonetic_key(token: str) -> str:
    """Compact consonant skeleton: canonicalize clusters, drop non-leading vowels.

    Cheap, dependency-free, and robust to the two noise types that break exact
    token matching here — vowel-level typos and transliteration spelling drift
    (shivshakti/shivshakthi, kumar/koomar).
    """
    if not token:
        return ""
    t = token.lower()
    for a, b in _PHON_SUBS:
        t = t.replace(a, b)
    if not t:
        return ""
    head, rest = t[0], t[1:]
    rest = "".join(c for c in rest if c not in _VOWELS)
    out = head + rest
    # collapse runs of the same character
    collapsed = []
    for c in out:
        if not collapsed or collapsed[-1] != c:
            collapsed.append(c)
    return "".join(collapsed)


def name_phonetic(name_norm: str) -> str:
    """Phonetic skeleton of the whole name, tokens sorted for order-independence."""
    toks = sorted(phonetic_key(t) for t in name_norm.split(" ") if len(t) > 2)
    return "".join(toks)


def phonetic_tokens(name_norm: str) -> set:
    return {p for p in (phonetic_key(t) for t in name_norm.split(" ") if len(t) > 2) if len(p) >= 3}
