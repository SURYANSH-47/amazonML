"""Text normalization for business names and addresses.

Everything here is a deterministic, rule-based transform (Unicode tables,
a hand-maintained legal-suffix list, regex tokenization) with no network
access or external lookup of any kind.
"""

import re
import unicodedata

from unidecode import unidecode

# Legal-entity suffixes observed in the US / India / France training and test
# data. Stripped from the *blocking/comparison* form of the name only (the raw
# business_name is preserved untouched in the DB) so "Acme Corp" and
# "Acme Corporation" collapse to the same token set.
_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "llp", "ltd", "limited", "pvt", "private",
    "sarl", "sasu", "sas", "sa", "eurl", "eirl", "snc",
    "group", "holdings", "enterprises", "enterprise",
}

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")


def strip_accents_and_transliterate(text: str) -> str:
    """Fold accented Latin to plain ASCII and transliterate other scripts.

    NFKD strips combining marks (handles French accents generically, not
    hardcoded per country). unidecode is a bundled, deterministic
    script->Latin table (Devanagari/Tamil/Kannada/... -> ASCII) — used as a
    fallback for names in non-Latin scripts, common in the Source 2/3 India
    records, so they become comparable to the always-Latin Source 1 names.
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
    return basic_clean(address)


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


def name_prefix(name_norm: str, length: int = 4) -> str:
    compact = name_norm.replace(" ", "")
    return compact[:length]
