"""Pair-level feature engineering for the Source-1 <-> Source-2/3 matcher.

Takes normalized fields already computed by db.py plus the blocking score,
and produces a fixed-order numeric feature vector per candidate pair. Uses
rapidfuzz (C-optimized) for edit-distance-family features to keep this fast
enough to run over tens of millions of pairs on a 4-core machine.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from normalize import (
    addr_tokens,
    digit_tokens,
    name_phonetic,
    name_sorted_key,
    name_tokens,
    phonetic_tokens,
    postal_tokens,
)

from rapidfuzz import fuzz

FEATURE_NAMES = [
    "blocking_score",
    "cand_source",  # 2 or 3
    "name_jaccard",
    "name_overlap_coef",
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_prefix_match",
    "name_full_exact",
    "name_len_ratio",
    "addr_jaccard",
    "addr_overlap_coef",
    "addr_levenshtein_ratio",
    "addr_digit_jaccard",
    "s1_addr_empty",
    "cand_addr_empty",
    # Added alongside the meta-blocking rewrite: these capture the signals the
    # new blocking key types are built on, so the model can weigh them directly
    # rather than only seeing their aggregate in blocking_score.
    "name_sorted_exact",
    "name_phon_exact",
    "phon_token_jaccard",
    "postal_exact",
    "n_shared_name_tokens",
    "n_shared_addr_tokens",
]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def _overlap_coef(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _fetch_fields(conn, table, ids):
    """Return {entity_id: (name_norm, addr_norm, business_address)} for the given ids."""
    out = {}
    ids = list(ids)
    CHUNK = 900  # stay under SQLite's default ~999 host-parameter limit
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i:i + CHUNK]
        placeholders = ",".join("?" * len(chunk))
        q = f"SELECT entity_id, name_norm, addr_norm, business_address FROM {table} WHERE entity_id IN ({placeholders})"
        for row in conn.execute(q, chunk):
            out[row[0]] = (row[1], row[2], row[3])
    return out


def featurize_batch(conn, batch_results):
    """batch_results: list of (s1_id, [(cand_id, cand_source, score), ...]).

    Returns (X: list[list[float]], s1_ids: list[str], cand_ids: list[str]) with
    one row per (s1_id, cand_id) pair, in the same order across the three.
    """
    s1_needed = [s1 for s1, _ in batch_results]
    cand2_needed, cand3_needed = [], []
    for _, cands in batch_results:
        for cid, csrc, _ in cands:
            (cand2_needed if csrc == 2 else cand3_needed).append(cid)

    s1_fields = _fetch_fields(conn, "source1", s1_needed)
    s2_fields = _fetch_fields(conn, "source2", cand2_needed) if cand2_needed else {}
    s3_fields = _fetch_fields(conn, "source3", cand3_needed) if cand3_needed else {}

    X, out_s1_ids, out_cand_ids = [], [], []
    for s1_id, cands in batch_results:
        s1_name_norm, s1_addr_norm, s1_raw_addr = s1_fields.get(s1_id, ("", "", ""))
        for cid, csrc, score in cands:
            fields = s2_fields if csrc == 2 else s3_fields
            cand_name_norm, cand_addr_norm, cand_raw_addr = fields.get(cid, ("", "", ""))
            feats = compute_features(
                s1_name_norm, s1_addr_norm, s1_raw_addr,
                cand_name_norm, cand_addr_norm, cand_raw_addr,
                csrc, score,
            )
            X.append(feats)
            out_s1_ids.append(s1_id)
            out_cand_ids.append(cid)
    return X, out_s1_ids, out_cand_ids


def compute_features(s1_name_norm, s1_addr_norm, s1_raw_addr,
                      cand_name_norm, cand_addr_norm, cand_raw_addr,
                      cand_source, blocking_score):
    nt1, nt2 = name_tokens(s1_name_norm), name_tokens(cand_name_norm)
    at1, at2 = addr_tokens(s1_addr_norm), addr_tokens(cand_addr_norm)
    dt1, dt2 = digit_tokens(s1_raw_addr), digit_tokens(cand_raw_addr)

    len1, len2 = len(s1_name_norm), len(cand_name_norm)
    len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 0.0

    prefix1 = s1_name_norm.replace(" ", "")[:4]
    prefix2 = cand_name_norm.replace(" ", "")[:4]

    sorted1, sorted2 = name_sorted_key(s1_name_norm), name_sorted_key(cand_name_norm)
    phon1, phon2 = name_phonetic(s1_name_norm), name_phonetic(cand_name_norm)
    pt1, pt2 = phonetic_tokens(s1_name_norm), phonetic_tokens(cand_name_norm)
    post1, post2 = postal_tokens(s1_raw_addr), postal_tokens(cand_raw_addr)

    return [
        float(blocking_score),
        float(cand_source),
        _jaccard(nt1, nt2),
        _overlap_coef(nt1, nt2),
        fuzz.ratio(s1_name_norm, cand_name_norm) / 100.0,
        fuzz.token_sort_ratio(s1_name_norm, cand_name_norm) / 100.0,
        fuzz.token_set_ratio(s1_name_norm, cand_name_norm) / 100.0,
        fuzz.partial_ratio(s1_name_norm, cand_name_norm) / 100.0,
        1.0 if (prefix1 and prefix1 == prefix2) else 0.0,
        1.0 if (s1_name_norm and s1_name_norm == cand_name_norm) else 0.0,
        len_ratio,
        _jaccard(at1, at2),
        _overlap_coef(at1, at2),
        fuzz.ratio(s1_addr_norm, cand_addr_norm) / 100.0 if (s1_addr_norm or cand_addr_norm) else 0.0,
        _jaccard(dt1, dt2),
        1.0 if not s1_addr_norm else 0.0,
        1.0 if not cand_addr_norm else 0.0,
        1.0 if (sorted1 and sorted1 == sorted2) else 0.0,
        1.0 if (phon1 and phon1 == phon2) else 0.0,
        _jaccard(pt1, pt2),
        1.0 if (post1 & post2) else 0.0,
        float(len(nt1 & nt2)),
        float(len(at1 & at2)),
    ]
