"""Blocking / candidate generation.

Strategy: inverted-index blocking over multiple key types (name tokens,
address tokens, digit tokens pulled from the address, name prefix), scoped to
matching country (verified on a 69k true-match sample: country is 100%
consistent between Source-1 and its true Source-2/3 matches). Overly common
keys ("inc", "store", generic city names) are purged before joining so the
SQL join never explodes on them — standard block-purging technique.

Everything runs as SQL aggregation in SQLite so peak Python memory stays
bounded regardless of table size; only one batch of Source-1 entities' worth
of candidates is materialized in Python at a time.
"""

import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from db import connect as db_connect

KEY_WEIGHT = {
    "digit_token": 3,
    "name_full": 6,
    "name_prefix": 2,
    "name_token": 1,
    "addr_token": 1,
}

DEFAULT_MAX_BLOCK_SIZE = 2000
# name_full is a much stronger signal (whole normalized name matches exactly),
# so it tolerates a far higher block size before being purged as uninformative.
DEFAULT_MAX_BLOCK_SIZE_OVERRIDES = {"name_full": 6000}
DEFAULT_MIN_SCORE = 2
DEFAULT_TOP_K = 25
DEFAULT_BATCH_SIZE = 1000


def purge_common_keys(
    conn: sqlite3.Connection,
    max_block_size: int = DEFAULT_MAX_BLOCK_SIZE,
    overrides: dict = None,
):
    """Drop (key_type, key_value, country) triples whose Source-2/3 frequency
    exceeds the cap, WITHIN that country.

    Frequency is computed per-country (not globally) because blocking joins
    are country-scoped: a key that's extremely common in one country but rare
    in another should only be purged where it's actually too common, not
    globally deleted and lost for the country where it was still selective.

    `overrides` lets specific key_types (e.g. name_full) use a different cap
    than the default, since some key types carry much stronger match signal
    per occurrence and can tolerate larger blocks.
    """
    overrides = overrides or {}
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS key_freq;")
    cur.execute(
        """
        CREATE TABLE key_freq AS
        SELECT key_type, key_value, country, COUNT(*) AS freq
        FROM blocking_keys
        WHERE source IN (2, 3)
        GROUP BY key_type, key_value, country
        """
    )
    cur.execute("CREATE INDEX idx_key_freq ON key_freq(key_type, key_value, country);")
    before = cur.execute("SELECT COUNT(*) FROM blocking_keys").fetchone()[0]

    key_types = [r[0] for r in cur.execute("SELECT DISTINCT key_type FROM key_freq")]
    for kt in key_types:
        cap = overrides.get(kt, max_block_size)
        cur.execute(
            """
            DELETE FROM blocking_keys
            WHERE key_type = ?
              AND (key_type, key_value, country) IN (
                  SELECT key_type, key_value, country FROM key_freq WHERE key_type = ? AND freq > ?
              )
            """,
            (kt, kt, cap),
        )
    conn.commit()
    after = cur.execute("SELECT COUNT(*) FROM blocking_keys").fetchone()[0]
    cur.execute("DROP TABLE key_freq;")
    conn.commit()
    print(f"  block purging: {before} -> {after} blocking_keys rows (max_block_size={max_block_size}, overrides={overrides})", flush=True)


def _score_expr():
    cases = " ".join(f"WHEN '{k}' THEN {w}" for k, w in KEY_WEIGHT.items())
    return f"SUM(CASE a.key_type {cases} ELSE 1 END)"


def generate_candidates(
    conn: sqlite3.Connection,
    top_k: int = DEFAULT_TOP_K,
    min_score: int = DEFAULT_MIN_SCORE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit_s1: int = None,
    s1_ids: list = None,
    progress_every: int = 50,
):
    """Yield (s1_entity_id, [(cand_entity_id, cand_source, score), ...]) sorted by score desc, capped at top_k.

    By default iterates every Source-1 entity in the DB. Pass `s1_ids` to
    restrict to a specific subset (e.g. a train/validation split or a random
    subsample for building the training set) without touching the DB.
    """
    cur = conn.cursor()
    conn.execute("DROP TABLE IF EXISTS _batch_ids;")
    conn.execute("CREATE TEMP TABLE _batch_ids (entity_id TEXT PRIMARY KEY);")

    if s1_ids is None:
        s1_ids = [r[0] for r in cur.execute("SELECT entity_id FROM source1 ORDER BY entity_id")]
    if limit_s1:
        s1_ids = s1_ids[:limit_s1]
    total = len(s1_ids)

    score_expr = _score_expr()
    query = f"""
        SELECT a.entity_id AS s1_id, b.entity_id AS cand_id, b.source AS cand_source,
               {score_expr} AS score
        FROM blocking_keys a
        JOIN _batch_ids t ON t.entity_id = a.entity_id
        JOIN blocking_keys b
          ON b.key_type = a.key_type
         AND b.key_value = a.key_value
         AND b.country = a.country
         AND b.source IN (2, 3)
        WHERE a.source = 1
        GROUP BY a.entity_id, b.entity_id, b.source
    """

    t0 = time.time()
    n_done = 0
    n_batches = 0
    for start in range(0, total, batch_size):
        batch = s1_ids[start:start + batch_size]
        conn.execute("DELETE FROM _batch_ids;")
        conn.executemany("INSERT INTO _batch_ids VALUES (?)", [(x,) for x in batch])

        results = {s1: [] for s1 in batch}
        for s1_id, cand_id, cand_source, score in cur.execute(query):
            if score >= min_score:
                results[s1_id].append((cand_id, cand_source, score))

        for s1_id in batch:
            cands = results[s1_id]
            cands.sort(key=lambda x: -x[2])
            yield s1_id, cands[:top_k]

        n_done += len(batch)
        n_batches += 1
        if progress_every and n_batches % progress_every == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (total - n_done) / rate if rate > 0 else float("inf")
            print(f"  blocking: {n_done}/{total} S1 entities ({rate:.0f}/s, ETA {eta/60:.1f} min)", flush=True)

    conn.execute("DROP TABLE IF EXISTS _batch_ids;")


def write_candidate_pairs_tsv(conn, out_path, **kwargs):
    n_with_cands = 0
    n_total_cands = 0
    n_entities = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, cands in generate_candidates(conn, **kwargs):
            ids = ",".join(c[0] for c in cands)
            f.write(f"{s1_id}\t{ids}\n")
            n_entities += 1
            if cands:
                n_with_cands += 1
                n_total_cands += len(cands)
    print(
        f"  candidate_pairs written: {n_entities} entities, {n_with_cands} with >=1 candidate, "
        f"avg candidates (of those with any)={n_total_cands / max(n_with_cands, 1):.2f}",
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-block-size", type=int, default=DEFAULT_MAX_BLOCK_SIZE)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--min-score", type=int, default=DEFAULT_MIN_SCORE)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--limit-s1", type=int, default=None)
    ap.add_argument("--skip-purge", action="store_true")
    args = ap.parse_args()

    conn = db_connect(args.db)
    if not args.skip_purge:
        purge_common_keys(conn, args.max_block_size, DEFAULT_MAX_BLOCK_SIZE_OVERRIDES)
    write_candidate_pairs_tsv(
        conn,
        args.out,
        top_k=args.top_k,
        min_score=args.min_score,
        batch_size=args.batch_size,
        limit_s1=args.limit_s1,
    )
    conn.close()
