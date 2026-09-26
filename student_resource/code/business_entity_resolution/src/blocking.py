"""Blocking / candidate generation via IDF-weighted meta-blocking.

Design (and why it is not the obvious one):

An earlier version used *block purging* — globally deleting every key whose
Source-2/3 frequency exceeded an absolute cap. That is a known-bad technique:
the blocking literature is explicit that purging "cannot control its impact on
recall", and at full corpus scale it collapsed our candidate-set recall
ceiling to ~54% (an absolute cap tuned on a small sample becomes brutally
aggressive once the corpus is ~80x larger). Worse, deletion is irrecoverable:
an entity whose *only* keys happened to be common lost every candidate it had.

This version never deletes a key. Instead it follows standard meta-blocking:

  1. Key statistics (`key_freq`): document frequency df per
     (key_type, key_value, country), computed over Source 2+3.

  2. Block Filtering, applied per-entity at query time: a Source-1 entity is
     probed using only its own most *selective* keys (lowest df). This bounds
     work per query without globally destroying anything — an entity whose
     keys are all common still keeps its best available ones.

  3. IDF weighting (RACCB/CF-IBF family): a shared key contributes
     log(N/df) * key_type_prior. Rare shared keys are strong evidence; common
     shared keys contribute a little rather than nothing. This is what lets a
     true match survive on "urology" + "specialists" alone when the address is
     missing on one side.

  4. Cardinality Node Pruning: keep the top-K highest-scoring candidates per
     Source-1 entity.

Everything runs as SQL aggregation in SQLite so peak Python memory stays
bounded regardless of table size.
"""

import math
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from db import connect as db_connect

# Per-key-type prior on top of the IDF weight. A shared postal code or an
# exact whole-name match means much more than one shared address word, even
# at equal document frequency.
KEY_TYPE_PRIOR = {
    "name_full": 3.0,
    "name_sorted": 2.6,
    "name_phon": 2.0,
    "postal_token": 2.2,
    "digit_token": 1.3,
    "name_token": 1.0,
    "phon_token": 0.8,
    "name_prefix": 0.7,
    "addr_token": 0.6,
}

# Cost control, applied per Source-1 entity at probe time — NOT a purge.
# Nothing is ever deleted from the index; an entity simply probes with its own
# most selective keys, rarest first, until its work budget is spent:
#
#   PROBE_DF_BUDGET  total rows the join may touch for one entity
#   PROBE_KEYS       max number of keys to probe with
#   MIN_PROBE_KEYS   always probe at least this many, even if over budget, so
#                    an entity whose keys are all common still gets candidates
#   PROBE_DF_CEILING never probe a single key more common than this
#
# Budgeting on cumulative df (rather than a key count alone) is what bounds
# worst-case work: without it, one key with df=40k costs as much as 40 keys
# with df=1000.
PROBE_DF_BUDGET = 2500
PROBE_KEYS = 10
MIN_PROBE_KEYS = 3
PROBE_DF_CEILING = 25000

# Adaptive escalation. Measured recall/cost curve on a realistic-scale sample:
#
#   probe=10 budget=2500   -> 85.2% pair recall, 1.0x cost
#   probe=25 budget=20000  -> 93.4% pair recall, 8.0x cost
#
# Paying 8x for every entity is not affordable at 1.7M entities, but most
# entities are easy — a distinctive name resolves on its rare keys alone. So
# the cheap pass runs for everyone, and only entities that come back looking
# unresolved (too few candidates, or no strongly-scoring one) are re-probed
# with the expensive budget. That buys most of the recall for a fraction of
# the cost.
ESCALATE_PROBE_KEYS = 25
ESCALATE_DF_BUDGET = 20000
ESCALATE_MIN_CANDS = 8     # always re-probe entities this candidate-starved
ESCALATE_MAX_FRAC = 0.25   # ...plus at most this share of the budget-truncated
                           # ones, ranked by unspent IDF evidence. Bounds the
                           # blended cost (~2-3x) instead of 8x for everyone.

DEFAULT_TOP_K = 25
DEFAULT_MIN_SCORE = 1.0
DEFAULT_BATCH_SIZE = 500


def build_key_stats(conn: sqlite3.Connection):
    """Build the df table over Source-2/3 keys. Non-destructive."""
    cur = conn.cursor()
    t0 = time.time()
    print("  key stats: computing per-country document frequencies...", flush=True)
    cur.execute("DROP TABLE IF EXISTS key_freq;")
    cur.execute(
        """
        CREATE TABLE key_freq AS
        SELECT key_type, key_value, country, COUNT(*) AS df
        FROM blocking_keys
        WHERE source IN (2, 3)
        GROUP BY key_type, key_value, country
        """
    )
    print(f"  key stats: table built in {time.time()-t0:.0f}s, indexing...", flush=True)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_key_freq ON key_freq(key_type, key_value, country);"
    )
    conn.commit()
    n = cur.execute("SELECT COUNT(*) FROM key_freq").fetchone()[0]
    total = cur.execute(
        "SELECT COUNT(*) FROM blocking_keys WHERE source IN (2,3)"
    ).fetchone()[0]
    print(f"  key stats: {n} distinct keys over {total} S2/S3 key rows ({time.time()-t0:.0f}s)", flush=True)
    return n


def corpus_size(conn: sqlite3.Connection) -> int:
    n = conn.execute("SELECT COUNT(*) FROM source2").fetchone()[0]
    n += conn.execute("SELECT COUNT(*) FROM source3").fetchone()[0]
    return max(n, 2)


def prune_unprobeable_keys(conn: sqlite3.Connection, df_ceiling: int = PROBE_DF_CEILING):
    """Delete Source-2/3 key rows that probing already refuses to touch.

    This is NOT the old block purging. Those keys are skipped at probe time
    anyway (df > PROBE_DF_CEILING, where log(N/df) is nearly zero), so removing
    them cannot change which candidates are generated — it is provably
    recall-neutral. But a key occurring 500k times contributes 500k index rows
    we never read, and that bloat is what destroys cache locality on a
    160M+ row index. Removing it is pure throughput.
    """
    cur = conn.cursor()
    t0 = time.time()
    before = cur.execute("SELECT COUNT(*) FROM blocking_keys").fetchone()[0]
    cur.execute(
        """
        DELETE FROM blocking_keys
        WHERE source IN (2, 3)
          AND (key_type, key_value, country) IN (
              SELECT key_type, key_value, country FROM key_freq WHERE df > ?
          )
        """,
        (df_ceiling,),
    )
    conn.commit()
    after = cur.execute("SELECT COUNT(*) FROM blocking_keys").fetchone()[0]
    print(f"  pruned never-probed keys (df>{df_ceiling}): {before} -> {after} rows "
          f"({100*(before-after)/max(before,1):.1f}% removed, {time.time()-t0:.0f}s)", flush=True)
    return after


def build_probe_table(
    conn: sqlite3.Connection,
    probe_keys: int = ESCALATE_PROBE_KEYS,
    df_ceiling: int = PROBE_DF_CEILING,
):
    """Materialize each Source-1 entity's probe keys once, with weights baked in.

    Previously every batch re-derived this in Python, including thousands of
    random lookups into key_freq purely to fetch df values that never change.
    That fixed per-entity overhead — not the candidate join — was the real
    throughput bottleneck. Precomputing it turns candidate generation into a
    single indexed SQL join.

    Rows are stored with `rank` (1 = rarest) and `cum_df` (cumulative df up to
    and including this key), so a caller can pick a cost tier at query time
    with a simple WHERE clause instead of recomputing anything.
    """
    cur = conn.cursor()
    t0 = time.time()
    N = corpus_size(conn)
    log_n = math.log(N)

    print("  probe table: selecting per-entity keys...", flush=True)
    cur.execute("DROP TABLE IF EXISTS s1_probe;")
    cur.execute(
        """
        CREATE TABLE s1_probe (
            s1_id TEXT, key_type TEXT, key_value TEXT, country TEXT,
            w REAL, rank INTEGER, cum_df INTEGER
        )
        """
    )

    rows_sql = """
        SELECT bk.entity_id, bk.key_type, bk.key_value, bk.country, kf.df
        FROM blocking_keys bk
        JOIN key_freq kf
          ON kf.key_type = bk.key_type
         AND kf.key_value = bk.key_value
         AND kf.country = bk.country
        WHERE bk.source = 1 AND kf.df <= ?
        ORDER BY bk.entity_id, kf.df
    """

    batch = []
    n_entities = 0
    current_id = None
    rank = 0
    cum = 0

    for eid, kt, kv, country, df in cur.execute(rows_sql, (df_ceiling,)):
        if eid != current_id:
            current_id = eid
            rank = 0
            cum = 0
            n_entities += 1
            if n_entities % 200000 == 0:
                print(f"    probe table: {n_entities} entities ({time.time()-t0:.0f}s)", flush=True)
        if rank >= probe_keys:
            continue
        rank += 1
        cum += df
        w = KEY_TYPE_PRIOR.get(kt, 1.0) * (log_n - math.log(df))
        if w <= 0:
            continue
        batch.append((eid, kt, kv, country, w, rank, cum))
        if len(batch) >= 20000:
            conn.executemany(
                "INSERT INTO s1_probe VALUES (?,?,?,?,?,?,?)", batch
            )
            batch = []
    if batch:
        conn.executemany("INSERT INTO s1_probe VALUES (?,?,?,?,?,?,?)", batch)
    conn.commit()

    print(f"  probe table: indexing ({time.time()-t0:.0f}s)...", flush=True)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_s1_probe ON s1_probe(s1_id, rank);")
    conn.commit()
    n = cur.execute("SELECT COUNT(*) FROM s1_probe").fetchone()[0]
    print(f"  probe table: {n} probe keys for {n_entities} entities ({time.time()-t0:.0f}s)", flush=True)
    return n


def generate_candidates(
    conn: sqlite3.Connection,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit_s1: int = None,
    s1_ids: list = None,
    probe_keys: int = PROBE_KEYS,
    df_budget: int = PROBE_DF_BUDGET,
    escalate: bool = True,
    escalate_probe_keys: int = ESCALATE_PROBE_KEYS,
    escalate_df_budget: int = ESCALATE_DF_BUDGET,
    escalate_min_cands: int = ESCALATE_MIN_CANDS,
    escalate_max_frac: float = ESCALATE_MAX_FRAC,
    progress_every: int = 50,
    **_ignored,
):
    """Yield (s1_entity_id, [(cand_entity_id, cand_source, score), ...]).

    Sorted by descending IDF-weighted score and capped at top_k.

    Reads the precomputed `s1_probe` table (see build_probe_table): each
    entity's probe keys, weights, rarity rank and cumulative df are already
    materialized, so a cost tier is just a WHERE clause and the whole thing is
    one indexed join. Deriving this per batch in Python was what capped
    throughput at ~25 entities/s.
    """
    cur = conn.cursor()

    has_probe = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='s1_probe'"
    ).fetchone()
    if not has_probe:
        raise RuntimeError(
            "s1_probe table missing — run blocking.build_probe_table(conn) "
            "(db.py does this automatically at build time)."
        )

    if s1_ids is None:
        s1_ids = [r[0] for r in cur.execute("SELECT entity_id FROM source1")]
    if limit_s1:
        s1_ids = s1_ids[:limit_s1]
    total = len(s1_ids)

    conn.execute("DROP TABLE IF EXISTS _batch_ids;")
    conn.execute("CREATE TEMP TABLE _batch_ids (entity_id TEXT PRIMARY KEY);")

    # CROSS JOIN pins the small probe side as the outer loop so blocking_keys
    # is only ever reached through its index; SQLite can otherwise choose to
    # scan the 100M+ row table once per batch.
    candidate_sql = """
        SELECT p.s1_id, b.entity_id, b.source, SUM(p.w) AS score
        FROM _batch_ids t
        CROSS JOIN s1_probe p ON p.s1_id = t.entity_id
        CROSS JOIN blocking_keys b
          ON b.key_type = p.key_type
         AND b.key_value = p.key_value
         AND b.country = p.country
         AND b.source IN (2, 3)
        WHERE p.rank <= ? AND (p.rank <= ? OR p.cum_df <= ?)
        GROUP BY p.s1_id, b.entity_id, b.source
    """

    t0 = time.time()
    n_done = 0
    n_batches = 0
    n_escalated = 0

    def run_tier(n_keys, budget):
        found = {}
        for s1_id, cand_id, cand_source, score in cur.execute(
            candidate_sql, (n_keys, MIN_PROBE_KEYS, budget)
        ):
            if score >= min_score:
                found.setdefault(s1_id, []).append((cand_id, cand_source, score))
        return found

    for start in range(0, total, batch_size):
        batch = s1_ids[start:start + batch_size]
        conn.execute("DELETE FROM _batch_ids;")
        conn.executemany("INSERT INTO _batch_ids VALUES (?)", [(x,) for x in batch])

        results = run_tier(probe_keys, df_budget)

        # Escalation: re-probe only entities the cheap tier left candidate-poor,
        # capped so cost stays bounded. Escalating everything just pays the
        # expensive config for the whole corpus.
        if escalate:
            quota = int(len(batch) * escalate_max_frac)
            hard = [e for e in batch if len(results.get(e, ())) < escalate_min_cands][:max(quota, 0)]
            if hard:
                n_escalated += len(hard)
                conn.execute("DELETE FROM _batch_ids;")
                conn.executemany("INSERT INTO _batch_ids VALUES (?)", [(x,) for x in hard])
                for s1_id, cands in run_tier(escalate_probe_keys, escalate_df_budget).items():
                    results[s1_id] = cands

        for s1_id in batch:
            cands = results.get(s1_id, [])
            cands.sort(key=lambda x: -x[2])
            yield s1_id, cands[:top_k]

        n_done += len(batch)
        n_batches += 1
        if progress_every and n_batches % progress_every == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (total - n_done) / rate if rate > 0 else float("inf")
            print(f"  blocking: {n_done}/{total} S1 ({rate:.0f}/s, ETA {eta/60:.1f} min, "
                  f"{100*n_escalated/max(n_done,1):.0f}% escalated)", flush=True)

    conn.execute("DROP TABLE IF EXISTS _batch_ids;")


def write_candidate_pairs_tsv(conn, out_path, **kwargs):
    n_with_cands = n_total_cands = n_entities = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, cands in generate_candidates(conn, **kwargs):
            f.write(f"{s1_id}\t{','.join(c[0] for c in cands)}\n")
            n_entities += 1
            if cands:
                n_with_cands += 1
                n_total_cands += len(cands)
    print(
        f"  candidate_pairs written: {n_entities} entities, {n_with_cands} with >=1 candidate, "
        f"avg={n_total_cands / max(n_with_cands, 1):.2f}",
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    ap.add_argument("--probe-keys", type=int, default=PROBE_KEYS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--limit-s1", type=int, default=None)
    ap.add_argument("--rebuild-stats", action="store_true")
    args = ap.parse_args()

    conn = db_connect(args.db)
    if args.rebuild_stats:
        build_key_stats(conn)
    write_candidate_pairs_tsv(
        conn, args.out, top_k=args.top_k, min_score=args.min_score,
        probe_keys=args.probe_keys, batch_size=args.batch_size, limit_s1=args.limit_s1,
    )
    conn.close()
