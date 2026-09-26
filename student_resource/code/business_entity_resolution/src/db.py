"""Stream source TSVs into a SQLite DB with normalized fields + blocking keys.

Never loads a full source file into memory: rows are read one at a time with
csv.reader and inserted in batches, which keeps peak memory bounded regardless
of file size (needed on this machine's ~4GB RAM).
"""

import csv
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(__file__))
from normalize import (
    addr_tokens,
    digit_tokens,
    name_prefix,
    name_tokens,
    normalize_address,
    normalize_name,
)

BATCH_SIZE = 5000

SCHEMA = """
CREATE TABLE IF NOT EXISTS source1 (
    entity_id TEXT PRIMARY KEY,
    business_name TEXT,
    business_address TEXT,
    country TEXT,
    name_norm TEXT,
    addr_norm TEXT
);
CREATE TABLE IF NOT EXISTS source2 (
    entity_id TEXT PRIMARY KEY,
    business_name TEXT,
    business_address TEXT,
    country TEXT,
    name_norm TEXT,
    addr_norm TEXT
);
CREATE TABLE IF NOT EXISTS source3 (
    entity_id TEXT PRIMARY KEY,
    business_name TEXT,
    business_address TEXT,
    country TEXT,
    name_norm TEXT,
    addr_norm TEXT
);
CREATE TABLE IF NOT EXISTS blocking_keys (
    source INTEGER NOT NULL,
    entity_id TEXT NOT NULL,
    country TEXT,
    key_type TEXT NOT NULL,
    key_value TEXT NOT NULL
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    # FILE (not MEMORY): CREATE INDEX / GROUP BY on 100M+ row tables need a
    # multi-GB external sort. Spilling that to disk (plenty free) instead of
    # RAM (only ~4GB total on this machine) is what keeps this from OOMing.
    conn.execute("PRAGMA temp_store=FILE;")
    conn.execute("PRAGMA cache_size=-65536;")  # ~64MB page cache, bounded
    return conn


def _row_keys(source_num, entity_id, country, name_norm, addr_norm, raw_address):
    keys = []
    for tok in name_tokens(name_norm):
        keys.append((source_num, entity_id, country, "name_token", tok))
    for tok in addr_tokens(addr_norm):
        keys.append((source_num, entity_id, country, "addr_token", tok))
    for d in digit_tokens(raw_address):
        keys.append((source_num, entity_id, country, "digit_token", d))
    prefix = name_prefix(name_norm)
    if len(prefix) >= 3:
        keys.append((source_num, entity_id, country, "name_prefix", prefix))
    # Exact-normalized-name key: strong signal even when tokens individually
    # are common (e.g. "urology specialists") and address is missing/sparse.
    # Purged with a much higher cap than the per-token keys (see blocking.py).
    if name_norm:
        keys.append((source_num, entity_id, country, "name_full", name_norm))
    return keys


def load_source(conn, tsv_path, source_num, progress_every=200000):
    table = f"source{source_num}"
    cur = conn.cursor()
    row_batch = []
    key_batch = []
    n = 0
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        assert header[:4] == ["entity_id", "business_name", "business_address", "country"], header
        for row in reader:
            if len(row) < 4:
                row = row + [""] * (4 - len(row))
            entity_id, business_name, business_address, country = row[0], row[1], row[2], row[3]
            name_norm = normalize_name(business_name)
            addr_norm = normalize_address(business_address)
            row_batch.append((entity_id, business_name, business_address, country, name_norm, addr_norm))
            key_batch.extend(_row_keys(source_num, entity_id, country, name_norm, addr_norm, business_address))
            n += 1
            if len(row_batch) >= BATCH_SIZE:
                _flush(cur, table, row_batch, key_batch)
                row_batch, key_batch = [], []
            if progress_every and n % progress_every == 0:
                print(f"  [{table}] {n} rows loaded", flush=True)
        if row_batch:
            _flush(cur, table, row_batch, key_batch)
    conn.commit()
    print(f"  [{table}] done: {n} rows", flush=True)
    return n


def _flush(cur, table, row_batch, key_batch):
    cur.executemany(
        f"INSERT OR REPLACE INTO {table} "
        f"(entity_id, business_name, business_address, country, name_norm, addr_norm) "
        f"VALUES (?,?,?,?,?,?)",
        row_batch,
    )
    if key_batch:
        cur.executemany(
            "INSERT INTO blocking_keys (source, entity_id, country, key_type, key_value) "
            "VALUES (?,?,?,?,?)",
            key_batch,
        )


def build_indexes(conn):
    import time
    cur = conn.cursor()
    # country before source: every blocking join filters key_type+key_value+country
    # together (see blocking.py), so this lets SQLite seek straight to the
    # exact country-scoped block instead of scanning every country's rows for
    # that key and filtering afterward. Matters much more now that blocks can
    # be up to 2000-6000 rows (vs. 200-2000 before the purge fix).
    t0 = time.time()
    print("  building index idx_bk_lookup...", flush=True)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bk_lookup ON blocking_keys(key_type, key_value, country, source);")
    print(f"  idx_bk_lookup done in {time.time()-t0:.0f}s, building idx_bk_entity...", flush=True)
    t1 = time.time()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bk_entity ON blocking_keys(source, entity_id);")
    print(f"  idx_bk_entity done in {time.time()-t1:.0f}s", flush=True)
    conn.commit()


def build_db(dataset_dir: str, db_path: str, split: str):
    """split: 'train' or 'test'. dataset_dir is the dataset/ folder."""
    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in ("-wal", "-shm"):
        p = db_path + ext
        if os.path.exists(p):
            os.remove(p)

    conn = connect(db_path)
    conn.executescript(SCHEMA)

    split_dir = os.path.join(dataset_dir, split)
    counts = {}
    counts[1] = load_source(conn, os.path.join(split_dir, f"{split}_source1.tsv"), 1)
    counts[2] = load_source(conn, os.path.join(split_dir, f"{split}_source2.tsv"), 2)
    counts[3] = load_source(conn, os.path.join(split_dir, f"{split}_source3.tsv"), 3)

    # Purge overly common blocking keys BEFORE indexing: the raw blocking_keys
    # table can be 100M+ rows, so indexing it unpurged is a much larger (and,
    # on this machine, RAM-riskier) sort than indexing the purged table.
    import blocking as blocking_mod
    blocking_mod.purge_common_keys(conn, blocking_mod.DEFAULT_MAX_BLOCK_SIZE, blocking_mod.DEFAULT_MAX_BLOCK_SIZE_OVERRIDES)

    build_indexes(conn)
    conn.close()
    return counts


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="dataset")
    ap.add_argument("--db", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    args = ap.parse_args()
    counts = build_db(args.dataset_dir, args.db, args.split)
    print("counts:", counts)
