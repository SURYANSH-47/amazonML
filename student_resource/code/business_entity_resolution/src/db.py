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
    name_phonetic,
    name_prefix,
    name_sorted_key,
    name_tokens,
    normalize_address,
    normalize_name,
    phonetic_tokens,
    postal_tokens,
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
    """Emit every blocking key for a record.

    Deliberately generous: keys are never deleted downstream (see blocking.py),
    they are only weighted by how rare they are, so an extra key type can add
    recall without the old risk of flooding the candidate set.
    """
    keys = []
    add = keys.append
    for tok in name_tokens(name_norm):
        add((source_num, entity_id, country, "name_token", tok))
    for tok in addr_tokens(addr_norm):
        add((source_num, entity_id, country, "addr_token", tok))
    for d in digit_tokens(raw_address):
        add((source_num, entity_id, country, "digit_token", d))
    # Postal codes are far more discriminative than generic house numbers.
    for p in postal_tokens(raw_address):
        add((source_num, entity_id, country, "postal_token", p))
    prefix = name_prefix(name_norm)
    if len(prefix) >= 3:
        add((source_num, entity_id, country, "name_prefix", prefix))
    if name_norm:
        # Exact whole-name match: the single strongest lexical signal.
        add((source_num, entity_id, country, "name_full", name_norm))
        # Order-independent name: survives word-order transposition.
        sorted_key = name_sorted_key(name_norm)
        if sorted_key and sorted_key != name_norm:
            add((source_num, entity_id, country, "name_sorted", sorted_key))
        # Phonetic skeletons: survive vowel typos and transliteration drift,
        # which is how many Source-2/3 India records differ from Source 1.
        phon = name_phonetic(name_norm)
        if len(phon) >= 3:
            add((source_num, entity_id, country, "name_phon", phon))
        for pt in phonetic_tokens(name_norm):
            add((source_num, entity_id, country, "phon_token", pt))
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

    build_indexes(conn)

    # Document frequencies for IDF weighting. Nothing is deleted here — keys
    # are weighted by rarity at query time instead of being purged, which is
    # what protects candidate-set recall (see blocking.py).
    import blocking as blocking_mod
    blocking_mod.build_key_stats(conn)

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
