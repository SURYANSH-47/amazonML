"""Run blocking + the trained matcher over a (test or train) DB, streaming
batches of Source-1 entities end to end so peak memory stays bounded
regardless of how many Source-1 entities are in the DB.

Writes both output files in a single pass:
  - candidate_pairs.tsv: every candidate the blocking stage produced (the
    exact set fed to the model)
  - matching_results.tsv: candidates whose model probability clears the
    tuned threshold, capped at the tuned top_n if one was chosen
"""

import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import blocking
import features as feat_mod

import lightgbm as lgb
import numpy as np

ENTITY_BATCH = 2000


def load_model(models_dir):
    booster = lgb.Booster(model_file=os.path.join(models_dir, "model.txt"))
    with open(os.path.join(models_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    return booster, config


def run(db_path, models_dir, candidate_out, matching_out,
        max_block_size=None, skip_purge=False, limit_s1=None):
    booster, config = load_model(models_dir)
    top_k = config["blocking_top_k"]
    min_score = config["blocking_min_score"]
    mbs = max_block_size if max_block_size is not None else config["blocking_max_block_size"]
    threshold = config["threshold"]
    top_n = config["top_n"]

    from db import connect as db_connect

    conn = db_connect(db_path)

    if not skip_purge:
        blocking.purge_common_keys(conn, mbs, blocking.DEFAULT_MAX_BLOCK_SIZE_OVERRIDES)

    cand_f = open(candidate_out, "w", encoding="utf-8", newline="")
    match_f = open(matching_out, "w", encoding="utf-8", newline="")
    cand_f.write("source1_entity_id\tcandidate_entity_ids\n")
    match_f.write("source1_entity_id\tmatched_entity_ids\n")

    batch_results = []
    n_entities = 0
    n_with_match = 0
    t0 = time.time()

    def flush():
        nonlocal n_with_match
        if not batch_results:
            return
        entity_order = [s1 for s1, _ in batch_results]
        # Entities with zero candidates: write empty rows directly.
        nonempty = [(s1, c) for s1, c in batch_results if c]
        empty_ids = {s1 for s1, c in batch_results if not c}

        probs_by_pair = {}
        if nonempty:
            X, s1_out, cand_out = feat_mod.featurize_batch(conn, nonempty)
            probs = booster.predict(np.asarray(X, dtype=np.float32))
            for s1_id, cid, p in zip(s1_out, cand_out, probs):
                probs_by_pair.setdefault(s1_id, []).append((cid, p))

        for s1 in entity_order:
            if s1 in empty_ids:
                cand_ids = []
                matched_ids = []
            else:
                pairs = probs_by_pair.get(s1, [])
                pairs.sort(key=lambda x: -x[1])
                cand_ids = [cid for cid, _ in pairs]
                kept = [cid for cid, p in pairs if p >= threshold]
                if top_n is not None:
                    kept = kept[:top_n]
                matched_ids = kept
                if matched_ids:
                    n_with_match += 1
            cand_f.write(f"{s1}\t{','.join(cand_ids)}\n")
            match_f.write(f"{s1}\t{','.join(matched_ids)}\n")

    for s1_id, cands in blocking.generate_candidates(
        conn, top_k=top_k, min_score=min_score, batch_size=blocking.DEFAULT_BATCH_SIZE,
        limit_s1=limit_s1, progress_every=0,
    ):
        batch_results.append((s1_id, cands))
        n_entities += 1
        if len(batch_results) >= ENTITY_BATCH:
            flush()
            batch_results = []
            elapsed = time.time() - t0
            rate = n_entities / elapsed if elapsed > 0 else 0
            print(f"  predict: {n_entities} entities done ({rate:.0f}/s, {n_with_match} with >=1 match)", flush=True)
    flush()

    cand_f.close()
    match_f.close()
    conn.close()
    print(f"  DONE: {n_entities} entities, {n_with_match} with >=1 predicted match", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--candidate-out", required=True)
    ap.add_argument("--matching-out", required=True)
    ap.add_argument("--limit-s1", type=int, default=None)
    ap.add_argument("--skip-purge", action="store_true")
    args = ap.parse_args()
    run(args.db, args.models_dir, args.candidate_out, args.matching_out,
        skip_purge=args.skip_purge, limit_s1=args.limit_s1)
