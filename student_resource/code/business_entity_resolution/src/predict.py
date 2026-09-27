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


def run(db_path, models_dir, candidate_out, matching_out, limit_s1=None,
        shard=0, num_shards=1, scores_out=None, score_floor=0.05, select="all",
        s1_list=None):
    """scores_out: if set, also write every scored pair (s1, cand, prob) with
    prob >= score_floor. This moves all final decisions (thresholds, one-owner
    enforcement across entities) to assign.py, which runs in minutes — so the
    decision layer can be retuned without repeating this multi-hour pass.

    select="val": restrict to held-out validation-bucket entities (never seen
    in training), for producing labeled score dumps on train.db.
    """
    booster, config = load_model(models_dir)
    top_k = config["blocking_top_k"]
    min_score = config["blocking_min_score"]
    probe_keys = config.get("blocking_probe_keys", blocking.PROBE_KEYS)
    threshold = config["threshold"]
    top_n = config["top_n"]

    from db import connect as db_connect

    conn = db_connect(db_path)

    cand_f = open(candidate_out, "w", encoding="utf-8", newline="")
    match_f = open(matching_out, "w", encoding="utf-8", newline="")
    cand_f.write("source1_entity_id\tcandidate_entity_ids\n")
    match_f.write("source1_entity_id\tmatched_entity_ids\n")
    score_f = None
    if scores_out:
        score_f = open(scores_out, "w", encoding="utf-8", newline="")
        score_f.write("source1_entity_id\tcandidate_entity_id\tprob\n")

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
                if score_f is not None and p >= score_floor:
                    score_f.write(f"{s1_id}\t{cid}\t{p:.5f}\n")

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

    # Sharding: each worker takes a disjoint slice of the Source-1 entities.
    # The work is embarrassingly parallel (entities are independent) and the
    # bottleneck is random disk I/O on one SQLite reader, so running N workers
    # over the same read-only DB scales close to linearly.
    shard_ids = None
    if num_shards > 1 or select != "all" or s1_list:
        if s1_list:
            with open(s1_list, encoding="utf-8") as f:
                next(f)
                all_ids = [ln.strip() for ln in f if ln.strip()]
            print(f"  s1-list: {len(all_ids)} entities from {s1_list}", flush=True)
        else:
            all_ids = [r[0] for r in conn.execute("SELECT entity_id FROM source1")]
        if select == "val":
            from train import stable_bucket, VAL_FRACTION_HASH
            all_ids = [e for e in all_ids if stable_bucket(e) < VAL_FRACTION_HASH]
            if limit_s1:
                all_ids = all_ids[:limit_s1]
            limit_s1 = None
            print(f"  select=val: {len(all_ids)} held-out entities", flush=True)
        shard_ids = [e for i, e in enumerate(all_ids) if i % num_shards == shard]
        print(f"  shard {shard}/{num_shards}: {len(shard_ids)} of {len(all_ids)} entities", flush=True)

    for s1_id, cands in blocking.generate_candidates(
        conn, top_k=top_k, min_score=min_score, probe_keys=probe_keys,
        batch_size=blocking.DEFAULT_BATCH_SIZE, limit_s1=limit_s1,
        s1_ids=shard_ids, progress_every=0,
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
    if score_f is not None:
        score_f.close()
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
    ap.add_argument("--shard", type=int, default=0,
                    help="This worker's index, 0-based (use with --num-shards).")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Run N workers in parallel over disjoint entity slices, "
                         "then merge with merge_shards.py.")
    ap.add_argument("--scores-out", default=None,
                    help="Also dump every scored pair (s1, cand, prob) for assign.py.")
    ap.add_argument("--score-floor", type=float, default=0.05,
                    help="Only dump pairs with prob >= this (keeps the file small).")
    ap.add_argument("--select", choices=["all", "val"], default="all",
                    help="'val' = only held-out validation-bucket entities (train.db).")
    ap.add_argument("--s1-list", default=None,
                    help="Only process the Source-1 ids in this file (one per line, "
                         "with a header), e.g. the output of conflicts.py --write-entities.")
    args = ap.parse_args()
    run(args.db, args.models_dir, args.candidate_out, args.matching_out,
        limit_s1=args.limit_s1, shard=args.shard, num_shards=args.num_shards,
        scores_out=args.scores_out, score_floor=args.score_floor, select=args.select,
        s1_list=args.s1_list)
