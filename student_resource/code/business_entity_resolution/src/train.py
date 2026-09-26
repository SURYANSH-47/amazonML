"""Build labeled candidate pairs from blocking + ground truth, train a LightGBM
matcher, and tune the decision threshold (+ optional per-entity top-N cap) on
a held-out validation split of Source-1 entities to maximize macro F_0.5.

Entity-level split (not pair-level) so there's no leakage between train-fit
and validation. Both the training-pair set and the validation set are built
by subsampling Source-1 entities (not the full 2.2M) — plenty for a ~15
feature tabular model, and keeps runtime/memory bounded on this machine.
"""

import csv
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import blocking
import features as feat_mod
from evaluate import macro_f_beta

import lightgbm as lgb
import numpy as np

SEED = 42
VAL_FRACTION_HASH = 15  # entities with hash%100 < this go to validation
TRAIN_FIT_SAMPLE_SIZE = 300_000
VAL_SAMPLE_SIZE = 150_000
ENTITY_BATCH = 2000


def stable_bucket(entity_id: str) -> int:
    h = hashlib.md5(entity_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100


def split_entities(conn, fit_size=TRAIN_FIT_SAMPLE_SIZE, val_size=VAL_SAMPLE_SIZE):
    all_ids = [r[0] for r in conn.execute("SELECT entity_id FROM source1")]
    val_ids = [i for i in all_ids if stable_bucket(i) < VAL_FRACTION_HASH]
    fit_ids = [i for i in all_ids if stable_bucket(i) >= VAL_FRACTION_HASH]
    rng = random.Random(SEED)
    rng.shuffle(fit_ids)
    rng.shuffle(val_ids)
    fit_sample = fit_ids[:fit_size]
    val_sample = val_ids[:val_size]
    print(f"  total S1 entities: {len(all_ids)}, fit_sample: {len(fit_sample)}, val_sample: {len(val_sample)}", flush=True)
    return fit_sample, val_sample


def load_ground_truth_subset(gt_path, needed_ids: set):
    gt = {}
    with open(gt_path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row and row[0] in needed_ids:
                ids = row[1].split(",") if len(row) > 1 and row[1] else []
                gt[row[0]] = set(ids)
    return gt


def build_pairs(conn, s1_ids, gt, top_k, min_score, batch_size, label=True,
                probe_keys=None):
    """Stream blocking candidates for s1_ids, featurize in entity-batches.

    Returns X (list[list[float]]), y (list[int] or None), s1_out, cand_out.
    """
    X, y, s1_out, cand_out = [], ([] if label else None), [], []
    batch_results = []
    t0 = time.time()
    n_entities = 0

    def flush():
        if not batch_results:
            return
        Xb, s1b, cb = feat_mod.featurize_batch(conn, batch_results)
        X.extend(Xb)
        s1_out.extend(s1b)
        cand_out.extend(cb)
        if label:
            for s1_id, cid in zip(s1b, cb):
                y.append(1 if cid in gt.get(s1_id, ()) else 0)

    for s1_id, cands in blocking.generate_candidates(
        conn, top_k=top_k, min_score=min_score, batch_size=batch_size,
        s1_ids=s1_ids, probe_keys=probe_keys or blocking.PROBE_KEYS,
        progress_every=0,
    ):
        if cands:
            batch_results.append((s1_id, cands))
        n_entities += 1
        if len(batch_results) >= ENTITY_BATCH:
            flush()
            batch_results = []
            elapsed = time.time() - t0
            print(f"    featurized {n_entities}/{len(s1_ids)} entities, {len(X)} pairs so far ({elapsed:.0f}s)", flush=True)
    flush()
    return X, y, s1_out, cand_out


def train_model(X, y, dev_frac=0.1):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X))
    n_dev = int(len(X) * dev_frac)
    dev_idx, fit_idx = idx[:n_dev], idx[n_dev:]

    fit_set = lgb.Dataset(X[fit_idx], label=y[fit_idx], feature_name=feat_mod.FEATURE_NAMES)
    dev_set = lgb.Dataset(X[dev_idx], label=y[dev_idx], feature_name=feat_mod.FEATURE_NAMES, reference=fit_set)

    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": SEED,
    }
    booster = lgb.train(
        params,
        fit_set,
        num_boost_round=500,
        valid_sets=[dev_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )
    return booster


def group_by_entity(s1_ids, cand_ids, probs):
    by_entity = {}
    for s1, cid, p in zip(s1_ids, cand_ids, probs):
        by_entity.setdefault(s1, []).append((cid, p))
    for v in by_entity.values():
        v.sort(key=lambda x: -x[1])
    return by_entity


def tune_threshold(by_entity, val_ids, gt, thresholds, top_ns):
    best = None
    for thr in thresholds:
        for top_n in top_ns:
            preds = {}
            for s1 in val_ids:
                cands = by_entity.get(s1, [])
                kept = [cid for cid, p in cands if p >= thr]
                if top_n is not None:
                    kept = kept[:top_n]
                preds[s1] = set(kept)
            result = macro_f_beta(val_ids, gt, preds)
            if best is None or result["macro_f0.5"] > best[0]["macro_f0.5"]:
                best = (result, thr, top_n)
    return best


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--top-k", type=int, default=blocking.DEFAULT_TOP_K)
    ap.add_argument("--min-score", type=float, default=blocking.DEFAULT_MIN_SCORE)
    ap.add_argument("--probe-keys", type=int, default=blocking.PROBE_KEYS)
    ap.add_argument("--batch-size", type=int, default=blocking.DEFAULT_BATCH_SIZE)
    ap.add_argument("--rebuild-stats", action="store_true",
                    help="Recompute key_freq before training (db.py already builds it).")
    ap.add_argument("--fit-sample", type=int, default=TRAIN_FIT_SAMPLE_SIZE,
                    help="Source-1 entities used to build training pairs.")
    ap.add_argument("--val-sample", type=int, default=VAL_SAMPLE_SIZE,
                    help="Held-out Source-1 entities for threshold tuning.")
    args = ap.parse_args()

    from db import connect as db_connect

    conn = db_connect(args.db)

    if args.rebuild_stats:
        blocking.build_key_stats(conn)

    fit_sample, val_sample = split_entities(conn, args.fit_sample, args.val_sample)
    needed = set(fit_sample) | set(val_sample)
    print("  loading ground truth subset...", flush=True)
    gt = load_ground_truth_subset(args.ground_truth, needed)

    print("  building training pairs...", flush=True)
    X, y, s1_out, cand_out = build_pairs(
        conn, fit_sample, gt, args.top_k, args.min_score, args.batch_size,
        label=True, probe_keys=args.probe_keys,
    )
    pos_rate = sum(y) / len(y) if y else 0.0
    print(f"  training pairs: {len(X)} (positive rate={pos_rate:.3f})", flush=True)

    print("  training LightGBM...", flush=True)
    booster = train_model(X, y)

    print("  building validation pairs...", flush=True)
    Xv, _, s1v, candv = build_pairs(
        conn, val_sample, gt, args.top_k, args.min_score, args.batch_size,
        label=False, probe_keys=args.probe_keys,
    )
    probs = booster.predict(np.asarray(Xv, dtype=np.float32))
    by_entity = group_by_entity(s1v, candv, probs)

    # F_0.5 weights precision 2x, so the optimum often sits at a high
    # threshold — the grid deliberately extends to 0.99 and is fine-grained at
    # the top end. An earlier grid capped at 0.90 and could not even express
    # the precision-heavy operating points this metric rewards.
    thresholds = [round(float(t), 3) for t in np.concatenate([
        np.arange(0.10, 0.80, 0.05),
        np.arange(0.80, 0.99, 0.01),
        [0.99, 0.995],
    ])]
    top_ns = [None, 2, 3, 4, 5, 6, 8, 10, 12, 20]
    best_result, best_thr, best_top_n = tune_threshold(by_entity, val_sample, gt, thresholds, top_ns)
    print(f"  BEST validation macro F0.5={best_result['macro_f0.5']:.4f} "
          f"(precision={best_result['mean_precision']:.4f}, recall={best_result['mean_recall']:.4f}) "
          f"at threshold={best_thr}, top_n={best_top_n}", flush=True)

    # Blocking recall ceiling on the same validation entities (informational).
    total_true = total_found = 0
    for s1 in val_sample:
        true_ids = gt.get(s1, set())
        cand_ids = {cid for cid, _ in by_entity.get(s1, [])}
        total_true += len(true_ids)
        total_found += len(true_ids & cand_ids)
    if total_true:
        print(f"  blocking pair-recall ceiling on val set: {total_found}/{total_true} = {total_found/total_true:.4f}", flush=True)

    os.makedirs(args.models_dir, exist_ok=True)
    booster.save_model(os.path.join(args.models_dir, "model.txt"))
    with open(os.path.join(args.models_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "threshold": best_thr,
            "top_n": best_top_n,
            "feature_names": feat_mod.FEATURE_NAMES,
            "blocking_top_k": args.top_k,
            "blocking_min_score": args.min_score,
            "blocking_probe_keys": args.probe_keys,
            "validation_macro_f0.5": best_result["macro_f0.5"],
            "validation_precision": best_result["mean_precision"],
            "validation_recall": best_result["mean_recall"],
        }, f, indent=2)
    print(f"  saved model + config to {args.models_dir}", flush=True)


if __name__ == "__main__":
    main()
