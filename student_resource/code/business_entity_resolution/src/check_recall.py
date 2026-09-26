"""Measure the blocking stage's recall ceiling and candidate-set size, fast.

This is the single most important diagnostic in the pipeline: the matching
model can never exceed the blocking stage's recall, so if this number is low,
nothing downstream can fix it. Run this BEFORE committing hours to train.py /
predict.py — it needs only a few thousand entities and finishes in minutes.

    python src/check_recall.py --db work/train.db \
        --ground-truth dataset/train/train_ground_truth.tsv --n 3000

Sweep the cost/recall trade-off before a full run:

    python src/check_recall.py --db work/train.db --ground-truth ... \
        --n 2000 --top-k 25 --probe-keys 10 --df-budget 2500
"""

import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import blocking
from db import connect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--n", type=int, default=3000, help="Source-1 entities to evaluate")
    ap.add_argument("--top-k", type=int, default=blocking.DEFAULT_TOP_K)
    ap.add_argument("--min-score", type=float, default=blocking.DEFAULT_MIN_SCORE)
    ap.add_argument("--probe-keys", type=int, default=blocking.PROBE_KEYS)
    ap.add_argument("--df-budget", type=int, default=blocking.PROBE_DF_BUDGET)
    ap.add_argument("--batch-size", type=int, default=blocking.DEFAULT_BATCH_SIZE)
    ap.add_argument("--no-escalate", action="store_true",
                    help="Disable the adaptive second probe pass for hard entities.")
    args = ap.parse_args()

    conn = connect(args.db)
    sample = [r[0] for r in conn.execute("SELECT entity_id FROM source1 LIMIT ?", (args.n,))]
    wanted = set(sample)

    gt = {}
    with open(args.ground_truth, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row and row[0] in wanted:
                gt[row[0]] = set(row[1].split(",")) if len(row) > 1 and row[1] else set()

    t0 = time.time()
    total_true = total_found = full_recall = with_true = 0
    sizes = []
    for s1_id, cands in blocking.generate_candidates(
        conn, top_k=args.top_k, min_score=args.min_score, probe_keys=args.probe_keys,
        df_budget=args.df_budget, batch_size=args.batch_size, s1_ids=sample,
        escalate=not args.no_escalate, progress_every=0,
    ):
        cand_ids = {c[0] for c in cands}
        sizes.append(len(cand_ids))
        true_ids = gt.get(s1_id, set())
        if true_ids:
            with_true += 1
            found = true_ids & cand_ids
            total_true += len(true_ids)
            total_found += len(found)
            if found == true_ids:
                full_recall += 1

    elapsed = time.time() - t0
    print(f"config: top_k={args.top_k} probe_keys={args.probe_keys} "
          f"df_budget={args.df_budget} escalate={not args.no_escalate}")
    print(f"  throughput:         {len(sizes)} entities in {elapsed:.0f}s ({len(sizes)/max(elapsed,1e-9):.1f}/s)")
    print(f"  PAIR RECALL:        {total_found}/{total_true} = {total_found/max(total_true,1):.4f}")
    print(f"  ENTITY FULL-RECALL: {full_recall}/{max(with_true,1)} = {full_recall/max(with_true,1):.4f}")
    if sizes:
        print(f"  candidates/entity:  mean={statistics.mean(sizes):.2f} "
              f"median={statistics.median(sizes)} "
              f"p95={sorted(sizes)[int(len(sizes)*0.95)]} "
              f"empty={sum(1 for s in sizes if s == 0)}")
    conn.close()


if __name__ == "__main__":
    main()
