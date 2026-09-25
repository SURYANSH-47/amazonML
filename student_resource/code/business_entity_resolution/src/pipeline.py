"""Thin CLI wrapping the individual pipeline stages so the whole thing is
reproducible with a handful of documented commands (see README.md).

    python -m src.pipeline build-db   --split train|test --dataset-dir DIR --db PATH
    python -m src.pipeline train      --db PATH --ground-truth PATH --models-dir DIR
    python -m src.pipeline predict    --db PATH --models-dir DIR --candidate-out PATH --matching-out PATH
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import db as db_mod
import train as train_mod
import predict as predict_mod


def main():
    ap = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    sub = ap.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-db", help="Stream a TSV split into a SQLite DB with blocking keys")
    p_build.add_argument("--dataset-dir", default="dataset")
    p_build.add_argument("--split", required=True, choices=["train", "test"])
    p_build.add_argument("--db", required=True)

    p_train = sub.add_parser("train", help="Build labeled pairs, train LightGBM, tune threshold")
    p_train.add_argument("--db", required=True)
    p_train.add_argument("--ground-truth", required=True)
    p_train.add_argument("--models-dir", required=True)
    p_train.add_argument("--top-k", type=int, default=25)
    p_train.add_argument("--min-score", type=int, default=2)
    p_train.add_argument("--max-block-size", type=int, default=200)
    p_train.add_argument("--batch-size", type=int, default=1000)
    p_train.add_argument("--skip-purge", action="store_true")

    p_pred = sub.add_parser("predict", help="Run blocking + trained model over a DB, streaming to output TSVs")
    p_pred.add_argument("--db", required=True)
    p_pred.add_argument("--models-dir", required=True)
    p_pred.add_argument("--candidate-out", required=True)
    p_pred.add_argument("--matching-out", required=True)
    p_pred.add_argument("--limit-s1", type=int, default=None)
    p_pred.add_argument("--skip-purge", action="store_true")

    args = ap.parse_args()

    if args.command == "build-db":
        counts = db_mod.build_db(args.dataset_dir, args.db, args.split)
        print("counts:", counts)
    elif args.command == "train":
        sys.argv = [
            "train.py", "--db", args.db, "--ground-truth", args.ground_truth,
            "--models-dir", args.models_dir, "--top-k", str(args.top_k),
            "--min-score", str(args.min_score), "--max-block-size", str(args.max_block_size),
            "--batch-size", str(args.batch_size),
        ] + (["--skip-purge"] if args.skip_purge else [])
        train_mod.main()
    elif args.command == "predict":
        predict_mod.run(
            args.db, args.models_dir, args.candidate_out, args.matching_out,
            skip_purge=args.skip_purge, limit_s1=args.limit_s1,
        )


if __name__ == "__main__":
    main()
