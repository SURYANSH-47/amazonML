"""Merge sharded predict outputs into the two submission files.

Verifies every required Source-1 entity appears exactly once, since a missing
entity is a hard submission rejection — if a shard died partway, this catches
it here rather than at upload time.

    python src/merge_shards.py --parts "output/matching_part*.tsv" \
        --out output/matching_results.tsv \
        --header "source1_entity_id\tmatched_entity_ids" \
        --test-source1 dataset/test/test_source1.tsv
"""

import argparse
import csv
import glob
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="Glob for the shard files.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--header", required=True,
                    help=r"Header line, e.g. 'source1_entity_id\tmatched_entity_ids'")
    ap.add_argument("--test-source1", default=None,
                    help="If given, verify every entity in it appears exactly once.")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.parts))
    if not paths:
        sys.exit(f"No files matched {args.parts!r}")
    print(f"merging {len(paths)} shard files", flush=True)

    seen = set()
    n_rows = 0
    header = args.header.replace("\\t", "\t")
    with open(args.out, "w", encoding="utf-8", newline="") as out:
        out.write(header + "\n")
        for p in paths:
            with open(p, encoding="utf-8") as f:
                first = f.readline()  # skip each shard's own header
                if first and not first.startswith("source1_entity_id"):
                    out.write(first)
                    n_rows += 1
                    seen.add(first.split("\t", 1)[0])
                for line in f:
                    if not line.strip("\n"):
                        continue
                    out.write(line)
                    n_rows += 1
                    seen.add(line.split("\t", 1)[0])
    print(f"  wrote {n_rows} rows to {args.out}", flush=True)

    if args.test_source1:
        required = set()
        with open(args.test_source1, encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r)
            for row in r:
                if row:
                    required.add(row[0])
        missing = required - seen
        extra = seen - required
        if missing:
            sys.exit(f"ERROR: {len(missing)} required entities missing, "
                     f"e.g. {sorted(missing)[:5]} — a shard likely died; rerun it.")
        if extra:
            print(f"  WARNING: {len(extra)} unexpected ids present")
        if n_rows != len(required):
            print(f"  WARNING: {n_rows} rows for {len(required)} entities (duplicates?)")
        else:
            print(f"  OK: all {len(required)} required entities present exactly once")


if __name__ == "__main__":
    main()
