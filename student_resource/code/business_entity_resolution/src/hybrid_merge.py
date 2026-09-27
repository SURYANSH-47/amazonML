"""Assemble a submission from per-country outputs of different runs.

For countries the multi-channel pipeline finished (match_<C>.tsv /
cand_<C>.tsv in --new-dir), rows come from it; every other country's rows
come from an older complete submission. Verifies every required Source-1
entity appears exactly once.

Only matching_results.tsv is scored on the leaderboard, so --old-candidates
is optional: candidate_pairs.tsv is only written when both candidate sources
are available.

    python src/hybrid_merge.py --test-source1 dataset/test/test_source1.tsv \
        --new-dir output/mc --old-matching output/matching_results.tsv \
        --out-dir output/hybrid [--old-candidates output/candidate_pairs.tsv]
"""

import argparse
import csv
import os
import sys

csv.field_size_limit(10 ** 8)


def read_rows(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            if not line.strip():
                continue
            rows[line.split("\t", 1)[0]] = line if line.endswith("\n") else line + "\n"
    return rows


def assemble(kind, header, s1_country, new_countries, new_dir, old_path, out_path):
    new_rows = {}
    for c in new_countries:
        new_rows.update(read_rows(os.path.join(new_dir, f"{kind}_{c}.tsv")))
    old_rows = read_rows(old_path)
    n_new = n_old = 0
    missing = []
    with open(out_path, "w", encoding="utf-8", newline="") as fo:
        fo.write(header + "\n")
        for s1, country in s1_country.items():
            if country in new_countries:
                row = new_rows.get(s1)
                n_new += row is not None
            else:
                row = old_rows.get(s1)
                n_old += row is not None
            if row is None:
                missing.append(s1)
                continue
            fo.write(row)
    if missing:
        sys.exit(f"ERROR: {len(missing)} entities missing from {kind} sources, "
                 f"e.g. {missing[:5]}")
    print(f"  {out_path}: {n_new} rows from new run ({', '.join(sorted(new_countries))}), "
          f"{n_old} rows from old run — all {len(s1_country)} entities present", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-source1", required=True)
    ap.add_argument("--new-dir", required=True, help="Folder with match_<C>.tsv / cand_<C>.tsv")
    ap.add_argument("--old-matching", required=True, help="Complete older matching_results.tsv")
    ap.add_argument("--old-candidates", default=None, help="Complete older candidate_pairs.tsv")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    s1_country = {}
    with open(args.test_source1, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for row in r:
            if row:
                s1_country[row[0]] = row[3] if len(row) > 3 else ""

    new_countries = {f[6:-4] for f in os.listdir(args.new_dir)
                     if f.startswith("match_") and f.endswith(".tsv")}
    print(f"new-run countries found: {sorted(new_countries)}", flush=True)

    assemble("match", "source1_entity_id\tmatched_entity_ids", s1_country, new_countries,
             args.new_dir, args.old_matching,
             os.path.join(args.out_dir, "matching_results.tsv"))

    cand_new = all(os.path.exists(os.path.join(args.new_dir, f"cand_{c}.tsv"))
                   for c in new_countries)
    if args.old_candidates and cand_new:
        assemble("cand", "source1_entity_id\tcandidate_entity_ids", s1_country,
                 new_countries, args.new_dir, args.old_candidates,
                 os.path.join(args.out_dir, "candidate_pairs.tsv"))
    else:
        print("  candidate_pairs.tsv not written (needs --old-candidates and "
              "cand_<C>.tsv for every new country)", flush=True)


if __name__ == "__main__":
    main()
