"""Count guaranteed false positives in a matching_results.tsv.

In the training ground truth every Source-2/3 record belongs to at most one
Source-1 entity (0 of 7,638,365 matched ids have two owners). So any record
claimed by several Source-1 entities in a prediction file is a certain error
for all but at most one of them.

    python src/conflicts.py output/matching_results.tsv
"""

import collections
import sys


def main(path, write_entities=None, include_empty=False):
    claims = collections.Counter()
    n_rows = n_nonempty = n_matches = 0
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            n_rows += 1
            _, _, rest = line.rstrip("\n").partition("\t")
            if not rest:
                continue
            ids = rest.split(",")
            n_nonempty += 1
            n_matches += len(ids)
            claims.update(ids)

    contested = {k: v for k, v in claims.items() if v > 1}
    excess = sum(v - 1 for v in contested.values())
    hit_ids = []
    n_empty = 0
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            if rest and any(i in contested for i in rest.split(",")):
                hit_ids.append(s1)
            elif not rest:
                n_empty += 1
                if include_empty:
                    hit_ids.append(s1)
    entities_hit = len(hit_ids) - (n_empty if include_empty else 0)
    print(f"entities predicted empty: {n_empty}"
          + (" (included in the re-score list)" if include_empty else ""))
    if write_entities:
        # Every claimant of a contested record is in this list by definition,
        # so re-scoring exactly these entities gives the full information
        # needed to resolve every conflict — nothing else needs re-running.
        with open(write_entities, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\n")
            for e in hit_ids:
                f.write(e + "\n")
        print(f"wrote {len(hit_ids)} entity ids to {write_entities} "
              f"({entities_hit} conflicted"
              + (f" + {n_empty} empty)" if include_empty else ")"))

    print(f"rows: {n_rows}, entities with matches: {n_nonempty}, total matches: {n_matches}")
    print(f"records claimed by >1 entity: {len(contested)}")
    print(f"GUARANTEED false positives (excess claims): {excess} "
          f"= {100*excess/max(n_matches,1):.2f}% of all predicted matches")
    print(f"entities touched by a conflict: {entities_hit} "
          f"({100*entities_hit/max(n_nonempty,1):.1f}% of entities with matches)")
    if contested:
        print(f"max claimants on one record: {max(contested.values())}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="output/matching_results.tsv")
    ap.add_argument("--write-entities", default=None,
                    help="Write the ids of every entity touched by a conflict here.")
    ap.add_argument("--include-empty", action="store_true",
                    help="Also list entities predicted empty, so a fallback rule can "
                         "re-decide them in the same targeted re-score.")
    args = ap.parse_args()
    main(args.path, args.write_entities, args.include_empty)
