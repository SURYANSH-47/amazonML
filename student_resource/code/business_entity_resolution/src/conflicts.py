"""Count guaranteed false positives in a matching_results.tsv.

In the training ground truth every Source-2/3 record belongs to at most one
Source-1 entity (0 of 7,638,365 matched ids have two owners). So any record
claimed by several Source-1 entities in a prediction file is a certain error
for all but at most one of them.

    python src/conflicts.py output/matching_results.tsv
"""

import collections
import sys


def main(path):
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
    entities_hit = 0
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            _, _, rest = line.rstrip("\n").partition("\t")
            if rest and any(i in contested for i in rest.split(",")):
                entities_hit += 1

    print(f"rows: {n_rows}, entities with matches: {n_nonempty}, total matches: {n_matches}")
    print(f"records claimed by >1 entity: {len(contested)}")
    print(f"GUARANTEED false positives (excess claims): {excess} "
          f"= {100*excess/max(n_matches,1):.2f}% of all predicted matches")
    print(f"entities touched by a conflict: {entities_hit} "
          f"({100*entities_hit/max(n_nonempty,1):.1f}% of entities with matches)")
    if contested:
        print(f"max claimants on one record: {max(contested.values())}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/matching_results.tsv")
