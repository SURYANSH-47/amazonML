"""Final decision layer: scored pairs -> matching_results.tsv, in minutes.

predict.py --scores-out dumps the model probability for every candidate pair.
All decisions happen here instead of inside the multi-hour predict pass, so
they can be retuned and re-applied without re-running it.

Two things this does that predict.py cannot, because they need a global view:

1. One-owner enforcement. In the training ground truth every Source-2/3
   record belongs to at most one Source-1 entity (0 of 7,638,365 matched ids
   have two owners). predict.py decides each entity independently, so one
   chain-store record can be matched to several same-name entities; every
   claim beyond the rightful one is a certain false merge. Here each record
   goes only to its highest-probability claimant.

2. Per-entity decision rule. Besides the fixed threshold + top-N, supports an
   expected-F0.5 rule: for each entity pick the number of matches m that
   maximises E[F0.5] ~= 1.25*S_m / (0.25*T + m), where S_m is the sum of the
   top-m probabilities and T the expected true-match count. It adapts to
   entities with one strong match vs five, and predicts empty (singleton,
   worth a full 1.0 when right) when nothing is convincing.

Tune on held-out validation dumps (with ground truth), then apply to test:

    python src/assign.py tune --scores "work/val_scores_p*.tsv" \
        --universe "work/val_match_p*.tsv" \
        --ground-truth dataset/train/train_ground_truth.tsv

    python src/assign.py apply --scores "output/scores_p*.tsv" \
        --universe dataset/test/test_source1.tsv \
        --rule expf --r 0.75 --s 0.6 --exclusive \
        --out output/matching_results.tsv
"""

import argparse
import csv
import glob
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import f_beta_one


def load_universe(pattern):
    """Every Source-1 id that must get a row. Accepts a source1 TSV or any
    per-entity output TSV (first column = Source-1 id)."""
    ids = []
    seen = set()
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8") as f:
            next(f)
            for line in f:
                e = line.split("\t", 1)[0].strip()
                if e and e not in seen:
                    seen.add(e)
                    ids.append(e)
    return ids


def load_pairs(pattern, exclusive):
    """Return {s1: [(prob, cand), ...] sorted desc}, applying one-owner rule."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"No score files matched {pattern!r}")
    t0 = time.time()

    best = {}
    if exclusive:
        for p in paths:
            with open(p, encoding="utf-8") as f:
                next(f)
                for line in f:
                    s1, cand, prob = line.rstrip("\n").split("\t")
                    prob = float(prob)
                    cur = best.get(cand)
                    if cur is None or prob > cur[0]:
                        best[cand] = (prob, s1)

    per_s1 = {}
    n_pairs = n_dropped = 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            next(f)
            for line in f:
                s1, cand, prob = line.rstrip("\n").split("\t")
                n_pairs += 1
                if exclusive and best[cand][1] != s1:
                    n_dropped += 1
                    continue
                per_s1.setdefault(s1, []).append((float(prob), cand))
    for v in per_s1.values():
        v.sort(reverse=True)
    print(f"  loaded {n_pairs} scored pairs from {len(paths)} files; "
          f"one-owner rule dropped {n_dropped} ({100*n_dropped/max(n_pairs,1):.1f}%) "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return per_s1


def decide_thresh(pairs, thr, top_n):
    kept = [c for p, c in pairs if p >= thr]
    return kept[:top_n] if top_n else kept


def decide_fallback(pairs, thr_hi, thr_lo, top_n):
    """Threshold rule plus a fallback for entities that would come out empty.

    Under macro-F0.5 an entity predicted empty scores 0.0 whenever it has any
    true match — and only ~5.6% of entities are genuine singletons, while a
    single fixed threshold leaves ~12% empty. So when nothing clears thr_hi,
    still take the best candidate if it clears thr_lo: right, it scores ~0.7
    (precision 1, partial recall); wrong, it scores 0 — which is what empty
    already scored for every non-singleton.
    """
    kept = [c for p, c in pairs if p >= thr_hi]
    if top_n:
        kept = kept[:top_n]
    if not kept and pairs and pairs[0][0] >= thr_lo:
        kept = [pairs[0][1]]
    return kept


def decide_expf(pairs, r, s, floor):
    """Pick m maximising approximate expected F0.5 (beta^2 = 0.25)."""
    pairs = [(p, c) for p, c in pairs if p >= floor]
    if not pairs:
        return []
    probs = [p for p, _ in pairs]
    total = sum(probs)
    T = total / r  # expected true matches incl. ones blocking never surfaced
    p_none = s
    for p in probs:
        p_none *= (1.0 - p)
    best_m, best_v = 0, p_none
    run = 0.0
    for m, p in enumerate(probs, start=1):
        run += p
        v = 1.25 * run / (0.25 * T + m)
        if v > best_v:
            best_m, best_v = m, v
    return [c for _, c in pairs[:best_m]]


def load_gt(path, universe):
    want = set(universe)
    gt = {}
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row and row[0] in want:
                gt[row[0]] = set(row[1].split(",")) if len(row) > 1 and row[1] else set()
    return gt


def score(universe, gt, per_s1, decide):
    tot = 0.0
    for e in universe:
        tot += f_beta_one(gt.get(e, set()), set(decide(per_s1.get(e, []))))
    return tot / max(len(universe), 1)


def diagnose(universe, gt, per_s1, decide, label):
    """Break the score down by where it is lost."""
    n = len(universe)
    empty_single = empty_miss_found = empty_miss_blocked = 0
    for e in universe:
        pred = decide(per_s1.get(e, []))
        if pred:
            continue
        truth = gt.get(e, set())
        if not truth:
            empty_single += 1
        elif truth & {c for _, c in per_s1.get(e, [])}:
            empty_miss_found += 1
        else:
            empty_miss_blocked += 1
    print(f"\n  diagnosis [{label}] over {n} entities:")
    print(f"    predicted empty, truly singleton (scores 1.0):        {empty_single:>7} ({100*empty_single/n:.1f}%)")
    print(f"    predicted empty, true match WAS scored (recoverable):  {empty_miss_found:>7} ({100*empty_miss_found/n:.1f}%)")
    print(f"    predicted empty, true match never scored (blocking):   {empty_miss_blocked:>7} ({100*empty_miss_blocked/n:.1f}%)")
    print(f"    -> every one of the last two lines scores 0.0; max recoverable by a "
          f"decision rule alone ~ {100*empty_miss_found/n:.1f} points of macro F0.5 x per-hit score")


def cmd_tune(args):
    universe = load_universe(args.universe)
    gt = load_gt(args.ground_truth, universe)
    print(f"  universe: {len(universe)} entities, ground truth for {len(gt)}", flush=True)

    results = []
    for exclusive in (False, True):
        per_s1 = load_pairs(args.scores, exclusive)
        if not exclusive:
            diagnose(universe, gt, per_s1, lambda ps: decide_thresh(ps, 0.6, 6),
                     "current rule thr=0.6 top_n=6")
        for thr in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
            for top_n in [None, 4, 5, 6, 8]:
                f = score(universe, gt, per_s1, lambda ps: decide_thresh(ps, thr, top_n))
                results.append((f, f"thresh thr={thr} top_n={top_n} exclusive={exclusive}",
                                ("thresh", thr, None, top_n, exclusive)))
        for hi in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            for lo in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
                if lo >= hi:
                    continue
                for top_n in [None, 5, 6, 8]:
                    f = score(universe, gt, per_s1, lambda ps: decide_fallback(ps, hi, lo, top_n))
                    results.append((f, f"fallback hi={hi} lo={lo} top_n={top_n} exclusive={exclusive}",
                                    ("fallback", hi, lo, top_n, exclusive)))
        for r in [0.55, 0.65, 0.75, 0.85, 1.0]:
            for s in [0.2, 0.4, 0.6, 0.8, 1.0]:
                for floor in [0.05, 0.1, 0.2]:
                    f = score(universe, gt, per_s1, lambda ps: decide_expf(ps, r, s, floor))
                    results.append((f, f"expf r={r} s={s} floor={floor} exclusive={exclusive}",
                                    ("expf", r, s, floor, exclusive)))

    results.sort(key=lambda x: -x[0])
    print("\n  TOP 15 configurations (macro F0.5 on held-out validation):")
    for f, name, _ in results[:15]:
        print(f"    {f:.4f}  {name}")
    base = [f for f, n, _ in results if n == "thresh thr=0.6 top_n=6 exclusive=False"]
    if base:
        print(f"\n  reference — current submission's rule (thr=0.6, top_n=6): {base[0]:.4f}")
        print(f"  best:                                                   {results[0][0]:.4f} "
              f"(+{results[0][0]-base[0]:.4f})")

    best = results[0][2]
    if best[0] == "fallback":
        per_s1 = load_pairs(args.scores, False)
        diagnose(universe, gt, per_s1, lambda ps: decide_fallback(ps, best[1], best[2], best[3]),
                 f"best fallback hi={best[1]} lo={best[2]} top_n={best[3]}")


def cmd_apply(args):
    universe = load_universe(args.universe)
    per_s1 = load_pairs(args.scores, args.exclusive)
    if args.rule == "thresh":
        decide = lambda ps: decide_thresh(ps, args.thr, args.top_n)
    else:
        decide = lambda ps: decide_expf(ps, args.r, args.s, args.floor)

    n_match = n_nonempty = 0
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for e in universe:
            ids = decide(per_s1.get(e, []))
            if ids:
                n_nonempty += 1
                n_match += len(ids)
            f.write(f"{e}\t{','.join(ids)}\n")
    print(f"  wrote {len(universe)} rows to {args.out}: {n_nonempty} with matches, "
          f"{n_match} total matches", flush=True)


def cmd_patch(args):
    """Resolve conflicts in an existing matching_results.tsv.

    Only the conflicted entities (conflicts.py --write-entities) are re-scored;
    their rows are replaced with a decision made under the one-owner rule,
    and every other row is copied through untouched. This is exact, not an
    approximation: every claimant of a contested record is a conflicted
    entity by definition, so the re-scored set contains every competitor.
    """
    rescored = set(load_universe(args.rescored))
    per_s1 = load_pairs(args.scores, exclusive=False)
    print(f"  re-scored entities: {len(rescored)}", flush=True)

    # Records already held by entities we are NOT re-deciding stay theirs.
    # Safe: such an owner cleared the original threshold, and any re-scored
    # entity that also cleared it would have claimed the record in the base
    # file too — making it a conflict, hence re-scored. So only a weaker
    # (fallback-level) claim can collide with a locked record, and it loses.
    locked = set()
    base_old = {}
    with open(args.base, encoding="utf-8") as fin:
        fin.readline()
        for line in fin:
            s1, _, rest = line.rstrip("\n").partition("\t")
            ids = rest.split(",") if rest else []
            if s1 in rescored:
                base_old[s1] = ids
            else:
                locked.update(ids)

    if args.rule == "fallback":
        decide = lambda ps: decide_fallback(ps, args.thr, args.lo, args.top_n)
    elif args.rule == "expf":
        decide = lambda ps: decide_expf(ps, args.r, args.s, args.floor)
    else:
        decide = lambda ps: decide_thresh(ps, args.thr, args.top_n)

    # Decide per entity first, then enforce one owner on the FINAL matches.
    # Resolving on candidate lists instead would strip a record from an
    # entity in favour of a rival that never actually keeps it.
    decisions = {}
    for e in rescored:
        pairs = [(p, c) for p, c in per_s1.get(e, []) if c not in locked]
        prob = {c: p for p, c in pairs}
        decisions[e] = [(prob[c], c) for c in decide(pairs)]
    owner = {}
    for e, kept in decisions.items():
        for p, c in kept:
            if c not in owner or p > owner[c][0]:
                owner[c] = (p, e)
    final = {e: [c for p, c in kept if owner[c][1] == e] for e, kept in decisions.items()}

    n_changed = n_removed = n_added = n_rows = n_filled = 0
    with open(args.base, encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8", newline="") as fout:
        fout.write(fin.readline())
        for line in fin:
            n_rows += 1
            s1, _, _ = line.rstrip("\n").partition("\t")
            if s1 not in rescored:
                fout.write(line if line.endswith("\n") else line + "\n")
                continue
            old, new = base_old.get(s1, []), final.get(s1, [])
            if new != old:
                n_changed += 1
                n_removed += len(set(old) - set(new))
                n_added += len(set(new) - set(old))
                if not old and new:
                    n_filled += 1
            fout.write(f"{s1}\t{','.join(new)}\n")
    print(f"  {n_rows} rows written to {args.out}", flush=True)
    print(f"  entities changed: {n_changed} | matches removed: {n_removed} | "
          f"matches added: {n_added} | previously-empty entities now matched: {n_filled}",
          flush=True)


def cmd_patch_cands(args):
    """Replace candidate rows for re-scored entities.

    Required when re-scoring used a larger --top-k: the submission rule is
    that candidate_pairs.tsv is exactly the set fed to the model, and every
    matched id must appear in it.
    """
    new_rows = {}
    for p in sorted(glob.glob(args.rescored_cands)):
        with open(p, encoding="utf-8") as f:
            f.readline()
            for line in f:
                s1 = line.split("\t", 1)[0]
                new_rows[s1] = line if line.endswith("\n") else line + "\n"
    n = n_rep = 0
    with open(args.base, encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8", newline="") as fout:
        fout.write(fin.readline())
        for line in fin:
            n += 1
            s1 = line.split("\t", 1)[0]
            if s1 in new_rows:
                fout.write(new_rows[s1])
                n_rep += 1
            else:
                fout.write(line if line.endswith("\n") else line + "\n")
    print(f"  {n} candidate rows written to {args.out}; {n_rep} replaced", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("patch-cands", help="Replace candidate rows for re-scored entities.")
    pc.add_argument("--base", required=True, help="Existing candidate_pairs.tsv.")
    pc.add_argument("--rescored-cands", required=True, help="Glob of re-scored candidate files.")
    pc.add_argument("--out", required=True)

    pt = sub.add_parser("patch", help="Resolve conflicts in an existing submission.")
    pt.add_argument("--base", required=True, help="Existing matching_results.tsv.")
    pt.add_argument("--scores", required=True, help="Glob of score dumps for conflicted entities.")
    pt.add_argument("--rescored", required=True,
                    help="Glob of files listing the re-scored entity ids.")
    pt.add_argument("--out", required=True)
    pt.add_argument("--rule", choices=["thresh", "fallback", "expf"], default="thresh")
    pt.add_argument("--r", type=float, default=0.75, help="expf: assumed blocking recall.")
    pt.add_argument("--s", type=float, default=1.0, help="expf: singleton prior weight.")
    pt.add_argument("--floor", type=float, default=0.05, help="expf: ignore probs below this.")
    pt.add_argument("--thr", type=float, default=0.6,
                    help="Main threshold (thr_hi for fallback). Keep equal to the "
                         "base run's threshold unless re-scoring every entity.")
    pt.add_argument("--lo", type=float, default=0.2,
                    help="Fallback: take the best candidate if nothing clears --thr "
                         "but it clears this.")
    pt.add_argument("--top-n", type=int, default=6)

    t = sub.add_parser("tune")
    t.add_argument("--scores", required=True, help="Glob of validation score dumps.")
    t.add_argument("--universe", required=True, help="Glob of files listing every S1 id.")
    t.add_argument("--ground-truth", required=True)

    a = sub.add_parser("apply")
    a.add_argument("--scores", required=True)
    a.add_argument("--universe", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--rule", choices=["thresh", "expf"], default="expf")
    a.add_argument("--exclusive", action="store_true")
    a.add_argument("--thr", type=float, default=0.6)
    a.add_argument("--top-n", type=int, default=6)
    a.add_argument("--r", type=float, default=0.75)
    a.add_argument("--s", type=float, default=0.6)
    a.add_argument("--floor", type=float, default=0.1)

    args = ap.parse_args()
    {"tune": cmd_tune, "apply": cmd_apply, "patch": cmd_patch,
     "patch-cands": cmd_patch_cands}[args.cmd](args)


if __name__ == "__main__":
    main()
