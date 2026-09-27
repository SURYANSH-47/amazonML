"""Production run of the multi-channel architecture over the test set.

Uses the reranker and decision rule chosen by mc_experiment.py, and the SAME
retrieval / union / feature code (imported from it) so production candidates
and features are constructed exactly as in validation.

Per country: load that country's records once, run every retrieval channel
once for all its queries, build word matrices once, then score queries in
chunks so memory stays bounded (1.7M queries x ~125 candidates would
otherwise be ~28GB of features). Each country writes its own files and is
skipped on re-run if already complete, so a crash costs one country, not
the whole run.

    python src/mc_predict.py --dataset-dir dataset --split test \
        --report work/mc/mc_report.json \
        --out-dir output/mc
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from collections import defaultdict

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import mc_experiment as mc

csv.field_size_limit(10 ** 8)


def parse_config(s):
    """'expf r=0.6 s=0.4 floor=0.05 exclusive=True' -> ('expf', {...})."""
    parts = s.split()
    kv = {}
    for p in parts[1:]:
        k, v = p.split("=")
        kv[k] = None if v == "None" else (v == "True" if v in ("True", "False") else float(v))
    return parts[0], kv


def make_decider(rule, kv):
    if rule == "expf":
        return lambda ps: mc.decide_expf(ps, kv["r"], kv["s"], kv["floor"])
    thr = kv["thr"]
    top_n = int(kv["top_n"]) if kv.get("top_n") else None

    def decide(ps):
        kept = [c for p, c in ps if p >= thr]
        return kept[:top_n] if top_n else kept
    return decide


def score_pairs(models, F, qi_a, ctx_extra=False):
    """Stage-1 score, plus stage 2 over rival-candidate context if enabled.
    Chunks are whole queries, so every candidate of an entity is present when
    its context features are computed — same as in validation."""
    m1, m2 = models
    p = m1.predict(F)
    if m2 is not None:
        p = m2.predict(np.hstack([F, mc.context_features(qi_a, p, F, ctx_extra)]))
    return p


def run_country(country, q_ids, s1rec, args, models, rule, kv, channels, max_df):
    out_c = os.path.join(args.out_dir, f"cand_{country}.tsv")
    out_s = os.path.join(args.out_dir, f"scores_{country}.tsv")
    out_m = os.path.join(args.out_dir, f"match_{country}.tsv")
    if os.path.exists(out_m) and not args.force:
        mc.log(f"\n=== {country}: already done, skipping ({out_m}) ===")
        return
    mc.log(f"\n=== {country}: {len(q_ids)} queries ===")
    t0 = time.time()
    split_dir = os.path.join(args.dataset_dir, args.split)

    Q, T = mc.load_country(country, q_ids, s1rec, split_dir, args.split, args.cache_dir)
    R, c4 = mc.retrieve(Q, T, channels, max_df)
    t1 = time.time()
    wm = mc.build_word_mats(Q, T)
    mc.log(f"  word matrices ({time.time()-t1:.0f}s)")

    n_pairs = 0
    tc = time.time()
    with open(out_c + ".part", "w", encoding="utf-8", newline="") as fc, \
         open(out_s + ".part", "w", encoding="utf-8", newline="") as fs:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fs.write("source1_entity_id\tcandidate_entity_id\tprob\n")
        for s in range(0, Q.n, args.chunk):
            e = min(s + args.chunk, Q.n)
            qi_a, ti_a, chan, _ = mc.build_union(range(s, e), R, c4)
            by_q = defaultdict(list)
            if len(qi_a):
                F = mc.pair_features(Q, T, qi_a, ti_a, chan, wm)
                p = score_pairs(models, F, qi_a, args.ctx_extra)
                del F
                for q, t, pr in zip(qi_a, ti_a, p):
                    by_q[q].append((float(pr), t))
            for q in range(s, e):
                lst = sorted(by_q.get(q, ()), reverse=True)
                qid = Q.ids[q]
                fc.write(f"{qid}\t{','.join(T.ids[t] for _, t in lst)}\n")
                for pr, t in lst:
                    if pr >= args.score_floor:
                        fs.write(f"{qid}\t{T.ids[t]}\t{pr:.5f}\n")
            n_pairs += len(qi_a)
            done = e
            rate = done / max(time.time() - tc, 1e-9)
            mc.log(f"  scored {done}/{Q.n} queries, {n_pairs} pairs "
                   f"({rate:.0f} q/s, ETA {(Q.n-done)/max(rate,1e-9)/60:.1f} min)")
            del qi_a, ti_a, chan, by_q
            gc.collect()
    os.replace(out_c + ".part", out_c)
    os.replace(out_s + ".part", out_s)

    decide = make_decider(rule, kv)
    per = defaultdict(list)
    with open(out_s, encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, cid, pr = line.rstrip("\n").split("\t")
            per[qid].append((float(pr), cid))
    for v in per.values():
        v.sort(reverse=True)
    if kv.get("exclusive"):
        # Records are country-scoped, so one-owner enforcement per country is
        # exactly global enforcement.
        preds = mc.exclusive(per, decide)
    else:
        preds = {qid: decide(ps) for qid, ps in per.items()}
    n_match = 0
    with open(out_m + ".part", "w", encoding="utf-8", newline="") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for qid in Q.ids:
            ids = preds.get(qid, [])
            n_match += bool(ids)
            fm.write(f"{qid}\t{','.join(ids)}\n")
    os.replace(out_m + ".part", out_m)
    mc.log(f"  {country}: {n_match}/{Q.n} entities matched, "
           f"{n_pairs/max(Q.n,1):.1f} candidates/query ({time.time()-t0:.0f}s)")
    del Q, T, R, c4, wm, per, preds
    gc.collect()


def merge(args, all_ids):
    for kind, header, out in (
        ("cand", "source1_entity_id\tcandidate_entity_ids", "candidate_pairs.tsv"),
        ("match", "source1_entity_id\tmatched_entity_ids", "matching_results.tsv"),
    ):
        seen = set()
        path = os.path.join(args.out_dir, out)
        with open(path, "w", encoding="utf-8", newline="") as fo:
            fo.write(header + "\n")
            for fname in sorted(os.listdir(args.out_dir)):
                if fname.startswith(kind + "_") and fname.endswith(".tsv"):
                    with open(os.path.join(args.out_dir, fname), encoding="utf-8") as fi:
                        next(fi)
                        for line in fi:
                            qid = line.split("\t", 1)[0]
                            if qid in seen:
                                sys.exit(f"ERROR: duplicate {qid} in {fname}")
                            seen.add(qid)
                            fo.write(line)
        missing = all_ids - seen
        if missing:
            sys.exit(f"ERROR: {out} missing {len(missing)} entities, e.g. "
                     f"{sorted(missing)[:5]} — a country did not finish.")
        mc.log(f"  merged {out}: {len(seen)} rows, all required entities present")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--report", default=None,
                    help="mc_report.json from mc_experiment.py; mc_m1.txt / mc_m2.txt must sit next to it.")
    ap.add_argument("--out-dir", default="output/mc")
    ap.add_argument("--countries", default=None, help="Comma list; default all.")
    ap.add_argument("--chunk", type=int, default=40000)
    ap.add_argument("--score-floor", type=float, default=0.02)
    ap.add_argument("--config", default=None,
                    help="Override decision rule, e.g. 'thresh thr=0.6 top_n=None exclusive=True'.")
    ap.add_argument("--force", action="store_true", help="Recompute finished countries.")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--cache-dir", default="work/mc_cache",
                    help="Normalized target records are cached here per country.")
    ap.add_argument("--prep-only", action="store_true",
                    help="Only normalize+cache target records for --countries, then exit. "
                         "Needs no model; lets a second laptop get ahead while training runs.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    split_dir = os.path.join(args.dataset_dir, args.split)

    if args.prep_only:
        want = args.countries.split(",") if args.countries else ["France", "India", "US"]
        for country in want:
            mc.log(f"\n=== prep {country} ===")
            mc.load_country(country, [], {}, split_dir, args.split, args.cache_dir)
        mc.log("prep done")
        return

    with open(args.report) as f:
        rep = json.load(f)
    mc.TOPK.update(rep["topk"])
    mc.QKEEP.update(rep.get("qkeep", {}))
    args.ctx_extra = bool(rep.get("ctx_extra", False))
    channels = set(rep["channels"]) - {"c1"}
    max_df = rep["max_df"]
    rule, kv = parse_config(args.config or rep["best_config"])
    mdir = os.path.dirname(args.report)
    m1 = lgb.Booster(model_file=os.path.join(mdir, "mc_m1.txt"))
    m2 = (lgb.Booster(model_file=os.path.join(mdir, "mc_m2.txt"))
          if rep.get("two_stage") else None)
    mc.log(f"channels {sorted(channels)}, topk {mc.TOPK}, qkeep {mc.QKEEP}, max_df {max_df}")
    mc.log(f"reranker: {'two-stage' if m2 is not None else 'stage 1 only'}; "
           f"decision: {rule} {kv}")

    s1_path = os.path.join(args.dataset_dir, args.split, f"{args.split}_source1.tsv")
    s1rec = {}
    with open(s1_path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for row in r:
            if row:
                row = row + [""] * (4 - len(row))
                s1rec[row[0]] = (row[1], row[2], row[3])
    by_country = defaultdict(list)
    for e, rec in s1rec.items():
        by_country[rec[2]].append(e)
    mc.log("queries per country: " + ", ".join(f"{c} {len(v)}" for c, v in sorted(by_country.items())))

    if not args.merge_only:
        want = set(args.countries.split(",")) if args.countries else set(by_country)
        # Smallest first: a failure surfaces fast, and each finished country
        # is banked on disk.
        for country in sorted(want, key=lambda c: len(by_country[c])):
            run_country(country, by_country[country], s1rec, args, (m1, m2),
                        rule, kv, channels, max_df)

    done = {f[6:-4] for f in os.listdir(args.out_dir)
            if f.startswith("match_") and f.endswith(".tsv")}
    if done >= set(by_country):
        merge(args, set(s1rec))
    else:
        mc.log(f"not merging yet — finished: {sorted(done)}, "
               f"remaining: {sorted(set(by_country) - done)}")


if __name__ == "__main__":
    main()
