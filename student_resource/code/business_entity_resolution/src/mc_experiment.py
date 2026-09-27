"""Multi-channel retrieval + reranker: validation harness.

Runs the whole new architecture on a held-out validation set and reports
exactly what decides the score, per channel and for the union:

  retrieval: true-pair recall, % entities with none / all true matches
             retrieved, candidate-count distribution
  final:     precision, recall, macro F0.5 under the best decision rule

Channels (each independent; candidates are UNIONED):
  c2   name char-3gram TF-IDF top-k       typos, partial / domain names
  c2b  phonetic-skeleton char-3gram top-k native-script <-> English names
  c3   address char-3gram TF-IDF top-k    replaced / garbage names
       (compound house numbers preserved as single tokens)
  c4   exact keys: full / sorted / phonetic name, house no., postal
  c1   (optional) the original SQLite IDF blocking, deep pool

Everything is vectorized in memory per country: sparse top-k matmul for
retrieval, rapidfuzz.cpdist and sparse row products for pair features.
Nothing reads SQLite per entity except the optional c1 channel.

    python src/mc_experiment.py --dataset-dir dataset --out-dir work/mc \
        --n-val 40000 --n-fit 30000 --db work/train.db
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import f_beta_one
from mc_normalize import (house_set, name_sorted, norm_addr, norm_name,
                          phon_compact, phon_string, postal_set)
from train import VAL_FRACTION_HASH, stable_bucket

csv.field_size_limit(10 ** 8)
THREADS = os.cpu_count() or 4

TOPK = {"c2": 50, "c2b": 40, "c3": 50}
EXACT_MAX_DF = 200  # an exact key shared by more records than this is not selective

FEATURES = [
    "cand_source",
    "r_name", "r_name_tsort", "r_name_tset", "r_name_part",
    "r_phon", "r_addr", "r_addr_tset",
    "name_word_jac", "name_word_ovl", "n_name_words",
    "addr_word_jac", "addr_word_ovl", "n_addr_words",
    "name_exact", "sorted_exact", "phon_exact",
    "house_match", "postal_match",
    "len_ratio", "s1_addr_empty", "cand_addr_empty",
    "c2_cos", "c2_rank", "c2b_cos", "c2b_rank", "c3_cos", "c3_rank",
    "c4_keys", "c1_score", "c1_rank", "n_channels",
]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- loading

def select_entities(s1_path, n_val, n_fit):
    """Val = first n_val validation-bucket ids in sorted order, which is the
    same set the earlier `predict.py --select val` dumps used (so results are
    comparable to the 0.7930 baseline). Fit = first n_fit training-bucket ids.
    """
    ids = []
    with open(s1_path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        ids = [row[0] for row in r if row]
    ids.sort()
    val = [e for e in ids if stable_bucket(e) < VAL_FRACTION_HASH][:n_val]
    fit = [e for e in ids if stable_bucket(e) >= VAL_FRACTION_HASH][:n_fit]
    return val, fit


class Records:
    """Normalized fields for a set of records, as parallel lists/arrays."""

    def __init__(self, ids, src, names_raw, addrs_raw):
        t0 = time.time()
        self.ids = ids
        self.src = np.asarray(src, dtype=np.int8)
        self.name = [norm_name(x) for x in names_raw]
        self.addr = [norm_addr(x) for x in addrs_raw]
        self.phon = [phon_string(x) for x in self.name]
        self.phonc = [p.replace(" ", "") for p in self.phon]
        self.sorted = [name_sorted(x) for x in self.name]
        self.house = [house_set(x) for x in addrs_raw]
        self.postal = [postal_set(x) for x in addrs_raw]
        self.n = len(ids)
        log(f"    normalized {self.n} records ({time.time()-t0:.0f}s)")


def load_s1(path, want):
    out = {}
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for row in r:
            if row and row[0] in want:
                row = row + [""] * (4 - len(row))
                out[row[0]] = (row[1], row[2], row[3])
    return out


def load_targets(split_dir, split, country):
    ids, src, names, addrs = [], [], [], []
    for sn in (2, 3):
        with open(os.path.join(split_dir, f"{split}_source{sn}.tsv"), encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t"); next(r)
            for row in r:
                if len(row) >= 4 and row[3] == country:
                    ids.append(row[0]); src.append(sn)
                    names.append(row[1]); addrs.append(row[2])
    return ids, src, names, addrs


def load_gt(path, want):
    gt = {}
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for row in r:
            if row and row[0] in want:
                gt[row[0]] = set(row[1].split(",")) if len(row) > 1 and row[1] else set()
    return gt


# -------------------------------------------------------------- channels

# Query-side n-gram pruning: each query searches with only its `keep`
# highest-weight (i.e. rarest) fragments. Search cost per query is the sum of
# the posting-list lengths of its fragments, which is dominated by the common
# ones ("del", "mum", "roa") that carry almost no identifying signal. At the
# full 70k validation queries the unpruned search extrapolated to ~12h for
# the 1.73M test queries; pruning attacks exactly that cost.
QKEEP = {"c2": 14, "c2b": 12, "c3": 22}


def prune_query_rows(Q, keep):
    """Keep each row's `keep` largest entries; drop the rest."""
    Q = Q.tocsr()
    ip, dt = Q.indptr, Q.data
    lens = np.diff(ip)
    if not len(lens) or lens.max() <= keep:
        return Q
    mask = np.ones(len(dt), dtype=bool)
    for i in np.nonzero(lens > keep)[0]:
        a, b = ip[i], ip[i + 1]
        n_drop = (b - a) - keep
        mask[a + np.argpartition(dt[a:b], n_drop)[:n_drop]] = False
    rows = np.repeat(np.arange(len(lens)), lens)
    counts = np.bincount(rows[mask], minlength=len(lens))
    new_ip = np.concatenate([[0], np.cumsum(counts)]).astype(ip.dtype)
    return sp.csr_matrix((dt[mask], Q.indices[mask], new_ip), shape=Q.shape)


def tfidf_topk(q_text, t_text, k, analyzer, max_df, keep=None, q_chunk=200_000):
    """Top-k cosine neighbours as a CSR matrix (row = query, sorted desc).

    Queries are multiplied in chunks so the transient product for 1.7M
    production queries stays bounded; results are identical to one pass.
    """
    t0 = time.time()
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=(3, 3), min_df=2,
                          max_df=max_df, sublinear_tf=True, dtype=np.float32)
    T = vec.fit_transform(t_text)
    Q = vec.transform(q_text)
    if keep:
        Q = prune_query_rows(Q, keep)
    TT = T.T.tocsr()
    del T
    t1 = time.time()
    parts = []
    for s in range(0, Q.shape[0], q_chunk):
        parts.append(sp_matmul_topn(Q[s:s + q_chunk], TT, top_n=k, sort=True,
                                    n_threads=THREADS))
    R = sp.vstack(parts).tocsr() if len(parts) > 1 else parts[0].tocsr()
    t2 = time.time()
    log(f"      {analyzer} top-{k} keep={keep}: vocab={len(vec.vocabulary_)} "
        f"queries={Q.shape[0]} (fit {t1-t0:.0f}s, search {t2-t1:.0f}s = "
        f"{1000*(t2-t1)/max(Q.shape[0],1):.2f} ms/query)")
    del TT, Q, vec, parts
    gc.collect()
    return R


def csr_row(R, i):
    """[(t_idx, cos, rank0), ...] for query i, best first."""
    a, b = R.indptr[i], R.indptr[i + 1]
    return [(int(j), float(c), rk) for rk, (j, c) in
            enumerate(zip(R.indices[a:b], R.data[a:b]))]


UNION_KEYS = ["c2_cos", "c2_rank", "c2b_cos", "c2b_rank", "c3_cos", "c3_rank",
              "c4", "c1_score", "c1_rank"]


def build_union(qidx, R, c4, c1_rows=None, track=False):
    """Union of every channel's candidates for the queries in qidx.

    Shared by the validation harness and production so the reranker sees
    identically constructed candidates and channel evidence in both.
    Returns (qi, ti, chan_arrays, membership_or_None).
    """
    qi_l, ti_l = [], []
    cols = {k: [] for k in UNION_KEYS}
    cols["n"] = []
    membership = defaultdict(lambda: defaultdict(set)) if track else None
    for qi in qidx:
        cands = {}
        for name in ("c2", "c2b", "c3"):
            if R.get(name) is not None:
                for ti, cos, rk in csr_row(R[name], qi):
                    d = cands.setdefault(ti, {})
                    d[f"{name}_cos"] = cos
                    d[f"{name}_rank"] = rk + 1
        for ti, nk in c4.get(qi, {}).items():
            cands.setdefault(ti, {})["c4"] = nk
        if c1_rows is not None:
            for ti, score, rk in c1_rows(qi):
                d = cands.setdefault(ti, {})
                d["c1_score"] = score
                d["c1_rank"] = rk + 1
        for ti, d in cands.items():
            qi_l.append(qi); ti_l.append(ti)
            for k in UNION_KEYS:
                cols[k].append(d.get(k, 0.0))  # 0 = not retrieved by this channel
            cols["n"].append(sum(1 for c in ("c2", "c2b", "c3") if f"{c}_cos" in d)
                             + ("c4" in d) + ("c1_rank" in d))
            if track:
                for c in ("c2", "c2b", "c3"):
                    if f"{c}_cos" in d:
                        membership[c][qi].add(ti)
                if "c4" in d:
                    membership["c4"][qi].add(ti)
                if "c1_rank" in d:
                    membership["c1"][qi].add(ti)
                    if d["c1_rank"] <= 60:
                        membership["c1@60"][qi].add(ti)
    chan = {k: np.asarray(v, np.float32) for k, v in cols.items()}
    return (np.asarray(qi_l, np.int64), np.asarray(ti_l, np.int64), chan, membership)


def exact_channel(Q, T):
    """{q_idx: {t_idx: n_key_types_matched}} over selective exact keys."""
    t0 = time.time()
    fields = [("name", lambda R, i: (R.name[i],)),
              ("sorted", lambda R, i: (R.sorted[i],)),
              ("phon", lambda R, i: (R.phonc[i],)),
              ("house", lambda R, i: tuple(R.house[i])),
              ("postal", lambda R, i: tuple(R.postal[i]))]
    out = defaultdict(lambda: defaultdict(int))
    for fname, get in fields:
        index = defaultdict(list)
        for i in range(T.n):
            for key in get(T, i):
                if key and len(key) >= 3:
                    index[key].append(i)
        for qi in range(Q.n):
            seen = set()
            for key in get(Q, qi):
                lst = index.get(key)
                if lst and len(lst) <= EXACT_MAX_DF:
                    for ti in lst:
                        if ti not in seen:
                            seen.add(ti)
                            out[qi][ti] += 1
        del index
    log(f"      exact keys ({time.time()-t0:.0f}s)")
    return out


def c1_channel(db_path, q_ids, top_k):
    """Original SQLite IDF blocking as an extra channel (slow: I/O bound)."""
    import blocking
    from db import connect
    t0 = time.time()
    conn = connect(db_path)
    out = {}
    for e, cands in blocking.generate_candidates(conn, top_k=top_k, s1_ids=q_ids,
                                                 progress_every=0):
        out[e] = [(c[0], float(c[2]), rk) for rk, c in enumerate(cands)]
    conn.close()
    log(f"      c1 sqlite top-{top_k} for {len(q_ids)} ({time.time()-t0:.0f}s)")
    return out


# -------------------------------------------------------------- features

def word_matrices(q_list, t_list):
    vec = CountVectorizer(analyzer=str.split, binary=True, dtype=np.float32)
    T = vec.fit_transform(t_list)
    Q = vec.transform(q_list)
    return Q.tocsr(), T.tocsr()


def build_word_mats(Q, T):
    """Word-incidence matrices + row sums for name and address, built once
    per country and reused by every chunk's pair_features call."""
    wm = {}
    for field in ("name", "addr"):
        Qw, Tw = word_matrices(getattr(Q, field), getattr(T, field))
        wm[field] = (Qw, Tw, np.asarray(Qw.sum(axis=1)).ravel(),
                     np.asarray(Tw.sum(axis=1)).ravel())
    return wm


def rowpair_dot(A, B, ai, bi, chunk=500_000):
    out = np.empty(len(ai), dtype=np.float32)
    for s in range(0, len(ai), chunk):
        e = s + chunk
        out[s:e] = np.asarray(A[ai[s:e]].multiply(B[bi[s:e]]).sum(axis=1)).ravel()
    return out


def pair_features(Q, T, qi, ti, chan, wm=None):
    """Vectorized features for pairs (qi[k], ti[k]).

    wm: precomputed build_word_mats(Q, T); built here if not given.
    """
    if wm is None:
        wm = build_word_mats(Q, T)
    n = len(qi)
    F = np.zeros((n, len(FEATURES)), dtype=np.float32)
    col = {f: j for j, f in enumerate(FEATURES)}

    def take(lst, idx):
        return [lst[i] for i in idx]

    qn, tn = take(Q.name, qi), take(T.name, ti)
    qa, ta = take(Q.addr, qi), take(T.addr, ti)
    qp, tp = take(Q.phon, qi), take(T.phon, ti)
    W = dict(workers=-1, dtype=np.float32)
    F[:, col["r_name"]] = cpdist(qn, tn, scorer=fuzz.ratio, **W)
    F[:, col["r_name_tsort"]] = cpdist(qn, tn, scorer=fuzz.token_sort_ratio, **W)
    F[:, col["r_name_tset"]] = cpdist(qn, tn, scorer=fuzz.token_set_ratio, **W)
    F[:, col["r_name_part"]] = cpdist(qn, tn, scorer=fuzz.partial_ratio, **W)
    F[:, col["r_phon"]] = cpdist(qp, tp, scorer=fuzz.token_set_ratio, **W)
    F[:, col["r_addr"]] = cpdist(qa, ta, scorer=fuzz.ratio, **W)
    F[:, col["r_addr_tset"]] = cpdist(qa, ta, scorer=fuzz.token_set_ratio, **W)

    for field, pre in (("name", "name"), ("addr", "addr")):
        Qw, Tw, qsum, tsum = wm[field]
        inter = rowpair_dot(Qw, Tw, qi, ti)
        qs = qsum[qi]
        ts = tsum[ti]
        union = qs + ts - inter
        F[:, col[f"{pre}_word_jac"]] = np.divide(inter, union, out=np.zeros(n, np.float32), where=union > 0)
        mn = np.minimum(qs, ts)
        F[:, col[f"{pre}_word_ovl"]] = np.divide(inter, mn, out=np.zeros(n, np.float32), where=mn > 0)
        F[:, col[f"n_{pre}_words"]] = inter

    ql = np.array([len(x) for x in qn], np.float32)
    tl = np.array([len(x) for x in tn], np.float32)
    mx = np.maximum(ql, tl)
    F[:, col["len_ratio"]] = np.divide(np.minimum(ql, tl), mx, out=np.zeros(n, np.float32), where=mx > 0)
    F[:, col["cand_source"]] = T.src[ti]

    for k in range(n):
        a, b = qi[k], ti[k]
        F[k, col["name_exact"]] = Q.name[a] == T.name[b] and Q.name[a] != ""
        F[k, col["sorted_exact"]] = Q.sorted[a] == T.sorted[b] and Q.sorted[a] != ""
        F[k, col["phon_exact"]] = Q.phonc[a] == T.phonc[b] and Q.phonc[a] != ""
        F[k, col["house_match"]] = bool(Q.house[a] & T.house[b])
        F[k, col["postal_match"]] = bool(Q.postal[a] & T.postal[b])
        F[k, col["s1_addr_empty"]] = Q.addr[a] == ""
        F[k, col["cand_addr_empty"]] = T.addr[b] == ""

    for name in ("c2", "c2b", "c3"):
        F[:, col[f"{name}_cos"]] = chan[f"{name}_cos"]
        F[:, col[f"{name}_rank"]] = chan[f"{name}_rank"]
    F[:, col["c4_keys"]] = chan["c4"]
    F[:, col["c1_score"]] = chan["c1_score"]
    F[:, col["c1_rank"]] = chan["c1_rank"]
    F[:, col["n_channels"]] = chan["n"]
    return F


# ------------------------------------------------------------ one country

def load_country(country, q_ids, s1rec, split_dir, split, cache_dir=None):
    """Target records are cached after normalization: normalizing ~5M records
    takes 10-12 min, reloading the pickle ~1 min."""
    t0 = time.time()
    cache = (os.path.join(cache_dir, f"records_{split}_{country}.pkl")
             if cache_dir else None)
    if cache and os.path.exists(cache):
        import pickle
        with open(cache, "rb") as f:
            T = pickle.load(f)
        log(f"  targets: {T.n} from cache ({time.time()-t0:.0f}s)")
    else:
        tids, tsrc, tnames, taddrs = load_targets(split_dir, split, country)
        log(f"  targets: {len(tids)} ({time.time()-t0:.0f}s)")
        T = Records(tids, tsrc, tnames, taddrs)
        del tnames, taddrs
        if cache:
            import pickle
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache + ".part", "wb") as f:
                pickle.dump(T, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(cache + ".part", cache)
            log(f"  cached targets -> {cache}")
    Q = Records(q_ids, [1] * len(q_ids), [s1rec[e][0] for e in q_ids],
                [s1rec[e][1] for e in q_ids])
    return Q, T


def retrieve(Q, T, channels, max_df):
    """Run every enabled channel once for all queries of a country."""
    R = {}
    if "c2" in channels:
        R["c2"] = tfidf_topk(Q.name, T.name, TOPK["c2"], "char_wb", max_df, QKEEP.get("c2"))
    if "c2b" in channels:
        R["c2b"] = tfidf_topk(Q.phonc, T.phonc, TOPK["c2b"], "char", max_df, QKEEP.get("c2b"))
    if "c3" in channels:
        R["c3"] = tfidf_topk(Q.addr, T.addr, TOPK["c3"], "char_wb", max_df, QKEEP.get("c3"))
    c4 = exact_channel(Q, T) if "c4" in channels else {}
    return R, c4


def run_country(country, q_ids, s1rec, split_dir, split, args, c1_all):
    log(f"\n=== {country}: {len(q_ids)} queries ===")
    t0 = time.time()
    Q, T = load_country(country, q_ids, s1rec, split_dir, split,
                        os.path.join(args.out_dir, "cache"))
    tpos = {e: i for i, e in enumerate(T.ids)}
    R, c4 = retrieve(Q, T, args.channels, args.max_df)

    def c1_rows(qi):
        out = []
        for cid, score, rk in c1_all.get(q_ids[qi], ()):
            ti = tpos.get(cid)
            if ti is not None:
                out.append((ti, score, rk))
        return out

    qi_a, ti_a, chan, membership = build_union(range(Q.n), R, c4,
                                               c1_rows if c1_all else None, track=True)
    log(f"  union: {len(qi_a)} pairs ({len(qi_a)/max(len(q_ids),1):.1f}/query)")

    t1 = time.time()
    F = pair_features(Q, T, qi_a, ti_a, chan, build_word_mats(Q, T))
    log(f"  features: {F.shape} ({time.time()-t1:.0f}s)")

    result = {
        "s1": np.asarray([q_ids[i] for i in qi_a]),
        "cand": np.asarray([T.ids[i] for i in ti_a]),
        "F": F,
        "membership": {c: {q_ids[qi]: {T.ids[t] for t in s} for qi, s in m.items()}
                       for c, m in membership.items()},
    }
    del T, Q, R, c4
    gc.collect()
    log(f"  {country} done ({time.time()-t0:.0f}s)")
    return result


# ------------------------------------------------------------ evaluation

def retrieval_report(name, ids, gt, cand_sets):
    tot = found = full = none = with_t = 0
    sizes = []
    for e in ids:
        c = cand_sets.get(e, set())
        sizes.append(len(c))
        t = gt.get(e, set())
        if not t:
            continue
        with_t += 1
        f = len(t & c)
        tot += len(t); found += f
        full += f == len(t)
        none += f == 0
    s = np.asarray(sizes)
    log(f"  {name:<8} pair recall {found/max(tot,1):.4f} | all-found "
        f"{full/max(with_t,1):.4f} | none-found {none/max(with_t,1):.4f} | "
        f"cands mean {s.mean():.1f} med {np.median(s):.0f} p90 {np.percentile(s,90):.0f} "
        f"p99 {np.percentile(s,99):.0f} max {s.max()}")


def decide_expf(pairs, r, s, floor):
    pairs = [(p, c) for p, c in pairs if p >= floor]
    if not pairs:
        return []
    probs = [p for p, _ in pairs]
    T = sum(probs) / r
    p_none = s
    for p in probs:
        p_none *= (1.0 - p)
    best_m, best_v, run = 0, p_none, 0.0
    for m, p in enumerate(probs, start=1):
        run += p
        v = 1.25 * run / (0.25 * T + m)
        if v > best_v:
            best_m, best_v = m, v
    return [c for _, c in pairs[:best_m]]


def exclusive(per, decide):
    dec = {e: decide(ps) for e, ps in per.items()}
    prob = {e: {c: p for p, c in ps} for e, ps in per.items()}
    owner = {}
    for e, cs in dec.items():
        for c in cs:
            p = prob[e][c]
            if c not in owner or p > owner[c][0]:
                owner[c] = (p, e)
    return {e: [c for c in cs if owner[c][1] == e] for e, cs in dec.items()}


def final_report(ids, gt, per):
    configs = []
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        for top_n in (None, 4, 6, 8):
            configs.append((f"thresh thr={thr} top_n={top_n}",
                            lambda ps, thr=thr, top_n=top_n:
                            [c for p, c in ps if p >= thr][:top_n] if top_n
                            else [c for p, c in ps if p >= thr]))
    for r in (0.6, 0.75, 0.9, 1.0):
        for s in (0.4, 0.7, 1.0):
            for fl in (0.05, 0.2):
                configs.append((f"expf r={r} s={s} floor={fl}",
                                lambda ps, r=r, s=s, fl=fl: decide_expf(ps, r, s, fl)))
    results = []
    for name, dec in configs:
        for excl in (False, True):
            preds = exclusive(per, dec) if excl else {e: dec(ps) for e, ps in per.items()}
            tot = p_sum = r_sum = 0.0
            n_pr = 0
            for e in ids:
                t, pr = gt.get(e, set()), set(preds.get(e, []))
                tot += f_beta_one(t, pr)
                if t or pr:
                    tp = len(t & pr)
                    p_sum += tp / len(pr) if pr else 0.0
                    r_sum += tp / len(t) if t else 0.0
                    n_pr += 1
            results.append((tot / len(ids), p_sum / max(n_pr, 1), r_sum / max(n_pr, 1),
                            f"{name} exclusive={excl}"))
    results.sort(key=lambda x: -x[0])
    log("\n  FINAL — top 10 decision configs on validation:")
    for f, p, r, name in results[:10]:
        log(f"    F0.5 {f:.4f}  P {p:.4f}  R {r:.4f}   {name}")
    return results[0]


# ------------------------------------------------------------------ main

# ------------------------------------------------------- two-stage reranker
#
# A pairwise model scores each candidate in isolation. But the question that
# actually decides the score is comparative: of this business's ~150
# candidates, which ones are it? A same-name chain store and the true match
# can look alike in isolation and very different side by side. Stage 2 sees
# each candidate's stage-1 score relative to its rivals for the same entity.

CTX_FEATURES = ["p1", "p1_rank", "p1_rel", "p1_gap_other",
                "ent_cands", "ent_n_hi", "ent_sum"]


def context_features(group, p1):
    """Per-pair features comparing p1 to the other candidates of the same
    entity. group: int entity code per pair; p1: stage-1 probabilities."""
    n = len(p1)
    if n == 0:
        return np.zeros((0, len(CTX_FEATURES)), np.float32)
    p1 = np.asarray(p1, np.float32)
    group = np.asarray(group)
    order = np.lexsort((-p1, group))
    gs, ps = group[order], p1[order]
    starts = np.r_[0, np.nonzero(gs[1:] != gs[:-1])[0] + 1]
    sizes = np.diff(np.r_[starts, n])
    top1 = ps[starts]
    top2 = np.where(sizes > 1, ps[np.minimum(starts + 1, n - 1)], 0.0)
    rank_s = np.arange(n) - np.repeat(starts, sizes) + 1
    gid_s = np.repeat(np.arange(len(starts)), sizes)
    csum = np.add.reduceat(ps, starts)
    nhi = np.add.reduceat((ps > 0.5).astype(np.float32), starts)
    inv = np.empty(n, np.int64)
    inv[order] = np.arange(n)
    gid, rank = gid_s[inv], rank_s[inv]
    t1, t2 = top1[gid], top2[gid]
    rel = np.divide(p1, t1, out=np.zeros(n, np.float32), where=t1 > 0)
    gap = np.where(rank == 1, p1 - t2, p1 - t1)
    return np.column_stack([p1, rank, rel, gap, sizes[gid], nhi[gid],
                            csum[gid]]).astype(np.float32)


LGB_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
                  num_leaves=127, min_data_in_leaf=40, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1, verbose=-1, seed=42,
                  num_threads=THREADS)


def train_lgb(X, y, names):
    import lightgbm as lgb
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    nd = max(len(idx) // 10, 1)
    dtr = lgb.Dataset(X[idx[nd:]], y[idx[nd:]], feature_name=names)
    ddv = lgb.Dataset(X[idx[:nd]], y[idx[:nd]], reference=dtr)
    m = lgb.train(LGB_PARAMS, dtr, 3000, valid_sets=[ddv],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def scored_lists(s1, cand, p, ids):
    per = defaultdict(list)
    for e, c, pr in zip(s1, cand, p):
        per[e].append((float(pr), c))
    for v in per.values():
        v.sort(reverse=True)
    for e in ids:
        per.setdefault(e, [])
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", default="work/mc")
    ap.add_argument("--n-val", type=int, default=40000)
    ap.add_argument("--n-fit", type=int, default=30000)
    ap.add_argument("--channels", default="c2,c2b,c3,c4")
    ap.add_argument("--db", default=None, help="Enable channel c1 using this SQLite DB.")
    ap.add_argument("--c1-topk", type=int, default=100)
    ap.add_argument("--max-df", type=float, default=0.02)
    ap.add_argument("--from-cache", action="store_true",
                    help="Reuse cached union+features; only retrain/re-evaluate the reranker.")
    args = ap.parse_args()
    args.channels = set(args.channels.split(","))
    if args.db:
        args.channels.add("c1")
    os.makedirs(args.out_dir, exist_ok=True)
    split_dir = os.path.join(args.dataset_dir, args.split)
    T0 = time.time()

    val, fit = select_entities(os.path.join(split_dir, f"{args.split}_source1.tsv"),
                               args.n_val, args.n_fit)
    want = set(val) | set(fit)
    s1rec = load_s1(os.path.join(split_dir, f"{args.split}_source1.tsv"), want)
    gt = load_gt(os.path.join(split_dir, f"{args.split}_ground_truth.tsv"), want)
    log(f"val {len(val)}, fit {len(fit)}, channels {sorted(args.channels)}")

    cache = os.path.join(args.out_dir, "cache", "union_features.npz")
    if args.from_cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        s1, cand, F = z["s1"], z["cand"], z["F"]
        membership = z["membership"].item()
        by_country = z["by_country"].item()
        log(f"loaded {len(s1)} union pairs from cache (retrieval skipped)")
    else:
        c1_all = c1_channel(args.db, sorted(want), args.c1_topk) if args.db else {}
        by_country = defaultdict(list)
        for e in sorted(want):
            by_country[s1rec[e][2]].append(e)
        by_country = dict(by_country)
        parts = []
        for country, q_ids in sorted(by_country.items()):
            parts.append(run_country(country, q_ids, s1rec, split_dir, args.split, args, c1_all))
        s1 = np.concatenate([p["s1"] for p in parts])
        cand = np.concatenate([p["cand"] for p in parts])
        F = np.vstack([p["F"] for p in parts])
        membership = {}
        for p in parts:
            for ch, m in p["membership"].items():
                membership.setdefault(ch, {}).update(m)
        del parts
        gc.collect()
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, s1=s1, cand=cand, F=F,
                 membership=np.array(membership, dtype=object),
                 by_country=np.array(by_country, dtype=object))
        log(f"cached union features -> {cache}")

    y = np.fromiter((c in gt.get(e, ()) for e, c in zip(s1, cand)), dtype=np.int8, count=len(s1))
    union = defaultdict(set)
    for e, c in zip(s1, cand):
        union[e].add(c)
    log(f"\n=== RETRIEVAL on {len(val)} validation entities ===")
    for ch in ("c1@60", "c1", "c2", "c2b", "c3", "c4"):
        if ch in membership:
            retrieval_report(ch, val, gt, membership[ch])
    retrieval_report("UNION", val, gt, union)
    for country in sorted(by_country):
        cv = [e for e in val if s1rec[e][2] == country]
        if cv:
            retrieval_report(f"U:{country[:5]}", cv, gt, union)
    # Leave-one-out: how much union recall each channel uniquely contributes.
    chans = [c for c in ("c1", "c2", "c2b", "c3", "c4") if c in membership]
    log("  leave-one-out (union recall WITHOUT the channel):")
    for c in chans:
        rest = defaultdict(set)
        for o in chans:
            if o != c:
                for e, s in membership[o].items():
                    rest[e] |= s
        tot = found = 0
        for e in val:
            t = gt.get(e, set())
            tot += len(t); found += len(t & rest.get(e, set()))
        log(f"    without {c:<4} {found/max(tot,1):.4f}")

    is_val = np.isin(s1, np.asarray(val))
    fit_m = ~is_val
    codes = np.unique(s1, return_inverse=True)[1]
    t1 = time.time()

    # Stage 1: out-of-fold scores for training entities (so stage 2 learns
    # from scores it will see at inference, not overfit in-sample ones).
    p1 = np.zeros(len(F), np.float32)
    fold = codes % 3
    for k in range(3):
        tr, te = fit_m & (fold != k), fit_m & (fold == k)
        mk = train_lgb(F[tr], y[tr], FEATURES)
        p1[te] = mk.predict(F[te], num_iteration=mk.best_iteration)
    m1 = train_lgb(F[fit_m], y[fit_m], FEATURES)
    p1[is_val] = m1.predict(F[is_val], num_iteration=m1.best_iteration)
    log(f"\nstage 1: {m1.best_iteration} rounds, {int(fit_m.sum())} training pairs "
        f"(pos rate {y[fit_m].mean():.3f}) ({time.time()-t1:.0f}s)")

    per1 = scored_lists(s1[is_val], cand[is_val], p1[is_val], val)
    log("\n--- STAGE 1 (pairwise only) ---")
    best1 = final_report(val, gt, per1)

    t2 = time.time()
    F2 = np.hstack([F, context_features(codes, p1)])
    m2 = train_lgb(F2[fit_m], y[fit_m], FEATURES + CTX_FEATURES)
    p2 = m2.predict(F2[is_val], num_iteration=m2.best_iteration)
    log(f"\nstage 2: {m2.best_iteration} rounds ({time.time()-t2:.0f}s)")
    per2 = scored_lists(s1[is_val], cand[is_val], p2, val)
    log("\n--- STAGE 2 (with rival-candidate context) ---")
    best2 = final_report(val, gt, per2)

    two_stage = best2[0] > best1[0]
    best = best2 if two_stage else best1
    m1.save_model(os.path.join(args.out_dir, "mc_m1.txt"), num_iteration=m1.best_iteration)
    m2.save_model(os.path.join(args.out_dir, "mc_m2.txt"), num_iteration=m2.best_iteration)
    mb = m2 if two_stage else m1
    names = FEATURES + CTX_FEATURES if two_stage else FEATURES
    imp = sorted(zip(names, mb.feature_importance("gain")), key=lambda x: -x[1])
    with open(os.path.join(args.out_dir, "mc_report.json"), "w") as f:
        json.dump({"best_f05": best[0], "best_precision": best[1], "best_recall": best[2],
                   "best_config": best[3], "two_stage": bool(two_stage),
                   "stage1_f05": best1[0], "stage2_f05": best2[0],
                   "features": FEATURES, "ctx_features": CTX_FEATURES,
                   "importance": [(k, float(v)) for k, v in imp],
                   "channels": sorted(c for c in args.channels if c != "c1"),
                   "topk": TOPK, "qkeep": QKEEP, "max_df": args.max_df}, f, indent=2)
    log("\n  top features by gain: " + ", ".join(k for k, _ in imp[:10]))
    log(f"\nTOTAL {time.time()-T0:.0f}s  |  stage 1 {best1[0]:.4f}  |  stage 2 {best2[0]:.4f}"
        f"  |  USING {'stage 2' if two_stage else 'stage 1'}: {best[0]:.4f}  "
        f"(previous run 0.8883, old architecture 0.7930)")
    log("Model files for the other laptop: mc_m1.txt, mc_m2.txt, mc_report.json "
        f"in {args.out_dir}")



if __name__ == "__main__":
    main()
