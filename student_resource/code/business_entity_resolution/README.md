# Business Entity Resolution — Pipeline

Matches Source-1 (deduplicated reference) businesses against Source-2 and
Source-3 records across any set of countries (training covers US and India;
test adds France). No external data, lookups, or pretrained models — every
transform is deterministic and every model is trained from scratch on the
provided data.

The submission was produced by the **multi-channel pipeline** (`mc_*`):
four independent retrieval channels unioned into a high-recall candidate
set, a two-stage LightGBM reranker, and an expected-F0.5 decision rule under
a one-owner constraint. Full methodology: `../../Documentation_template.md`.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9+. Run all commands from `student_resource/` (the folder containing
`dataset/`). `--dataset-dir` must point to the folder that contains the
`train/` and `test/` subfolders.

## Reproduce the submission

```bash
PIPELINE_DIR=code/business_entity_resolution

# 1. Validate on 40k held-out training entities and train the reranker.
#    Prints per-channel and union retrieval recall, then precision / recall /
#    macro F0.5 for stage 1 and stage 2, and writes mc_m1.txt, mc_m2.txt and
#    mc_report.json (models + the best decision rule) to work/mc.
python $PIPELINE_DIR/src/mc_experiment.py --dataset-dir dataset \
    --out-dir work/mc --n-val 40000 --n-fit 30000

# 2. Production over the test set. Reads the models and decision rule from
#    mc_report.json; runs France -> US -> India; merges and verifies that every
#    required entity appears exactly once.
python $PIPELINE_DIR/src/mc_predict.py --dataset-dir dataset --split test \
    --report work/mc/mc_report.json --out-dir output/mc

# 3. Validate the submission files.
python utils/validate_submission.py \
    --matching output/mc/matching_results.tsv \
    --candidate output/mc/candidate_pairs.tsv \
    --test-dir dataset/test
```

Measured on the full training corpus (40k held-out entities): union
retrieval recall **0.971**, macro F0.5 **0.898**.

## Running on two machines

Production splits cleanly by country and every country's output is written
atomically, so work can be divided between machines and resumed after a
crash (re-running the same command skips finished countries).

```bash
# Machine B, while machine A trains (no model needed): pre-normalize records
python $PIPELINE_DIR/src/mc_predict.py --dataset-dir dataset --split test \
    --prep-only --countries US,France

# Copy mc_m1.txt, mc_m2.txt, mc_report.json from A's work/mc to B's work/mc,
# then run a subset of countries on each machine:
python $PIPELINE_DIR/src/mc_predict.py --dataset-dir dataset --split test \
    --report work/mc/mc_report.json --out-dir output/mc --countries India,US

# Collect every cand_<country>.tsv / match_<country>.tsv into one output/mc,
# then merge:
python $PIPELINE_DIR/src/mc_predict.py --dataset-dir dataset --split test \
    --report work/mc/mc_report.json --out-dir output/mc --merge-only
```

## Iterating on the reranker

Retrieval is the slow part; the reranker is not. `mc_experiment.py` caches
normalized records and the union features, so the reranker can be retrained
and re-evaluated in minutes without re-running retrieval:

```bash
python $PIPELINE_DIR/src/mc_experiment.py --dataset-dir dataset \
    --out-dir work/mc_v2 --cache-dir work/mc/cache --from-cache [--ctx-extra]
```

## Module map (multi-channel pipeline)

| File | Responsibility |
| --- | --- |
| `src/mc_normalize.py` | Name/address normalization: transliteration, legal suffixes, DBA/honorific/domain stripping, compound house numbers, phonetic skeleton |
| `src/mc_experiment.py` | Retrieval channels (TF-IDF top-k + exact keys), candidate union, vectorized pair features, two-stage reranker, held-out evaluation |
| `src/mc_predict.py` | Production over the test set: per country, chunked, resumable, merged and verified |
| `src/evaluate.py` | Official macro F0.5 scorer |

## Earlier pipeline (kept for reference)

`db.py`, `blocking.py`, `features.py`, `train.py`, `predict.py`, `assign.py`,
`conflicts.py`, `check_recall.py`, `merge_shards.py`, `upgrade_db.py`,
`pipeline.py` implement the first-generation SQLite-backed IDF meta-blocking
pipeline, which produced the 0.767 leaderboard submission. Its retrieval
reached 76% recall; the multi-channel pipeline replaced that layer entirely.
