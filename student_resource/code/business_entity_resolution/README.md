# Business Entity Resolution — Pipeline

Matches Source-1 (deduplicated reference) business records against Source-2
and Source-3 records across US / India / France (and any other open-set
country label), using SQLite-backed blocking + a LightGBM classifier over
string-similarity features. No pretrained/foundation model, no external data
lookups — every normalization step (accent folding, script transliteration
via `unidecode`'s bundled tables, legal-suffix stripping) is a deterministic,
offline transform.

## Why this design

The full dataset is much larger than a laptop's RAM: ~2.2M Source-1 and ~5M
each Source-2/3 records for training, similar for test, on a machine with
~4GB total RAM. Every stage therefore streams rows (never loads a full TSV
into pandas) and stores normalized records + an inverted-index "blocking
keys" table in **SQLite**, so candidate generation is expressed as indexed
SQL joins that SQLite pages to/from disk itself instead of Python holding
millions of rows in memory. See `../../Documentation_template.md` (filled in)
for the full methodology write-up.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. Needs `dataset/train/` and `dataset/test/` (the
challenge's TSVs) available relative to wherever you run these commands —
the examples below assume you run from `student_resource/` (one level above
`code/`), with `code/business_entity_resolution` as `PIPELINE_DIR`.

## Reproduce end-to-end

All commands below are run from `student_resource/`.

```bash
PIPELINE_DIR=code/business_entity_resolution

# 1. Build SQLite DBs (streams the TSVs; ~12.5M rows train, ~11.7M rows test)
python $PIPELINE_DIR/src/db.py --dataset-dir dataset --db work/train.db --split train
python $PIPELINE_DIR/src/db.py --dataset-dir dataset --db work/test.db  --split test

# 2. Train: builds labeled candidate pairs from an entity-level train/validation
#    split of train.db, trains LightGBM, tunes the probability threshold (and
#    an optional per-entity top-N cap) to maximize macro F_0.5 on the held-out
#    validation entities. Saves model + config to $PIPELINE_DIR/models/.
python $PIPELINE_DIR/src/train.py \
    --db work/train.db \
    --ground-truth dataset/train/train_ground_truth.tsv \
    --models-dir $PIPELINE_DIR/models

# 3. Predict: runs blocking + the trained model over test.db in streaming
#    batches, writing both required output files in one pass.
python $PIPELINE_DIR/src/predict.py \
    --db work/test.db \
    --models-dir $PIPELINE_DIR/models \
    --candidate-out output/candidate_pairs.tsv \
    --matching-out output/matching_results.tsv

# 4. Validate the outputs before submitting
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

`src/pipeline.py` wraps the same three stages as subcommands
(`build-db` / `train` / `predict`) if you prefer a single entry point.

## Module map

| File | Responsibility |
| --- | --- |
| `src/normalize.py` | Text normalization: accent/script folding (NFKD + `unidecode`), legal-suffix stripping (US/India/France patterns), tokenization, digit-token extraction |
| `src/db.py` | Streams a source TSV into SQLite (`source1/2/3` tables + `blocking_keys` inverted index), batched inserts, indexes built after bulk load |
| `src/blocking.py` | Candidate generation: block-purges overly common keys, then per-entity-batch SQL joins scored by key-type weight, top-K per Source-1 entity |
| `src/features.py` | Pair-level features (token/character similarity via `rapidfuzz`, address/digit overlap, blocking score) |
| `src/train.py` | Entity-level train/validation split, builds labeled pairs, trains LightGBM, sweeps threshold + top-N cap for macro F_0.5 |
| `src/predict.py` | Streams blocking → features → model → both output TSVs in one pass over the test DB |
| `src/evaluate.py` | The official macro F_0.5 scorer, for our own held-out validation (test has no ground truth) |
| `src/pipeline.py` | CLI wrapper around the above |

## Key design choices

- **Blocking**: inverted-index over name tokens, address tokens, digit
  tokens pulled from the raw address (house/PIN/zip-like numbers), a 4-char
  name prefix, and the full normalized name (a high-signal exact-match key).
  Keys scoped to matching `country` — verified on a 69k true-match sample
  that country is 100% consistent between a Source-1 entity and its true
  matches. Overly common key values (generic tokens, common city names) are
  purged (default cap 200 occurrences, 2000 for the full-name key) before
  joining, which bounds worst-case join cost independent of corpus size.
  Candidates are ranked by a weighted key-type score and capped at the top-K
  (default 25) per Source-1 entity.
- **Matching model**: LightGBM binary classifier over ~17 features (token
  Jaccard/overlap, `rapidfuzz` Levenshtein/token-sort/token-set/partial
  ratios, address token & digit overlap, prefix/full-name exact match,
  blocking score). Trained from scratch on this data — not a downloaded
  foundation model — so it trivially satisfies the "MIT/Apache-2.0, ≤8B
  params" constraint.
- **Threshold + top-N tuning**: swept on a held-out validation split of
  Source-1 entities (never used for training labels) to directly maximize
  the challenge's macro F_0.5 metric, singletons included.
- **Memory safety**: `db.py` streams with `csv.reader` + batched inserts;
  `blocking.py` processes Source-1 entities in batches via a SQL join rather
  than loading the key index into Python; `train.py` subsamples entities
  (not pairs) to bound the in-memory training set; `predict.py` streams
  entity-batches straight to the output TSVs without ever holding the full
  test set's candidates/features in memory at once.
