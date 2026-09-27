# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We resolve Source-1 business entities against Source-2/3 with a two-stage
pipeline: SQLite-backed inverted-index blocking (name/address/digit tokens,
country-scoped, with block-purging of uninformative keys) narrows ~10M
candidate records per source down to a small ranked top-K per Source-1
entity, and a from-scratch-trained LightGBM classifier over ~17 string-
similarity features makes the final call, with the probability threshold and
an optional per-entity match cap tuned directly against the macro F_0.5
metric on a held-out validation split. The whole pipeline streams data
end-to-end (never loading a full source file into memory) to run within this
machine's ~4GB RAM budget.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA surfaced several noise patterns not obvious from the problem statement
alone:

- **Scale**: the provided files are far larger than "billions of records"
  suggests in practice — ~2.2M Source-1 / ~5M Source-2 / ~5.3M Source-3 rows
  for training, similarly sized for test — but still far too large to load
  into pandas on a memory-constrained machine, which drove the SQLite-backed
  streaming design.
- **Script mismatch**: Source 1 names are always Latin/ASCII, but a material
  fraction of Source 2/3 India records use native scripts (Devanagari,
  Tamil, Kannada, ...) for the *same* business name, while addresses for
  those same records stay mostly Latin. This makes pure Latin-token
  comparison fail on name alone for these rows; we fold non-Latin scripts to
  a Latin approximation with `unidecode` (a bundled, deterministic
  transliteration table — no live lookups) so they become comparable.
- **Country is a very strong, very consistent signal**: on a 69k true-match
  sample, `country` matched between a Source-1 entity and every one of its
  true Source-2/3 matches 100% of the time. We use it as a hard blocking
  filter.
- **Legal-suffix and locale noise**: abbreviation variants (Corp/Corporation,
  Pvt/Private, Ltd/Limited), French suffixes not present in training
  (SARL/SASU/EURL) that appear in the test-only France segment, punctuation
  differences, and word-order changes are all handled by a shared
  normalization step (lowercase, accent/script fold, punctuation strip,
  legal-suffix removal) rather than per-country special-casing, since the
  problem statement explicitly requires treating `country` as an open set.
- **Missing addresses**: Source-1 addresses are never empty in the sampled
  data, but Source-3 has empty addresses on ~3% of rows — handled via
  missingness indicator features rather than dropped.
- **Ground truth shape**: only ~5.6% of Source-1 entities are singletons;
  non-singletons average ~3.5 matches, with a handful up to double digits —
  so the "many" case in the problem statement is common, not an edge case.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (candidate generation followed by a
supervised pairwise matcher), the standard architecture for large-scale
entity resolution.

**Core Innovation:** IDF-weighted meta-blocking with a *per-entity* cost
budget, implemented entirely as indexed SQL over SQLite so peak memory stays
bounded regardless of corpus size. Rather than discarding common keys
globally (which is what a first version did, and which capped recall at 54%
— see §5), every key is retained and weighted by rarity, while cost is
controlled per Source-1 entity by probing only its own most selective keys.
Combined with tuning the decision threshold *and* a per-entity top-N cap
directly against the official macro F_0.5 metric rather than a generic 0.5
cutoff.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used (9 types per record):** normalized-name tokens,
  normalized-address tokens, digit tokens from the raw address, postal-code
  tokens (5/6-digit runs, far more discriminative than house numbers), a
  4-character name prefix, the exact normalized name, the
  alphabetically-sorted name (makes word-order transposition a no-op), and
  phonetic skeletons of both the whole name and its individual tokens (these
  survive vowel-level typos and transliteration drift — `shivshakti` and
  `shivshakthi` collapse to the same key, which matters because Source 1 is
  always Latin while Source 2/3 India records are often in native script).
  Every key is scoped to an exact `country` match.

- **No purging — IDF weighting instead.** Keys are never deleted. A shared
  key contributes `key_type_prior × log(N/df)`, so a rare shared key is
  strong evidence while a common one still counts for something. This is
  what lets a true match survive on two moderately-common name tokens when
  one side has no address at all.

- **Cost control is per-entity, not global.** Each Source-1 entity probes
  using only its own most selective keys (rarest first) until a cumulative-df
  budget is spent — Block Filtering. Entities left candidate-poor are
  re-probed with a larger budget, capped at a fraction of each batch so the
  blended cost stays bounded. The only deletion performed is of keys with
  `df > 25,000`, which probing already refuses to touch; this is
  recall-neutral by construction and removed 24-27% of index rows, purely
  for cache locality.

- **Ranking / cap:** candidates ranked by accumulated IDF weight, capped at
  top-K per entity (Cardinality Node Pruning). K=60 for the final run.

- **Candidate pairs generated:** 1,732,544 Source-1 entities, ≤60 candidates
  each (see `output/candidate_pairs.tsv`).

- **How true matches were not lost:** measured directly with
  `src/check_recall.py`, which reports the recall ceiling in minutes and was
  run before every long job. On the full training corpus, final configuration:
  **75.5% pair-level recall**, up from 54% under the initial purging design.
  Measured cost/recall trade-off (full corpus, `--n 800`):

  | config | pair recall | throughput |
  | --- | --- | --- |
  | top_k=25, budget=400 | 65.4% | 25.3/s |
  | top_k=25, budget=1500 | 70.1% | 24.6/s |
  | top_k=60, budget=3000 | 76.8% | 9.1/s |
  | top_k=25, budget=15000 | 79.4% | 2.4/s |

  The residual misses are pairs sharing too little literal text for lexical
  blocking to reach at affordable cost — closing that gap needs learned
  embeddings with an ANN index rather than more tuning (see §6).

---

## 4. Matching Model

**Features used** (`src/features.py`, 23 features per pair):
- Name features: token Jaccard, token overlap coefficient, `rapidfuzz`
  Levenshtein ratio, token-sort ratio, token-set ratio, partial ratio,
  4-char prefix exact match, full normalized-name exact match, name length
  ratio.
- Address features: token Jaccard, token overlap coefficient, `rapidfuzz`
  Levenshtein ratio, digit-token (house/PIN/zip-like number) Jaccard,
  missing-address indicators for both sides.
- Blocking-derived: the accumulated IDF blocking score, exact match on the
  sorted-token name, exact match on the phonetic skeleton, phonetic-token
  Jaccard, exact postal-code match, and raw counts of shared name and
  address tokens.
- Other: candidate source (2 vs 3).

**Model type:** LightGBM binary classifier (gradient-boosted trees),
`num_leaves=31`, `learning_rate=0.05`, early stopping on a held-out dev
slice of the training pairs. Trained entirely from scratch on this
challenge's data — not a downloaded/pretrained foundation model — so it
trivially satisfies the "MIT/Apache-2.0 license, ≤8B parameters" constraint.

**Threshold selection method:** entity-level train/validation split (hashed,
not random, for reproducibility) of Source-1 entities so there is no
pair-level leakage. On the held-out validation entities, we sweep both the
probability threshold and an optional per-entity top-N cap on kept matches,
directly maximizing the official macro F_0.5 metric (`src/evaluate.py`,
matching the singleton-inclusive per-entity definition in the problem
statement) rather than optimizing a proxy metric like AUC or accuracy.

---

## 5. Results & Error Analysis

### Final results

| metric | value |
| --- | --- |
| **Leaderboard macro F_0.5** | **0.767** |
| Held-out validation macro F_0.5 | 0.7899 |
| Validation precision / recall | 0.8575 / 0.6561 |
| Blocking pair-recall ceiling (validation) | 0.7553 |
| Chosen operating point | threshold 0.6, top_n 6 |
| Training pairs | 3,580,933 (positive rate 4.4%) |
| Dev AUC | 0.9989 |

### The result that mattered most: a blocking design error

The first version used **block purging** — globally deleting any key whose
Source-2/3 frequency exceeded an absolute cap. On a 20k-entity sample this
measured 94% recall and looked fine. At full scale it collapsed to **54%**,
and that single number capped the leaderboard score at 0.641 no matter how
good the classifier was (dev AUC was already 0.999).

Two compounding causes:
1. The cap was **absolute**, tuned on a sample ~80x smaller than the real
   corpus. The same threshold is drastically more aggressive once document
   frequencies scale up — it removed 78% of all keys at full scale versus
   ~25% on the sample.
2. Frequency was computed **globally** rather than per-country, even though
   the join is country-scoped, so a key common in one country was deleted
   everywhere.

Deletion is also irrecoverable: an entity whose only keys happened to be
common was left with no candidates at all. Replacing purging with IDF
weighting plus per-entity cost budgeting took recall 54% → 75.5% and the
score 0.641 → 0.767.

**Lesson:** validate blocking recall at realistic corpus scale. A small
sample cannot expose a frequency-threshold bug, because the thing that
breaks *is* the frequency distribution. `src/check_recall.py` exists
precisely so this number is measured in minutes, before any multi-hour job.

### Error analysis

- **False positives (precision 0.858 — now the binding constraint).** Under
  F_0.5 a false merge costs roughly twice a miss, so this is where the
  remaining headroom is. The dominant pattern is same-name-different-location
  businesses — chains and generic trade names ("City Dental", "Sri Sai
  Enterprises") where name evidence is strong and address evidence is weak
  or missing. The sorted-token and phonetic keys, which bought recall, also
  pull in more of exactly this kind of near-duplicate.
- **False negatives.** Two distinct sources: ~24% of true pairs never reach
  the model at all (blocking ceiling), and the model discards a further
  ~10pp of what does reach it. The blocking misses are pairs sharing very
  little literal text — heavy transliteration, a genuinely different trade
  name, or an address present on only one side.
- **Country generalization.** Validation is necessarily train-only (US +
  India), while the test set is ~15% France. Validation predicted 0.79 and
  the leaderboard returned 0.767; the earlier submission tracked within
  0.3%. The most likely explanation is that the threshold was tuned on two
  countries and applied to a third that validation cannot observe.

---

## 6. Conclusion

Country-scoped lexical blocking plus a lightweight gradient-boosted
classifier reaches **0.767** macro F_0.5 on this data with no pretrained
model, provided normalization handles script mismatch and open-set locale
variation. The decisive lesson was not about modelling: the classifier was
near-ceiling (dev AUC 0.999) throughout, and every point of score came from
candidate generation. Hard block purging — discarding keys above an absolute
frequency cap — is the trap, because it is tuned on a distribution that
shifts with corpus size and destroys recall irrecoverably; replacing it with
IDF weighting and per-entity cost budgeting moved recall 54% → 75.5% and the
score 0.641 → 0.767.

With more time, the next gains in priority order: (1) **precision** (0.858),
which F_0.5 penalizes at double weight and which is dominated by
same-name-different-location chains — addressable with targeted features and
hard-negative mining, within the existing architecture; (2) **per-country
threshold tuning**, since the single global threshold is set by countries the
test set only partly shares; (3) **learned-embedding retrieval with an ANN
index** to reach the ~24% of true pairs that share too little literal text
for lexical keys — the only one of the three that requires a different
architecture rather than refinement of this one.

---

## Appendix

### A. Code Artefacts

Complete, runnable code ships under `code/business_entity_resolution/`
(`src/` for all source, `README.md` for exact reproduce steps,
`requirements.txt` for pinned dependencies). Entry points:

1. `src/db.py` — stream a source TSV split into a SQLite DB with normalized
   fields, blocking keys, key statistics and the materialized probe table.
2. `src/check_recall.py` — measure the blocking recall ceiling in minutes.
   Run before any long job; the model can never exceed this number.
3. `src/train.py` — build labeled candidate pairs (blocking ∩ ground truth),
   train the LightGBM matcher, tune threshold/top-N on held-out validation.
4. `src/predict.py` — blocking → features → model, writing both output TSVs.
   Supports `--shard i --num-shards N` for parallel execution.
5. `src/merge_shards.py` — merge shard outputs and verify every required
   Source-1 entity appears exactly once.
6. `src/upgrade_db.py` — add key stats / probe table to an existing DB
   without re-streaming the TSVs.

`src/pipeline.py` wraps build/train/predict as subcommands. See the README
for the exact commands used to produce the submitted outputs.

### B. Additional Results

**Scale.** Train 2,206,821 / 5,034,616 / 5,285,603 records (S1/S2/S3); test
1,732,544 / 4,887,273 / 5,082,316. 162.5M Source-2/3 blocking-key rows over
18.6M distinct keys before pruning; 139.8M after removing never-probed keys
(26.8%). 19.1M materialized probe keys across the test entities.

**Engineering constraints that shaped the design.** The pipeline was
developed on a machine with ~4GB RAM, so every stage streams and is
disk-backed; nothing loads a full source file into memory. Three measured
optimizations mattered:

| change | effect |
| --- | --- |
| `CROSS JOIN` to pin SQLite's join order | ~6x throughput (the planner otherwise scanned the 100M+ row table per batch) |
| Materializing per-entity probe keys | removed the fixed per-entity overhead that capped throughput at ~25/s regardless of budget |
| Pruning keys above the probe ceiling | 24-27% fewer index rows, recall-neutral by construction |
| Sharding predict across 8 processes | ~53h → ~8h wall clock for the full test set |

**Verification.** `utils/validate_submission.py` reports `PASS` on both
output files; `merge_shards.py` independently confirms all 1,732,544
required Source-1 entities are present exactly once.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
