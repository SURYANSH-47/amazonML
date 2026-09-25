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

**Core Innovation:** A SQLite-backed blocking layer that expresses candidate
generation as indexed SQL joins with block-purging of overly common keys,
so peak Python memory stays bounded regardless of source-table size — this
is what makes the pipeline runnable end-to-end on a ~4GB-RAM machine without
subsampling the production data, plus tuning the decision threshold *and* an
optional per-entity top-N cap directly against the official macro F_0.5
metric (rather than a generic classification threshold like 0.5).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** inverted index over four key types per record —
  normalized-name tokens, normalized-address tokens, digit tokens extracted
  from the raw address (house/unit/PIN/zip-like numbers), a 4-character name
  prefix, and the full normalized name as a high-signal exact-match key.
  Every key is scoped to an exact `country` match.
- **Block purging:** key values whose Source-2/3 frequency exceeds a cap
  (default 200; 2000 for the full-name key, since an exact whole-name match
  is strong evidence even if moderately frequent) are dropped before
  joining. This bounds worst-case per-key join cost independent of corpus
  size — the standard block-purging technique — and is what keeps blocking
  runtime near-linear in the number of Source-1 entities rather than
  exploding on generic tokens ("inc", common city names, etc.).
- **Ranking / cap:** candidates are scored by a weighted sum of matching key
  types (digit-token and full-name matches weighted higher than a single
  common name/address token) and capped at the top **25** per Source-1
  entity — directly targeting the challenge's "smaller candidate set scores
  higher" criterion while protecting recall.
- **Candidate pairs generated:** [fill in from the full run — see
  `output/candidate_pairs.tsv` summary stats]
- **How true matches were not lost:** validated blocking recall on a
  correlated sample (20k Source-1 entities plus every one of their true
  Source-2/3 matches, pulled by streaming the full training files) before
  scaling up: **93.95% pair-level recall** (fraction of true Source-1↔match
  pairs present in the candidate set) and **84.2% entity-level full-recall**
  (fraction of Source-1 entities whose candidate set contains *all* of their
  true matches) at top-K=25. This recall ceiling upper-bounds what the
  matching model can achieve downstream; remaining misses are cases with no
  address overlap and only generic, purged name tokens in common (a genuine
  hard case for pure lexical blocking).

---

## 4. Matching Model

**Features used** (`src/features.py`, ~17 features per pair):
- Name features: token Jaccard, token overlap coefficient, `rapidfuzz`
  Levenshtein ratio, token-sort ratio, token-set ratio, partial ratio,
  4-char prefix exact match, full normalized-name exact match, name length
  ratio.
- Address features: token Jaccard, token overlap coefficient, `rapidfuzz`
  Levenshtein ratio, digit-token (house/PIN/zip-like number) Jaccard,
  missing-address indicators for both sides.
- Other: the blocking-stage weighted score (captures key-type evidence not
  fully reducible to string metrics), candidate source (2 vs 3).

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

- **F_0.5 Score (macro), held-out validation:** [fill in from the full run —
  see `models/config.json` for the exact number, threshold, and top_n chosen]
- **Blocking recall ceiling on held-out validation:** [fill in]
- **Common false positives (wrong merges):** [fill in after error analysis on
  the full run — expected pattern from the sample run: near-duplicate chain
  businesses with generic names and overlapping cities]
- **Common false negatives (missed matches):** cases with an empty/sparse
  address on one side *and* only generic (purged) name tokens in common —
  the blocking stage's inherent hard case; see Section 3.

---

## 6. Conclusion

[Fill in after the full run: summarize the achieved macro F_0.5, the
candidate-set-size vs. recall trade-off actually realized, and the main
lesson — e.g., that country-scoped, block-purged lexical blocking plus a
lightweight tree classifier gets most of the way on this data without any
pretrained embedding model, provided normalization handles script mismatch
and open-set country/locale variation.]

---

## Appendix

### A. Code Artefacts

Complete, runnable code ships under `code/business_entity_resolution/`
(`src/` for all source, `README.md` for exact reproduce steps,
`requirements.txt` for pinned dependencies). Entry points:

1. `src/db.py` — stream a source TSV split into a SQLite DB with normalized
   fields and blocking keys.
2. `src/train.py` — build labeled candidate pairs (blocking ∩ ground truth),
   train the LightGBM matcher, tune threshold/top-N on held-out validation.
3. `src/predict.py` — stream blocking → features → model over a DB, writing
   `output/candidate_pairs.tsv` and `output/matching_results.tsv` in one pass.

`src/pipeline.py` wraps all three as `build-db` / `train` / `predict`
subcommands. See the README for the exact commands used to produce the
submitted outputs.

### B. Additional Results

[Fill in: candidate-set size distribution (mean/median/p95) from the full
test run, precision/recall at the chosen threshold, any additional error
analysis.]

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
