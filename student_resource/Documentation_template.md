# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We resolve each Source-1 business against Source-2/3 with **multi-channel
retrieval followed by a two-stage learned reranker**. Four independent
retrieval channels — name character n-grams, a phonetic skeleton of the name
that bridges native-script and English spellings, address character n-grams
with compound house numbers preserved, and exact keys — are unioned into
~146 candidates per business, lifting the retrieval recall ceiling from 76%
to **97.1%**. A LightGBM reranker scores every candidate; a second stage then
re-scores each candidate *relative to its rivals for the same business*.
Final matches are chosen by an expected-F0.5 rule under a one-owner
constraint discovered in the ground truth. Held-out macro F0.5: **0.8978**,
up from 0.7930 for our first architecture (leaderboard 0.767 → **[final
leaderboard score]**).

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale and constraints.** ~2.2M Source-1 and ~10.3M Source-2/3 records for
  training, similar for test (1.73M Source-1 queries against ~10M records).
  Everything is streamed or held per country; nothing assumes the full
  corpus fits in memory.
- **Country is exact.** On a 69k true-match sample, a Source-1 entity and
  every one of its true matches share `country` 100% of the time, so every
  channel is scoped per country. `country` is treated as an open label set —
  the test-only France segment flows through the same code path.
- **One owner per record.** In the training ground truth, 0 of 7,638,365
  matched Source-2/3 records belong to more than one Source-1 entity. Any
  record claimed by two businesses is a certain error for all but one.
- **Retrieval, not the classifier, was the bottleneck.** Our first
  classifier already reached AUC 0.999; the score was capped by true matches
  never reaching it.
- **What the missed matches actually were.** Reading true pairs that our
  first retrieval never surfaced revealed three concrete failure classes:
  1. *Native-script names*: English business names written phonetically in
     Devanagari, Gujarati, Kannada or Tamil (`सुप्रीम इंजीनियरिंग एलएलपी` =
     "Supreme Engineering LLP"). Source 1 is always Latin.
  2. *Compound house numbers*: in Indian addresses the most distinctive
     field is the house/plot number (`6-3-10/3`, `D-2/2A`, `23-12-22`).
     A generic normalizer strips `-` and `/` and shreds these into common
     digits shared by thousands of records.
  3. *Replaced names*: some records carry a DBA wrapper, a domain
     (`smartit.com`), or an unrelated name (`Synnex`) and survive only via
     the address.
- **Two different retrieval failures.** At full scale, most misses were
  retrieved but *ranked* below lookalikes (top-60 recall 76%, top-300 85%);
  the rest were never retrieved at all. They need different fixes.

### 2.2 Solution Strategy

**Approach Type:** Hybrid — multi-channel candidate retrieval (unioned) +
two-stage learned reranker + constrained assignment.

**Core Innovation:** Retrieval channels designed around the specific
observed failure classes, unioned so each rescues what the others miss;
a second-stage reranker that judges each candidate against its rivals for
the same business; and one-owner assignment exploiting a structural
property of the data.

---

## 3. Candidate Generation (Blocking)

### Normalization (`src/mc_normalize.py`)
Deterministic, rule-based, no external data or lookups:
- Unicode NFKD + `unidecode` transliteration; zero-width joiners removed
  first (inside Indic words they otherwise become spurious word breaks).
- Legal suffixes stripped (US/India/France forms: Inc, LLC, Pvt, Ltd, SARL,
  SASU, EURL, ...); DBA wrappers resolved to the trade name; honorifics
  (`Mr`, `M/s`) and domain tokens (`www`, `com`) removed.
- Address abbreviations canonicalized (`Rd`→road, `St`→street); placeholders
  (`<NULL>`, `null`) dropped; compound house numbers re-appended as single
  tokens (`6-3-10/3` → `h6x3x10x3`) so they survive character n-grams.
- **Phonetic skeleton** of the name: soft g/c handled first (English
  "energy"/"center" use sounds Indic scripts write as ज/स), consonant
  clusters canonicalized, non-leading vowels dropped, and transliterated
  legal suffixes stripped at the phonetic level. On observed misses this
  raises cross-script name similarity e.g. *Prime Om Energy* 71→100,
  *Jain Infotech* 52→100, *Creative Investments* →100.

### Retrieval channels (`src/mc_experiment.py`)

| Channel | Method | Targets |
| --- | --- | --- |
| c2 | name char-3-gram TF-IDF, top-50 | typos (`lnfotech`), partial and domain names |
| c2b | phonetic-skeleton char-3-gram TF-IDF, top-40 | native script ↔ English |
| c3 | address char-3-gram TF-IDF, top-50 | replaced / garbage names, sparse addresses |
| c4 | exact keys: full / sorted / phonetic name, house no., postal | cheap near-certain hits |

Candidates from all channels are **unioned**, keeping each channel's score
and rank as reranker features. TF-IDF top-k is a sparse matrix product
(`sparse_dot_topn`), per country, in memory.

**Query-side n-gram pruning.** Each query searches with only its ~14–22
rarest n-grams. Search cost is the sum of the posting-list lengths of the
query's n-grams, dominated by common fragments (`del`, `mum`, `roa`) that
carry little identifying signal. Pruning cut search from a projected ~12h to
a few ms per query while changing union recall by only 97.10% → 97.06%.

### Measured retrieval (40,000 held-out Source-1 entities, full corpus)

| Channel | Pair recall | Entities with all matches | Entities with none |
| --- | --- | --- | --- |
| c2 name | 0.630 | 0.370 | 0.167 |
| c2b phonetic | 0.583 | 0.303 | 0.197 |
| c3 address | 0.875 | 0.688 | 0.021 |
| c4 exact | 0.631 | 0.322 | 0.114 |
| **UNION** | **0.971** | **0.913** | **0.003** |

India 0.960, US 0.978. For comparison, our first architecture reached 0.761
(top-60) with ~6.7% of entities having no true match retrieved at all.

**Leave-one-out** (union recall without the channel): c2 0.963, c2b 0.967,
**c3 0.799**, c4 0.963 — every channel contributes matches no other finds.

- **Candidate pairs generated:** mean ~146 per Source-1 entity (median 132,
  p90 209). The larger candidate set is deliberate: it is what moves the
  recall ceiling from 76% to 97%.
- **How true matches were not lost:** every retrieval change was measured
  on the held-out set at **full corpus scale** before use. A small-sample
  measurement misled us once — an early design measured 94% recall on 20k
  entities and 54% at full scale — so no retrieval decision here rests on a
  sample.

---

## 4. Matching Model

**Pair features (32)**, fully vectorized (`rapidfuzz.process.cpdist`,
sparse row products):
- Name: Levenshtein, token-sort, token-set and partial ratios; phonetic
  token-set ratio; word Jaccard / overlap / shared count; exact, sorted and
  phonetic equality; length ratio.
- Address: Levenshtein and token-set ratios; word Jaccard / overlap /
  shared count; compound house-number match; postal-code match; empty flags.
- Retrieval evidence: each channel's cosine and rank, exact-key count,
  number of channels that found the candidate.

**Two-stage reranker (LightGBM):**
- *Stage 1* scores each pair in isolation (4.37M training pairs, 30k
  training entities, disjoint from validation).
- *Stage 2* adds each candidate's standing among its rivals for the same
  business: stage-1 score, its rank, ratio to the best, margin over the best
  alternative, candidate count, number of strong candidates, total score.
  A same-name chain store and the true match can look alike in isolation
  and very different side by side. Stage 2 is trained on **out-of-fold**
  stage-1 scores (3 folds by entity) so it learns from the score
  distribution it will see at inference. The five most important stage-2
  features are all rival-comparison features.

**Decision rule and threshold selection.** Tuned directly on macro F0.5 on
the held-out set (never on AUC). Best: an **expected-F0.5 rule** — for each
business choose the number of matches m maximizing
`1.25·S_m / (0.25·T + m)` (S_m = sum of the top-m probabilities, T = expected
true-match count), predicting empty when a singleton is more likely — plus
**one-owner enforcement**: each record goes only to its highest-probability
claimant. (Chosen config: r=0.9, s=1.0, floor=0.2, exclusive.)

**Models:** LightGBM gradient-boosted trees trained from scratch on the
provided training data only — no pretrained or external model — well within
the MIT/Apache-2.0, ≤8B-parameter constraint.

---

## 5. Results & Error Analysis

| Version | Retrieval recall | Held-out F0.5 | Leaderboard |
| --- | --- | --- | --- |
| v1: SQLite blocking, hard key purging | 0.54 | 0.639 | 0.641 |
| v2: purge fix, stale config re-applied old cap | ~0.55 | — | 0.648 |
| v3: IDF meta-blocking, top-60 | 0.755 | 0.790 | 0.767 |
| **v4: multi-channel + two-stage reranker** | **0.971** | **0.898** | **[final]** |

Final held-out detail: precision 0.927, recall 0.825; stage 1 alone 0.8876,
stage 2 0.8978.

- **Where the remaining score is lost.** Retrieval now surfaces 97% of true
  matches, but final recall is 0.825 — the reranker/decision layer is the
  binding constraint, not retrieval.
- **Common false positives:** same-name businesses at different locations
  (chains, generic trade names) and different businesses sharing a building
  (the address channel surfaces every tenant of a commercial complex).
  One-owner enforcement removes the subset where the rightful owner is also
  a candidate; it is understated on validation, where only 70k of 2.2M
  entities compete, and fully active on the test set.
- **Common false negatives:** heavily perturbed pairs where the name is
  replaced *and* the address is sparse; Tamil/Gujarati transliterations
  where voicing differs (Tamil script does not distinguish b/p).

---

## 6. Conclusion

The decisive lesson was that the score was capped by retrieval, not by the
classifier, and that the fix came from reading the missed matches rather than
tuning: they fell into concrete classes — native-script names, shredded
compound house numbers, replaced names — each addressable by a dedicated
retrieval channel. Unioning those channels took retrieval recall from 76% to
97%, and a reranker that compares candidates against their rivals, plus the
one-owner constraint, turned that into a held-out F0.5 of 0.898. A second
lesson: every retrieval decision must be validated at full corpus scale,
because frequency-dependent behaviour does not show up on small samples.

---

## Appendix

### A. Code Artefacts

Runnable code is under `code/business_entity_resolution/` (`src/`,
`README.md`, `requirements.txt`). The submission was produced by the
multi-channel pipeline:

| File | Role |
| --- | --- |
| `src/mc_normalize.py` | normalization and phonetic skeleton |
| `src/mc_experiment.py` | retrieval channels, union, features, two-stage reranker training, held-out evaluation; writes `mc_m1.txt`, `mc_m2.txt`, `mc_report.json` |
| `src/mc_predict.py` | production over the test set: per-country, chunked, resumable; writes and verifies `candidate_pairs.tsv` and `matching_results.tsv` |
| `src/evaluate.py` | official macro F0.5 scorer for held-out evaluation |

The first-generation pipeline (`db.py`, `blocking.py`, `features.py`,
`train.py`, `predict.py`, `assign.py`, ...) is kept for reference; it
produced the 0.767 submission. Exact commands are in the README.

### B. Additional Results

**Runtime engineering.** Features are fully vectorized (~40–60k pairs/s);
retrieval is per-country sparse top-k with query pruning; normalized target
records and union features are cached, so the reranker can be retrained
from cache in minutes. Production runs one country at a time in chunks of
40k queries to bound memory, writes each country atomically, and resumes
after a crash without redoing finished countries.

**Earlier-architecture findings retained for the record.** Hard block purging
(global deletion of keys above an absolute frequency cap) collapsed recall
from 94% on a 20k sample to 54% at full scale — the cap was tuned on a
frequency distribution that shifts with corpus size, and deletion is
irrecoverable for an entity whose only keys were common. Replacing it with
IDF weighting and per-entity cost budgets gave v3 (0.767); the multi-channel
design (v4) replaced that retrieval layer entirely.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
