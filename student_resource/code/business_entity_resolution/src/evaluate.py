"""Macro-averaged F_0.5 scorer matching the challenge's official definition.

Per Source-1 entity: precision/recall over predicted vs true matched-id sets,
F_0.5 = 1.0 for a correctly-predicted singleton (both empty), 0.0 for a false
merge on a true singleton, else the standard F-beta formula. Macro-averaged
over every required Source-1 entity (singletons included).
"""

BETA2 = 0.25  # beta=0.5 -> beta^2 = 0.25


def f_beta_one(true_ids: set, pred_ids: set) -> float:
    if not true_ids and not pred_ids:
        return 1.0
    if not pred_ids:  # true_ids non-empty, nothing predicted
        return 0.0
    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids) if true_ids else 0.0
    denom = BETA2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + BETA2) * precision * recall / denom


def macro_f_beta(required_ids, ground_truth: dict, predictions: dict):
    """required_ids: iterable of every Source-1 entity id that must be scored."""
    scores = []
    total_p = total_r = 0.0
    n_scored_pr = 0
    for s1 in required_ids:
        true_ids = ground_truth.get(s1, set())
        pred_ids = predictions.get(s1, set())
        scores.append(f_beta_one(true_ids, pred_ids))
        if pred_ids or true_ids:
            tp = len(true_ids & pred_ids)
            p = tp / len(pred_ids) if pred_ids else 0.0
            r = tp / len(true_ids) if true_ids else 0.0
            total_p += p
            total_r += r
            n_scored_pr += 1
    macro_f = sum(scores) / len(scores) if scores else 0.0
    mean_p = total_p / n_scored_pr if n_scored_pr else 0.0
    mean_r = total_r / n_scored_pr if n_scored_pr else 0.0
    return {
        "macro_f0.5": macro_f,
        "mean_precision": mean_p,
        "mean_recall": mean_r,
        "n_entities": len(scores),
    }


def read_id_list_tsv(path):
    """Read a matching_results.tsv / candidate_pairs.tsv / ground_truth.tsv-style file into {id: set(ids)}."""
    import csv

    out = {}
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if not row:
                continue
            s1 = row[0]
            ids = row[1].split(",") if len(row) > 1 and row[1] else []
            out[s1] = set(ids)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ground-truth", required=True)
    args = ap.parse_args()

    gt = read_id_list_tsv(args.ground_truth)
    pred = read_id_list_tsv(args.predictions)
    result = macro_f_beta(gt.keys(), gt, pred)
    print(result)
