"""Can a magnitude explanation be said in words? Score a verbal rule with no fitted weights.

The magnitude explanations (circuit/copy_task/autointerp_magnitude.py) fit a weight per term and use token tables
copied from fitted tables. Here an explanation is a rule written in words, turned into numbers using only the numbers
the words give (nothing fitted), and scored on the test split by rank correlation with activation minus timing
(scale-free, as in Towards Monosemanticity's simulation scoring). For comparison: the fitted agent explanation,
scored the same way. R^2 beyond timing with one fitted scale is also reported, only to compare with earlier numbers.

Feature 145. Rule as first stated: "builds up while c/3/t/2 arrive early in the sequence and decays over about 3 steps;
in the copy phase it dips below timing while those same tokens are being output". Then one phrase added at a time.

    python -m circuit.copy_task.verbal_rules --store features/<run>_store \\
        --magnitude_results features/<run>_magnitude --out verbal_rules_145.json
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from circuit.copy_task.autointerp import TOKEN_IDX
from circuit.copy_task.autointerp_magnitude import Mag


def rules_145(m):
    ip, c = m.ip, m.ip.cols
    t, L, ph = c["t"].astype(int), c["L"].astype(int), c["phase"].astype(int)
    copy = ph == 2
    S, Sm = [TOKEN_IDX[s] for s in "c3t2"], [TOKEN_IDX[s] for s in "bms0"]

    def build(tokens, decay, early=4):
        """Tokens of the set that arrived at positions < early, the last few steps, newer counting more."""
        return sum(w * (np.isin(ip.rel_tok(-d), tokens) & (t - d >= 0) & (t - d < np.minimum(L, early)))
                   for d, w in enumerate(decay))
    third = [1, 2 / 3, 1 / 3]                                            # "decays over about 3 steps"
    fed_back_S = copy & np.isin(ip.rel_tok(0), S)
    queue = copy * sum(np.isin(ip.out_at(d), S) for d in (1, 2, 3))      # S tokens still waiting (next 3 outputs)
    queue_m = copy * sum(np.isin(ip.out_at(d), Sm) for d in (1, 2, 3))
    return {
        "as stated: build-up (1, 2/3, 1/3) of early c/3/t/2, minus 1 per c/3/t/2 fed back in the copy phase":
            build(S, third) - 1.0 * fed_back_S,
        "  same, 'being output' read as the token output at this step":
            build(S, third) - 1.0 * (copy & np.isin(ip.out_at(0), S)),
        "  build-up only (no copy-phase dip)": build(S, third),
        "  no 'early' restriction": build(S, third, early=99) - 1.0 * fed_back_S,
        "+ b/m/s/0 push it down at half strength": build(S, third) - 0.5 * build(Sm, third),
        "+ copy phase: +1/2 per c/3/t/2 still waiting in the next 3 outputs": build(S, third) + 0.5 * queue,
        "+ both, the same for b/m/s/0 in the queue, and a 1/2 dip":
            build(S, third) - 0.5 * build(Sm, third) + 0.5 * (queue - 0.5 * queue_m) - 0.5 * fed_back_S,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--magnitude_results", required=True, help="agent magnitude records (hidden_<f>.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    feature = 145
    m = Mag(args.store, "hidden", feature)
    rows = m.rows["test"]
    y = m.act[rows]
    res = y - m.timing[rows]
    phase = m.ip.cols["phase"][rows]

    def score(p):
        p = np.asarray(p, dtype=np.float64)[rows]
        X = np.c_[p, np.ones_like(p)]
        a = np.linalg.lstsq(X, res, rcond=None)[0]
        out = {"spearman_beyond_timing": float(spearmanr(p, res).statistic),
               "r2_beyond_timing_one_scale": float(1 - ((res - X @ a) ** 2).sum() / (res ** 2).sum()),
               "spearman_activation": float(spearmanr(m.timing[rows] + p, y).statistic)}
        for k, name in ((0, "input"), (2, "copy")):
            sel = phase == k
            out[f"spearman_beyond_timing_{name}_phase"] = float(spearmanr(p[sel], res[sel]).statistic)
        return out

    terms = json.load(open(os.path.join(args.magnitude_results, f"hidden_{feature}.json")))["test_scores"]["terms"]
    _, X, w = m.fit(" ;; ".join(terms))
    results = {name: score(p) for name, p in rules_145(m).items()}
    results["fitted agent explanation (6 terms)"] = score(X @ w[:-1])
    share = {name: float((res[phase == k] ** 2).sum() / (res ** 2).sum()) for k, name in ((0, "input"), (1, "delimiter"), (2, "copy"))}
    for name, r in results.items():
        print(f"{name:100s} spearman {r['spearman_beyond_timing']:.2f}  R2 {r['r2_beyond_timing_one_scale']:.2f}  "
              f"input {r['spearman_beyond_timing_input_phase']:.2f}  copy {r['spearman_beyond_timing_copy_phase']:.2f}")
    print("share of what timing leaves unexplained, by phase:", {k: round(v, 2) for k, v in share.items()})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"feature": feature, "results": results, "residual_share_by_phase": share}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
