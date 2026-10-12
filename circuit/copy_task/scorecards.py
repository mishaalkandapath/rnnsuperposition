"""Fitted scorecards for features that no short predicate explains.

A scorecard predicts whether a feature fires from additive weights over one-hot
"cues" of the step (phase, position, the current / previous / output tokens,
each original token by position, and a few pairs). It complements the agent
predicates in circuit/copy_task/autointerp.py for features that respond to graded
combinations of cues. Per feature, three scorecards are fitted on the dev
split and scored on the held-out test split:

  full   L1-sparse scorecard over all cue blocks (fidelity reference)
  top40  the 40 most important cues of the full fit, refitted alone
  slot2  timing cues + graded weights over two token slots, chosen greedily

The readable format (top40 or slot2; slot2 if within 0.03 dev F1) gets a short
scripted summary, written into the feature's result JSON; all weights go to
<results>/scorecards/<gate>_<feature>.json.

    python -m circuit.copy_task.scorecards STORE RESULTS_DIR [--workers 3] [--features gate/feat ...]
"""
import argparse
import json
import os
import time
import warnings
from multiprocessing import get_context
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")

TOKCOLS = ["in_tok", "prev_tok", "prev2_tok", "prev3_tok", "out_tok", "next_out"] + [f"orig{k}" for k in range(9)]
SINGLE = [("phase", 0, 3), ("t", 0, 18), ("L", -3, 7), ("recall_idx", 1, 10), ("orig_idx", 1, 10),
          ("to_go", 0, 18), ("correct", 0, 2), ("in_count", 0, 10), ("in_first", 0, 2), ("n_digits", 0, 10)]
SINGLE += [(name, 1, 32) for name in TOKCOLS]
PAIRS = [(("t", 0, 18), ("in_tok", 1, 32)), (("t", 0, 18), ("prev_tok", 1, 32)),
         (("phase", 0, 3), ("in_tok", 1, 32)), (("phase", 0, 3), ("out_tok", 1, 32)),
         (("recall_idx", 1, 10), ("in_tok", 1, 32)), (("L", -3, 7), ("t", 0, 18)),
         (("L", -3, 7), ("recall_idx", 1, 10))]
FULL_BLOCKS = [[s] for s in SINGLE] + [list(p) for p in PAIRS]
TIMING = [[("phase", 0, 3)], [("t", 0, 18)], [("L", -3, 7)], [("recall_idx", 1, 10)], [("to_go", 0, 18)],
          [("L", -3, 7), ("t", 0, 18)]]
SLOTS = {name: [[(name, 1, 32)]] for name in ("in_tok", "out_tok", "prev_tok", "prev2_tok", "next_out", "orig0", "orig1", "orig2")}
SLOTS["phase x in_tok"] = [[("phase", 0, 3), ("in_tok", 1, 32)]]
SLOTS["phase x out_tok"] = [[("phase", 0, 3), ("out_tok", 1, 32)]]
SLOTS["t x in_tok"] = [[("t", 0, 18), ("in_tok", 1, 32)]]
SLOT_NAMES = {"in_tok": "the current input token", "out_tok": "the token being output", "prev_tok": "the previous input",
              "prev2_tok": "the input two steps back", "next_out": "the next token to output",
              "orig0": "the first original token", "orig1": "the second original token", "orig2": "the third original token",
              "phase x in_tok": "the current input token, by phase", "phase x out_tok": "the token being output, by phase",
              "t x in_tok": "the current input token, by step"}
READABLE_MARGIN = 0.03

_W = {}   # per-process worker state


def _init(store_path: str, results_dir: str):
    from circuit.copy_task.autointerp import Interp, TOKENS
    ip = Interp(store_path)
    c = ip.cols
    raw = {name: np.asarray(c[name]) for name in ("phase", "t", "L", "recall_idx", "orig_idx", "to_go", "correct",
                                                  "in_tok", "prev_tok", "prev2_tok", "out_tok", "next_out",
                                                  "in_count", "in_first", "n_digits")}
    raw["prev3_tok"] = ip.rel_tok(-3)
    for k in range(9):
        raw[f"orig{k}"] = ip.orig(k)
    dev = np.nonzero(~ip.test)[0]
    rng = np.random.default_rng(0)
    _W.update(ip=ip, raw=raw, tokens=TOKENS, dev=dev, test=np.nonzero(ip.test)[0],
              dev_sub=np.sort(rng.choice(dev, min(len(dev), 2_000_000), replace=False)), results=results_dir)


def _layout(blocks):
    offs, off = [], 0
    for b in blocks:
        offs.append(off)
        off += int(np.prod([p[2] for p in b]))
    return offs, off


def _cols(blocks, rows):
    raw = _W["raw"]
    offs, _ = _layout(blocks)
    out = np.empty((len(rows), len(blocks)), dtype=np.int32)
    for j, b in enumerate(blocks):
        code = np.zeros(len(rows), dtype=np.int32)
        for name, shift, size in b:
            code = code * size + (raw[name][rows].astype(np.int32) + shift)
        out[:, j] = offs[j] + code
    return out


def _design(blocks, rows):
    from scipy import sparse
    cols = _cols(blocks, rows)
    _, width = _layout(blocks)
    return sparse.csr_matrix((np.ones(cols.size, dtype=np.float32), cols.ravel(),
                              np.arange(0, cols.size + 1, len(blocks))), shape=(len(rows), width))


def _scores(blocks, w, b, rows, chunk=1_000_000):
    out = np.empty(len(rows), dtype=np.float32)
    for s in range(0, len(rows), chunk):
        out[s:s + chunk] = w[_cols(blocks, rows[s:s + chunk])].sum(1) + b
    return out


def _best_threshold(scores, labels):
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(labels[order])
    f1 = 2 * tp / (np.arange(1, len(order) + 1) + labels.sum())
    i = int(np.argmax(f1))
    return float(scores[order[i]]), float(f1[i])


def _metrics(scores, labels, thr, mags):
    pred = scores >= thr
    tp, n_pred, n_act = int((pred & labels).sum()), int(pred.sum()), int(labels.sum())
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_act if n_act else 0.0
    return {"f1": 2 * p * r / (p + r) if p + r else 0.0, "precision": p, "recall": r,
            "mass_recall": float(mags[pred].sum() / mags.sum()) if mags.sum() else 0.0}


def _cue_name(blocks, col):
    offs, _ = _layout(blocks)
    for b, off in zip(blocks, offs):
        size = int(np.prod([p[2] for p in b]))
        if off <= col < off + size:
            code, vals = col - off, []
            for name, shift, sz in reversed(b):
                v = code % sz - shift
                code //= sz
                vals.append(f"{name}={_tok(v) if name in TOKCOLS else v}")
            return " & ".join(reversed(vals))


def _tok(v):
    return _W["tokens"][v] if v >= 0 else "-"


def _fit(blocks, X, y, C, penalty):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty=penalty, solver="liblinear", C=C, max_iter=300)
    lr.fit(X, y)
    return lr.coef_[0].astype(np.float32), float(lr.intercept_[0])


def fit_feature(gate: str, feature: int) -> Dict:
    t0 = time.time()
    ip, dev, test, dev_sub = _W["ip"], _W["dev"], _W["test"], _W["dev_sub"]
    act = ip.activations(gate, feature)
    y_sub, y_test, m_test = act[dev_sub] > 0, act[test] > 0, act[test]
    rng = np.random.default_rng(feature)
    pos, neg = dev[act[dev] > 0], dev[act[dev] == 0]
    train = np.concatenate([rng.choice(pos, min(len(pos), 100_000), replace=False),
                            rng.choice(neg, min(len(neg), 150_000), replace=False)])
    y = act[train] > 0
    out = {"gate": gate, "feature": feature}

    def finish(blocks, w, b):
        thr, dev_f1 = _best_threshold(_scores(blocks, w, b, dev_sub), y_sub)
        return thr, dev_f1, _metrics(_scores(blocks, w, b, test), y_test, thr, m_test)

    # full
    X_full = _design(FULL_BLOCKS, train)
    w_full, b_full = _fit(FULL_BLOCKS, X_full, y, 0.3, "l1")
    thr, dev_f1, test_m = finish(FULL_BLOCKS, w_full, b_full)
    nz = np.nonzero(w_full)[0]
    out["full"] = {"n_cues": int(len(nz)), "dev_f1": dev_f1, "test": test_m, "intercept": b_full, "threshold": thr,
                   "cues": sorted(((_cue_name(FULL_BLOCKS, int(i)), round(float(w_full[i]), 4)) for i in nz),
                                  key=lambda x: -abs(x[1]))}

    # top40: rank by |w| * std of the cue, refit those cues alone
    p = np.asarray(X_full.mean(0)).ravel()
    importance = np.abs(w_full) * np.sqrt(p * (1 - p))
    keep = [i for i in np.argsort(-importance)[:40] if importance[i] > 0]
    if keep:
        w_k, b_k = _fit(FULL_BLOCKS, X_full[:, keep], y, 1.0, "l2")
        w40 = np.zeros_like(w_full)
        w40[keep] = w_k
        thr, dev_f1, test_m = finish(FULL_BLOCKS, w40, b_k)
        out["top40"] = {"n_cues": len(keep), "dev_f1": dev_f1, "test": test_m, "intercept": b_k, "threshold": thr,
                        "cues": sorted(((_cue_name(FULL_BLOCKS, int(i)), round(float(w40[i]), 4)) for i in keep),
                                       key=lambda x: -abs(x[1]))}

    # slot2: timing + two greedily chosen slots
    chosen = []
    for _ in range(2):
        best = None
        for name, blks in SLOTS.items():
            if name in [c[0] for c in chosen]:
                continue
            blocks = TIMING + [bl for _, bb in chosen for bl in bb] + blks
            w, b = _fit(blocks, _design(blocks, train), y, 1.0, "l2")
            thr, dev_f1 = _best_threshold(_scores(blocks, w, b, dev_sub), y_sub)
            if best is None or dev_f1 > best[0]:
                best = (dev_f1, name, blks, w, b, thr)
        chosen.append((best[1], best[2]))
    dev_f1, _, _, w, b, thr = best
    blocks = TIMING + [bl for _, bb in chosen for bl in bb]
    test_m = _metrics(_scores(blocks, w, b, test), y_test, thr, m_test)
    offs, _ = _layout(blocks)
    slot_weights = {}
    for j, (name, _) in enumerate(chosen):
        bl = blocks[len(TIMING) + j]
        size = int(np.prod([q[2] for q in bl]))
        ws = w[offs[len(TIMING) + j]:offs[len(TIMING) + j] + size]
        slot_weights[name] = sorted(((_cue_name([bl], k), round(float(ws[k]), 4)) for k in range(size)),
                                    key=lambda x: -x[1])
    timing = sorted(((_cue_name(TIMING, k), round(float(w[k]), 4)) for k in range(offs[len(TIMING)])),
                    key=lambda x: -x[1])
    out["slot2"] = {"slots": [c[0] for c in chosen], "dev_f1": dev_f1, "test": test_m, "intercept": b,
                    "threshold": thr, "slot_weights": slot_weights, "timing_weights": timing}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _fmt(cues, n, sign):
    sel = [(c, w) for c, w in cues if (w > 0) == (sign > 0) and w != 0][:n]
    return ", ".join(f"{c} ({w:+.2f})" for c, w in sel)


def summarize(sc: Dict) -> Dict:
    """Choose the readable format and write a short description of it."""
    top40, slot2 = sc.get("top40"), sc["slot2"]
    use_slot = top40 is None or slot2["dev_f1"] >= top40["dev_f1"] - READABLE_MARGIN
    if use_slot:
        parts = []
        for name in slot2["slots"]:
            ws = slot2["slot_weights"][name]
            if " x " in name:
                # Paired slot (phase / step x token): list the top tokens within each context value.
                ctx_name = name.split(" x ")[0]
                by_ctx = {}
                for cue, wt in ws:
                    ctx, tokpart = cue.split(" & ")
                    by_ctx.setdefault(ctx.split("=")[1], []).append((tokpart.split("=")[1], wt))
                label = {"phase": {"0": "input phase", "1": "delimiter step", "2": "copy phase"}}.get(ctx_name, {})
                ranked = sorted(by_ctx.items(), key=lambda kv: -max(w for _, w in kv[1]))[:3]
                desc = "; ".join(f"{label.get(v, f'{ctx_name}={v}')}: " +
                                 ", ".join(t for t, w in sorted(toks, key=lambda x: -x[1])[:4] if w > 0)
                                 for v, toks in ranked)
                parts.append(f"{SLOT_NAMES.get(name, name)} ({desc})")
            else:
                hi = ", ".join(c.split("=")[-1] for c, _ in ws[:6])
                lo = ", ".join(c.split("=")[-1] for c, _ in ws[-4:])
                parts.append(f"{SLOT_NAMES.get(name, name)} (most: {hi}; least: {lo})")
        timing = ", ".join(c for c, _ in slot2["timing_weights"][:3])
        text = (f"Graded preference over {' and '.join(parts)}; timing cues that raise it most: {timing}.")
        fmt, chosen = "slot2", slot2
    else:
        text = (f"Weighted combination of {top40['n_cues']} cues. Raise it: {_fmt(top40['cues'], 6, +1)}. "
                f"Lower it: {_fmt(top40['cues'], 4, -1)}.")
        fmt, chosen = "top40", top40
    return {"format": fmt, "summary": text, "test_f1": chosen["test"]["f1"],
            "test_precision": chosen["test"]["precision"], "test_recall": chosen["test"]["recall"],
            "full_test_f1": sc["full"]["test"]["f1"], "full_n_cues": sc["full"]["n_cues"],
            "top40_test_f1": top40["test"]["f1"] if top40 else None, "slot2_test_f1": slot2["test"]["f1"]}


def _work(key):
    gate, feature = key
    results = _W["results"]
    os.makedirs(os.path.join(results, "scorecards"), exist_ok=True)
    sc = fit_feature(gate, feature)
    with open(os.path.join(results, "scorecards", f"{gate}_{feature}.json"), "w") as f:
        json.dump(sc, f)
    path = os.path.join(results, f"{gate}_{feature}.json")
    rec = json.load(open(path))
    rec["scorecard"] = summarize(sc)
    rec["scorecard"]["file"] = f"scorecards/{gate}_{feature}.json"
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    agent = (rec.get("test_scores") or {}).get("f1")
    return gate, feature, rec["scorecard"], agent, sc["seconds"]


TARGET_VERDICTS = ("partially_explained", "no_explanation", "polysemantic")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("store")
    ap.add_argument("results")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--features", nargs="*", help="gate/feature list; default: all partial / unexplained / polysemantic")
    ap.add_argument("--redo", action="store_true", help="refit features that already have a scorecard")
    args = ap.parse_args()
    if args.features:
        keys = [(f.split("/")[0], int(f.split("/")[1])) for f in args.features]
    else:
        keys = []
        for name in sorted(os.listdir(args.results)):
            if not name.endswith(".json") or name == "summary.json":
                continue
            rec = json.load(open(os.path.join(args.results, name)))
            verdict = rec.get("final_verdict") or rec.get("verdict")
            if verdict in TARGET_VERDICTS and (args.redo or "scorecard" not in rec):
                keys.append((rec["gate"], int(rec["feature"])))
    print(f"fitting scorecards for {len(keys)} features with {args.workers} workers", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.store, args.results)) as pool:
        for i, (gate, feature, s, agent, secs) in enumerate(pool.imap_unordered(_work, keys), 1):
            agent_s = f"{agent:.2f}" if agent is not None else " -  "
            print(f"[{i}/{len(keys)}] {gate}/{feature}: agent {agent_s} | {s['format']} {s['test_f1']:.2f} "
                  f"| full {s['full_test_f1']:.2f} ({s['full_n_cues']} cues) | {secs}s | "
                  f"elapsed {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
