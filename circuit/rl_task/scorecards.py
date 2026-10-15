"""Fitted scorecards for RL features that no short predicate explains.

RL version of circuit/copy_task/scorecards.py. A scorecard predicts whether a
feature fires from additive weights over one-hot "cues" of the step: phase,
position in the window and the episode, the inputs, this / the previous / the
next trial's choice, state and reward, the hidden good state, and bins of the
model's value and policy, plus pairs with phase. Per feature, fitted on the dev
split (even episodes) and scored on the test split (odd episodes):

  full   L1-sparse scorecard over all cue blocks (fidelity reference)
  top40  the 40 most important cues of the full fit, refitted alone
  slot2  timing cues + graded weights over two "slots" (phase x a trial-level
         cue), chosen greedily

The readable format (top40 or slot2; slot2 if within 0.03 dev F1) gets a short
summary in the feature's result JSON; all weights go to
<results>/scorecards/<gate>_<feature>.json.

    python -m circuit.rl_task.scorecards STORE RESULTS_DIR [--workers 4] [--features gate/feat ...]
"""
import argparse
import json
import os
import time
import warnings
from multiprocessing import get_context
from typing import Dict

import numpy as np

from circuit.copy_task.scorecards import READABLE_MARGIN, TARGET_VERDICTS, _best_threshold, _fit, _layout, _metrics

warnings.filterwarnings("ignore")

# (name, shift, size): one-hot code = value + shift
SINGLE = [("phase", 0, 3), ("rel", 3, 5), ("ep_dec", 0, 10), ("ep_bucket", 0, 6), ("in_rew", 1, 3),
          ("in_action", 0, 3), ("in_state", 0, 3), ("action", 0, 3), ("state", 0, 3), ("common", 1, 3),
          ("reward", 2, 4), ("prev_action", 1, 4), ("prev_state", 1, 4), ("prev_common", 1, 3),
          ("stay", 1, 3), ("next_stay", 1, 3), ("good", 1, 4), ("chose_good", 1, 3), ("since_rev", 1, 22),
          ("start_high", 0, 2), ("model_act", 0, 3), ("value_bin", 0, 10), ("choice_bin", 0, 10),
          ("last", 0, 9), ("before", 0, 9)]
PH = ("phase", 0, 3)
PAIRS = [(PH, ("last", 0, 9)), (PH, ("before", 0, 9)), (PH, ("this", 0, 10)), (PH, ("ep_dec", 0, 10)),
         (PH, ("value_bin", 0, 10)), (PH, ("choice_bin", 0, 10)), (PH, ("good", 1, 4)), (PH, ("rel", 3, 5)),
         (PH, ("reward", 2, 4)), (PH, ("since_rev", 1, 22))]
FULL_BLOCKS = [[s] for s in SINGLE] + [list(p) for p in PAIRS]
TIMING = [[PH], [PH, ("ep_dec", 0, 10)], [("rel", 3, 5)]]
SLOTS = {"last trial": [[PH, ("last", 0, 9)]], "trial before": [[PH, ("before", 0, 9)]],
         "this trial": [[PH, ("this", 0, 10)]], "this reward (future)": [[PH, ("reward", 2, 4)]],
         "good state (hidden)": [[PH, ("good", 1, 4)]], "trials since reversal (hidden)": [[PH, ("since_rev", 1, 22)]],
         "policy (model)": [[PH, ("choice_bin", 0, 10)]], "value (model)": [[PH, ("value_bin", 0, 10)]]}
PATTERNS = ["?", "a1 S1 r", "a1 S1 u", "a1 S2 r", "a1 S2 u", "a2 S2 r", "a2 S2 u", "a2 S1 r", "a2 S1 u"]
THIS = ["fix", "a1 ?", "a2 ?", "a1 S1", "a1 S2", "a2 S2", "a2 S1", "?", "?", "?"]
PHASE_NAMES = ["D1", "GO", "D2"]

_W = {}   # per-process worker state


def _pattern(action, state, reward):
    """Trial pattern code (PATTERNS): choice x state reached x reward; 0 unknown."""
    out = np.zeros(len(action), dtype=np.int16)
    for k, name in enumerate(PATTERNS[1:], 1):
        a, s, r = name.split()
        out[(action == int(a[1])) & (state == int(s[1])) & (reward == (1 if r == "r" else 0))] = k
    return out


def _init(store_path: str, results_dir: str):
    from circuit.rl_task.autointerp import Interp
    ip = Interp(store_path)
    c = ip.cols
    raw = {name: np.asarray(c[name]) for name, _, _ in SINGLE if name in c}
    raw["ep_dec"] = np.clip(np.asarray(c["ep_trial"]) // 10, 0, 9)
    for name in ("value", "choice"):
        q = np.quantile(np.asarray(c[name])[~ip.test], np.linspace(0.1, 0.9, 9))
        raw[f"{name}_bin"] = np.searchsorted(q, np.asarray(c[name]))
    raw["since_rev"] = np.clip(np.asarray(c["since_rev"]), -1, 20)
    # previous trial and the one before it: choice x state x reward (the reward seen at the next D1)
    raw["last"] = _pattern(np.asarray(c["prev_action"]), np.asarray(c["prev_state"]), np.asarray(c["prev_reward"]))
    raw["before"] = _pattern(ip.tv("action", -2), ip.tv("state", -2), ip.tv("reward", -2))
    # this trial: choice x state (state known from D2; at D1 / GO both are upcoming)
    a, s = np.asarray(c["action"]), np.asarray(c["state"])
    this = np.select([a == 0, (a == 1) & (s == 1), (a == 1) & (s == 2), (a == 2) & (s == 2), (a == 2) & (s == 1)],
                     [0, 3, 4, 5, 6], 7)
    raw["this"] = this
    dev = np.nonzero(~ip.test)[0]
    _W.update(ip=ip, raw={k: v.astype(np.int32) for k, v in raw.items()}, dev=dev, test=np.nonzero(ip.test)[0],
              results=results_dir)


def _cols(blocks, rows):
    raw = _W["raw"]
    offs, _ = _layout(blocks)
    out = np.empty((len(rows), len(blocks)), dtype=np.int32)
    for j, b in enumerate(blocks):
        code = np.zeros(len(rows), dtype=np.int32)
        for name, shift, size in b:
            code = code * size + np.clip(raw[name][rows] + shift, 0, size - 1)
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


def _value_name(name, v):
    if name == "phase":
        return PHASE_NAMES[v]
    if name in ("last", "before"):
        return PATTERNS[v]
    if name == "this":
        return THIS[v]
    if name == "choice_bin":
        return f"decile {v} (9 = most a1)"
    if name == "value_bin":
        return f"decile {v} (9 = highest)"
    return str(v)


def _cue_name(blocks, col):
    offs, _ = _layout(blocks)
    for b, off in zip(blocks, offs):
        size = int(np.prod([p[2] for p in b]))
        if off <= col < off + size:
            code, vals = col - off, []
            for name, shift, sz in reversed(b):
                vals.append(f"{name}={_value_name(name, code % sz - shift)}")
                code //= sz
            return " & ".join(reversed(vals))


def fit_feature(gate: str, feature: int) -> Dict:
    t0 = time.time()
    ip, dev, test = _W["ip"], _W["dev"], _W["test"]
    act = ip.activations(gate, feature)
    y_dev, y_test, m_test = act[dev] > 0, act[test] > 0, act[test]
    rng = np.random.default_rng(feature)
    pos, neg = dev[act[dev] > 0], dev[act[dev] == 0]
    train = np.concatenate([rng.choice(pos, min(len(pos), 100_000), replace=False),
                            rng.choice(neg, min(len(neg), 150_000), replace=False)])
    y = act[train] > 0
    out = {"gate": gate, "feature": feature}

    def finish(blocks, w, b):
        thr, dev_f1 = _best_threshold(_scores(blocks, w, b, dev), y_dev)
        return thr, dev_f1, _metrics(_scores(blocks, w, b, test), y_test, thr, m_test)

    X_full = _design(FULL_BLOCKS, train)
    w_full, b_full = _fit(FULL_BLOCKS, X_full, y, 0.3, "l1")
    thr, dev_f1, test_m = finish(FULL_BLOCKS, w_full, b_full)
    nz = np.nonzero(w_full)[0]
    out["full"] = {"n_cues": int(len(nz)), "dev_f1": dev_f1, "test": test_m, "intercept": b_full, "threshold": thr,
                   "cues": sorted(((_cue_name(FULL_BLOCKS, int(i)), round(float(w_full[i]), 4)) for i in nz),
                                  key=lambda x: -abs(x[1]))}

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

    chosen = []
    for _ in range(2):
        best = None
        for name, blks in SLOTS.items():
            if name in [c[0] for c in chosen]:
                continue
            blocks = TIMING + [bl for _, bb in chosen for bl in bb] + blks
            w, b = _fit(blocks, _design(blocks, train), y, 1.0, "l2")
            thr, dev_f1 = _best_threshold(_scores(blocks, w, b, dev), y_dev)
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
            by_phase = {}
            for cue, wt in slot2["slot_weights"][name]:
                ph, val = cue.split(" & ")
                by_phase.setdefault(ph.split("=")[1], []).append((val.split("=")[1], wt))
            ranked = sorted(((ph, vals) for ph, vals in by_phase.items() if max(w for _, w in vals) > 0),
                            key=lambda kv: -max(w for _, w in kv[1]))[:2]
            desc = "; ".join(f"{ph}: " + ", ".join(v for v, w in sorted(vals, key=lambda x: -x[1])[:3] if w > 0)
                             for ph, vals in ranked)
            parts.append(f"{name} ({desc})")
        timing = ", ".join(c for c, _ in slot2["timing_weights"][:3])
        text = f"Graded preference over {' and '.join(parts)}; timing cues that raise it most: {timing}."
        fmt, chosen = "slot2", slot2
    else:
        text = (f"Weighted combination of {top40['n_cues']} cues. Raise it: {_fmt(top40['cues'], 6, +1)}. "
                f"Lower it: {_fmt(top40['cues'], 4, -1)}.")
        fmt, chosen = "top40", top40
    return {"format": fmt, "summary": text, "test_f1": chosen["test"]["f1"],
            "test_precision": chosen["test"]["precision"], "test_recall": chosen["test"]["recall"],
            "full_test_f1": sc["full"]["test"]["f1"], "full_n_cues": sc["full"]["n_cues"],
            "top40_test_f1": top40["test"]["f1"] if top40 else None, "slot2_test_f1": slot2["test"]["f1"],
            "slot2_slots": slot2["slots"]}


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("store")
    ap.add_argument("results")
    ap.add_argument("--workers", type=int, default=4)
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
