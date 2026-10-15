"""Per-feature clamp tests: does changing one feature have the downstream effect its explanation predicts? (real GRU)

Each test edits one candidate feature along its decoder direction inside the real GRU (n += (a_target - a) d_f,
the transcoder error kept), in the situation its explanation describes, and checks the effect the explanation
implies. Predictions come from the explanations alone:

  firing "input X" features (original phase, e.g. 120 "input g in the first five positions")
    on:  set to its mean active value at one step where the label's other conditions hold but the input is not X
         -> predicted: the output of that slot moves towards X
    off: set to 0 at the steps where the label holds
         -> predicted: X's logit at that slot's output drops
  firing "copy input X" features (the fed-back token, e.g. 95 "copy-phase input q")
    off: set to 0 where X is fed back -> predicted: X is not popped, so the next output repeats X
    on:  set to its mean active value at a copy step where X is fed back later -> predicted: X is popped early,
         so X's logit at its own output drops
  magnitude features (sets S+ / S- from circuit/copy_task/magnitude_scores.py)
    boost / suppress by delta (1.5 std of its input-phase activation) at every input step
    -> predicted: the outputs move towards S+ (boost) or S- (suppress)

Controls: for firing tests, the rank of X among all 30 tokens by logit change (1 = the most moved in the predicted
direction); for magnitude tests, the same change for random token sets of the same sizes.

    python -m circuit.copy_task.clamp_tests --rnn RNN.ckpt --traces T4.pt T7.pt \\
        --transcoders reset=R.ckpt update=U.ckpt candidate=C.ckpt \\
        --firing features/<run>_autointerp --magnitude features/copy_circuits/magnitude_scores_hidden.json \\
        --out clamp_tests.json
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch
from torch.utils.data import StackDataset

from circuit.copy_task.autointerp import DIGITS, LETTERS, TOKEN_IDX, TOKENS, VOWELS, build_step_table
from circuit.copy_task.circuit_tests import clean_acts, run
from circuit.copy_task.explained_fidelity import load_transcoder
from training.splice_eval import build_task_rnn

torch.serialization.add_safe_globals([StackDataset])
MAGNITUDE_FEATURES = [0, 32, 54, 79, 91, 140, 145, 162, 266, 274, 280, 283, 284, 440, 487, 488]
FIRING_FEATURES = [120, 36, 95, 215, 316, 408]          # sparse features that matter most alone
SIMPLE_PREDICATE = re.compile(r'in_tok == tok\("(.)"\)')


def set_hook(tc, f, target, mask):
    """Candidate hook: feature f takes target[:, t] where mask[:, t]."""
    d = tc.features_to_outputs.weight[:, f]

    def hook(t, h, xt, r, z, n):
        m = mask[:, t]
        if not m.any():
            return r, z, n
        a = tc.act(tc.input_to_features(torch.cat([r * h, xt], -1)))[:, f]
        return r, z, n + ((target[:, t] - a) * m)[:, None] * d
    return hook


def step_vars(x, L):
    """Step variables (N, T) of circuit/copy_task/autointerp.py for the sequences in x."""
    toks = x.argmax(-1).numpy()
    cols = build_step_table([[TOKENS[k] for k in row] for row in toks])
    return {k: v.reshape(len(x), 2 * L) for k, v in cols.items() if k != "row_start"}


def eval_predicate(pred, cols):
    ns = {**cols, "np": np, "tok": lambda s: TOKEN_IDX[s], "isin": np.isin, "LETTERS": LETTERS, "DIGITS": DIGITS,
          "VOWELS": VOWELS}
    return np.broadcast_to(np.asarray(eval(pred, {"__builtins__": {}}, ns), dtype=bool), cols["t"].shape)


def rank_of(delta, tok, sign):
    """Rank (1 = moved most in the predicted direction) of tok among all tokens, per row of delta (n, V)."""
    order = (-sign * delta).argsort(1).argsort(1)
    return order[np.arange(len(delta)), tok] + 1


def firing_test(rnn, tc, x, L, acts, base_lg, f, pred, X, rng):
    cols = step_vars(x, L)
    N, T = cols["t"].shape
    holds = eval_predicate(pred, cols)
    with_x = eval_predicate(pred, {**cols, "in_tok": np.full_like(cols["in_tok"], X)})   # label true if input were X
    a = acts[:, :, f]
    on_value = float(a[torch.from_numpy(holds.copy())].mean()) if holds.any() else float(a[a > 0].mean())
    phase = cols["phase"]
    out = {}
    base = base_lg.numpy()
    orig = x[:, :L, :-1].argmax(-1).numpy()

    def pick(cand):
        """One random candidate step per sequence (rows without one are dropped)."""
        rows, steps = [], []
        for s in range(N):
            c = np.flatnonzero(cand[s])
            if len(c):
                rows.append(s)
                steps.append(int(rng.choice(c)))
        return np.array(rows, dtype=np.int64), np.array(steps, dtype=np.int64)

    def edit(rows, steps, value):
        mask = torch.zeros(len(rows), T, dtype=torch.bool)
        mask[torch.arange(len(rows)), torch.from_numpy(steps)] = True
        target = torch.full((len(rows), T), value)
        return run(rnn, x[rows], L, set_hook(tc, f, target, mask)).numpy()

    def summarise(lg, rows, out_idx, sign):
        """Logit change at output out_idx (per row) for X vs the other tokens, and X's rank among all tokens."""
        r = np.arange(len(rows))
        delta = lg[r, out_idx] - base[rows, out_idx]
        dx = delta[r, X]
        others = (delta.sum(1) - dx) / (delta.shape[1] - 1)
        return {"n": int(len(rows)), "dlogit_X": float(dx.mean()), "dlogit_others": float(others.mean()),
                "agree": float(((dx - others) * sign > 0).mean()), "median_rank_of_X": float(np.median(rank_of(delta, X, sign))),
                "argmax_is_X": float((lg[r, out_idx].argmax(-1) == X).mean())}

    if (phase[holds] == 0).all():                      # "input X": the slot written at step s is output at L + s
        rows, steps = pick(with_x & (cols["in_tok"] != X) & (phase == 0))
        if len(rows):
            out["on"] = summarise(edit(rows, steps, on_value), rows, steps, +1)
        rows, steps = pick(holds)
        if len(rows):
            lg = edit(rows, steps, 0.0)
            res = summarise(lg, rows, steps, -1)
            res["acc_at_slot"] = float((lg[np.arange(len(rows)), steps].argmax(-1) == orig[rows, steps]).mean())
            out["off"] = res
        out["kind"] = "input X"
    elif (phase[holds] == 2).all():                    # "copy input X": fed back at step s = L + j, output j there
        rows, steps = pick(holds)
        if len(rows):
            out["off"] = summarise(edit(rows, steps, 0.0), rows, steps - L, +1)    # predicted: X repeated
        later = np.zeros((N, T), dtype=bool)           # copy steps where X is fed in later and output later
        pos = {}
        for s in range(N):
            k = np.flatnonzero(orig[s] == X)
            if len(k):
                pos[s] = int(k[-1])
                later[s, L + 1:L + pos[s]] = True      # steps before X's own output step L + k
        rows, steps = pick(with_x & later & (cols["in_tok"] != X))
        if len(rows):
            kk = np.array([pos[s] for s in rows])
            out["on"] = summarise(edit(rows, steps, on_value), rows, kk, -1)     # predicted: X's own logit drops
        out["kind"] = "copy input X"
    else:
        out["kind"] = "mixed phases (not tested)"
    out.update(on_value=on_value)
    return out


def magnitude_test(rnn, tc, x, L, acts, base_lg, f, plus, minus, rng, n_random=200):
    a = acts[:, :L, f]
    delta = 1.5 * float(a.std())
    T = 2 * L
    mask = torch.zeros(len(x), T, dtype=torch.bool)
    mask[:, :L] = True
    base = base_lg.numpy()
    out = {"delta": delta}
    pool = np.arange(30)
    rand_sets = [rng.permutation(pool) for _ in range(n_random)]
    for name, sign in (("boost", +1), ("suppress", -1)):
        target = torch.zeros(len(x), T)
        target[:, :L] = (acts[:, :L, f] + sign * delta).clamp(min=0)
        lg = run(rnn, x, L, set_hook(tc, f, target, mask)).numpy()
        d = lg - base                                                  # (N, L, V)
        gap = lambda P, M: d[..., P].mean(-1) - d[..., M].mean(-1)     # (N, L)
        g = gap(plus, minus)
        null = np.array([gap(p[:len(plus)], p[len(plus):len(plus) + len(minus)]).mean() for p in rand_sets])
        out[name] = {"dgap_S+_minus_S-": float(g.mean()), "agree": float((g * sign > 0).mean()),
                     "random_sets_dgap_mean": float(null.mean()), "random_sets_dgap_std": float(null.std()),
                     "z_vs_random_sets": float((g.mean() - null.mean()) / (null.std() + 1e-9)),
                     "acc": float((lg.argmax(-1) == x[:, :L, :-1].argmax(-1).numpy()).mean()),
                     "share_outputs_in_S+": float(np.isin(lg.argmax(-1), plus).mean()),
                     "share_outputs_in_S+_clean": float(np.isin(base.argmax(-1), plus).mean())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--transcoders", nargs=3, required=True, metavar="GATE=PATH")
    ap.add_argument("--firing", required=True, help="firing autointerp results dir (hidden_<f>.json)")
    ap.add_argument("--magnitude", required=True, help="magnitude_scores_hidden.json (token sets)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n_sample", type=int, nargs=2, default=[25, 15], metavar=("INPUT", "COPY"),
                    help="random explained 'input X' / 'copy input X' features to add")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(args.seed)

    paths = dict(item.split("=", 1) for item in args.transcoders)
    rnn = build_task_rnn("copy", 31, 128)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    tcs = {k: load_transcoder(paths[k]) for k in ("reset", "update", "candidate")}
    tc = tcs["candidate"]

    firing = {}
    for p in glob.glob(os.path.join(args.firing, "hidden_*.json")):
        r = json.load(open(p))
        firing[r["feature"]] = r
    simple = {f: r for f, r in firing.items() if r.get("final_verdict") == "explained"
              and len(SIMPLE_PREDICATE.findall(r["predicate"])) == 1 and "prev_tok" not in r["predicate"]}
    input_pool = sorted(f for f, r in simple.items() if "phase == 2" not in r["predicate"] and f not in FIRING_FEATURES)
    copy_pool = sorted(f for f, r in simple.items() if "phase == 2" in r["predicate"] and f not in FIRING_FEATURES)
    sample = (list(rng.choice(input_pool, min(args.n_sample[0], len(input_pool)), replace=False)) +
              list(rng.choice(copy_pool, min(args.n_sample[1], len(copy_pool)), replace=False)))
    tested = [(f, "important") for f in FIRING_FEATURES] + [(int(f), "random explained") for f in sample]
    sets = {r["feature"]: r["token_sets"] for r in json.load(open(args.magnitude))["features"]}

    results = {"firing": {}, "magnitude": {}}
    for path in args.traces:
        data = torch.load(path, map_location="cpu", mmap=True).datasets
        L = data["outputs"].shape[1]
        x = data["inputs"][:args.n].float()
        acts = clean_acts(rnn, x, L, tcs)["candidate"]
        base_lg = run(rnn, x, L)
        print(f"L={L}: baseline accuracy {(base_lg.argmax(-1) == x[:, :L, :-1].argmax(-1)).float().mean():.3f}")
        for f, group in tested:
            r = firing[f]
            X = TOKEN_IDX[SIMPLE_PREDICATE.findall(r["predicate"])[0]]
            res = firing_test(rnn, tc, x, L, acts, base_lg, f, r["predicate"], X, rng)
            results["firing"].setdefault(str(f), {"label": r["label"], "predicate": r["predicate"], "group": group})[f"L{L}"] = res
            msg = " | ".join(f"{k}: dX {v['dlogit_X']:+.2f} vs {v['dlogit_others']:+.2f}, agree {v['agree']:.2f}, "
                             f"rank {v['median_rank_of_X']:.0f}, n {v['n']}" for k, v in res.items() if isinstance(v, dict))
            print(f"  firing {f:>3} ({r['label'][:40]}): {msg}")
        for f in MAGNITUDE_FEATURES:
            s = sets[f]
            plus, minus = [TOKEN_IDX[t] for t in s["S+"]], [TOKEN_IDX[t] for t in s["S-"]]
            res = magnitude_test(rnn, tc, x, L, acts, base_lg, f, plus, minus, rng)
            results["magnitude"].setdefault(str(f), {"S+": s["S+"], "S-": s["S-"]})[f"L{L}"] = res
            print(f"  magnitude {f:>3}: " + " | ".join(
                f"{k}: dgap {res[k]['dgap_S+_minus_S-']:+.2f} (random {res[k]['random_sets_dgap_mean']:+.2f}±{res[k]['random_sets_dgap_std']:.2f}), "
                f"agree {res[k]['agree']:.2f}, acc {res[k]['acc']:.2f}" for k in ("boost", "suppress")))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), **results}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
