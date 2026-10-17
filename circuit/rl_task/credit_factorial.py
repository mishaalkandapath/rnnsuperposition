"""Credit assignment in the two-step-task GRU: does a reward count for the state reached or for the action taken?

For each held-out trace window (contrast_flow.load_windows) the core trial c is rerun in the real GRU with all 8
combinations of its choice (a1 / a2, shown at c.D2), the state reached (S1 / S2, shown at c.D2) and its reward
(1 / 0, shown at (c+1).D1); the history before and the inputs after are as recorded. Target: the next choice,
logit a1 - logit a2 at (c+1).GO. All 8 combinations occur in the task (a rare transition is 20%).

  reward effect   y(r = 1) - y(r = 0), in each (action, state) cell
  credit to state reward effect after S1 minus after S2, averaged over the two actions (model-based: large)
  credit to action reward effect after a1 minus after a2, averaged over the two states (model-free: large)
  perseveration   y(a1) - y(a2), averaged over states and rewards

What the hidden state keeps: h after c.D2, (c+1).D1 and (c+1).GO is an exact sum of writes, the carried state
(1 - z) h_prev, each candidate feature's write z d_f a_f, and the decoder-bias and error write z (b + e). The state
effect on h (S1 minus S2, averaged over actions) and the action effect (a1 minus a2, averaged over states), at each
reward, split the same way: each write's share is its projection on the total effect.

Each interaction is a difference of exact contrastive flows (contrast_flow.full_flow), seen three ways, which give
the same total:
  reward side   reward contrasts in each (action, state) cell: which next-D1 / GO features carry the reward
                differently after S1 and S2 (or after a1 and a2)
  state side    state contrasts at r = 1 minus at r = 0: where the remembered state meets the reward
  action side   action contrasts at r = 1 minus at r = 0

    python -m circuit.rl_task.credit_factorial --out rl_features/rl_circuits/credit_factorial.json [--n_windows 75]
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.contrast_flow import A1, A2, REW, S1, S2, full_flow, gru_run, load_model, load_windows, step_name

CELLS = [(a, s, r) for a in (1, 2) for s in (1, 2) for r in (1, 0)]


def set_trial(w, a, s, r):
    x, c = w["x"].clone(), w["c"]
    x[3 * c + 2, [A1, A2]] = torch.tensor([float(a == 1), float(a == 2)])
    x[3 * c + 2, [S1, S2]] = torch.tensor([float(s == 1), float(s == 2)])
    x[3 * c + 3, REW] = float(r)
    return x


def writes(tracer, acts, t):
    """h after step t as {term: vector}: carried state, each active candidate feature's write, bias and error."""
    z, dec = acts["z_ts"][t], tracer.hidden_transcoder.features_to_outputs
    out = {"carry": (1 - z) * acts["h_prevs"][t], "bias+error": z * (dec.bias + acts["e_n"][t])}
    for f in torch.nonzero(acts["f_n"][t] > 0).flatten().tolist():
        out[f"n:{f}"] = z * dec.weight[:, f] * acts["f_n"][t, f]
    return out


def effect_split(tracer, acts, t, r, which):
    """State (S1 - S2, mean over actions) or action (a1 - a2, mean over states) effect on h after step t at reward
    r: (total effect vector, {term: its share, the projection on the total over the total's norm})."""
    on = [(1, 1, r), (2, 1, r)] if which == "state" else [(1, 1, r), (1, 2, r)]
    off = [(1, 2, r), (2, 2, r)] if which == "state" else [(2, 1, r), (2, 2, r)]
    eff = defaultdict(lambda: torch.zeros(acts[on[0]]["h_ts"].shape[1]))
    for sign, cells in ((0.5, on), (-0.5, off)):
        for cell in cells:
            for k, v in writes(tracer, acts[cell], t).items():
                eff[k] = eff[k] + sign * v
    total = sum(eff.values())
    nrm = float(total.norm())
    return total, {k: float(v @ total) / max(nrm ** 2, 1e-12) for k, v in eff.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", default="rl_features/rl_trace_sequences")
    ap.add_argument("--n_windows", type=int, default=75, help="base windows per recorded core-trial pattern")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rnn, tcs, tracer = load_model()
    windows = load_windows(args.traces, args.n_windows, args.seed)
    print(f"{len(windows)} base windows ({time.time() - t0:.0f}s)", flush=True)
    w_T = tracer.W_choice
    Y = []
    # interaction name -> view -> list of (sign, contrast pair): pair = (cell_on, cell_off)
    pairs = {
        "credit_to_state": {
            "reward_side": [(+1, (a, 1, 1), (a, 1, 0)) for a in (1, 2)] + [(-1, (a, 2, 1), (a, 2, 0)) for a in (1, 2)],
            "state_side": [(+1, (a, 1, 1), (a, 2, 1)) for a in (1, 2)] + [(-1, (a, 1, 0), (a, 2, 0)) for a in (1, 2)]},
        "credit_to_action": {
            "reward_side": [(+1, (1, s, 1), (1, s, 0)) for s in (1, 2)] + [(-1, (2, s, 1), (2, s, 0)) for s in (1, 2)],
            "action_side": [(+1, (1, s, 1), (2, s, 1)) for s in (1, 2)] + [(-1, (1, s, 0), (2, s, 0)) for s in (1, 2)]},
    }
    agg = {(i, v): {"node": defaultdict(float), "src": defaultdict(float), "total": []}
           for i, views in pairs.items() for v in views}
    checks = []
    held = defaultdict(lambda: {"norm": [], "choice": [], "share": defaultdict(float)})
    for k, w in enumerate(windows):
        c, tau = w["c"], 3 * w["c"] + 4
        acts = {cell: tracer.run_forward_pass(gru_run(rnn, set_trial(w, *cell), w["h0"])) for cell in CELLS}
        Y.append({cell: float(a["choice"][tau]) for cell, a in acts.items()})
        for t in (3 * c + 2, 3 * c + 3, 3 * c + 4):
            for r in (1, 0):
                for which in ("state", "action"):
                    total, share = effect_split(tracer, acts, t, r, which)
                    hk = held[(which, step_name(t, c), r)]
                    hk["norm"].append(float(total.norm())); hk["choice"].append(float(total @ w_T))
                    for kk, v in share.items():
                        hk["share"][kk] += v
        flows = {}
        for i, views in pairs.items():
            for v, terms in views.items():
                a = agg[(i, v)]
                tot = 0.0
                for sign, on, off in terms:
                    if (on, off) not in flows:
                        flows[(on, off)] = full_flow(tracer, acts[on], acts[off], w_T, tau)
                    f = flows[(on, off)]
                    checks.append(max(abs(f["check"]), f["preact_check"]))
                    tot += sign * f["d_target"] / 2                       # averaged over the two cells
                    for (kd, t, fi), fl in zip(f["nodes"], f["flow"].tolist()):
                        if fl != 0:
                            a["node"][f"{step_name(t, c)}/{kd}:{fi}"] += sign * fl / 2
                    for key, val in f["into"].items():
                        for (_, t, _), x in zip(f["nodes"], val.tolist()):
                            a["src"][f"{key}@{step_name(t, c)}"] += sign * x / 2
                    for key, val in f["target"].items():
                        a["src"][key] += sign * val / 2
                a["total"].append(tot)
        if k % 50 == 0:
            print(f"{k}/{len(windows)} ({time.time() - t0:.0f}s)", flush=True)
    n = len(windows)
    y = {cell: np.array([d[cell] for d in Y]) for cell in CELLS}
    rew = {cell[:2]: y[cell[:2] + (1,)] - y[cell[:2] + (0,)] for cell in CELLS}
    stats = lambda v: {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v)), 3),
                       "q10_q90": [round(float(q), 3) for q in np.quantile(v, [0.1, 0.9])]}
    result = {
        "n_windows": n, "max_check": float(np.max(checks)),
        "next_choice_by_cell": {f"a{a}->S{s} r={r}": stats(y[(a, s, r)]) for a, s, r in CELLS},
        "reward_effect_by_cell": {f"a{a}->S{s}": stats(rew[(a, s)]) for a in (1, 2) for s in (1, 2)},
        "credit_to_state": stats((rew[(1, 1)] + rew[(2, 1)] - rew[(1, 2)] - rew[(2, 2)]) / 2),
        "credit_to_action": stats((rew[(1, 1)] + rew[(1, 2)] - rew[(2, 1)] - rew[(2, 2)]) / 2),
        "reward_x_state_x_action": stats((rew[(1, 1)] - rew[(1, 2)] - rew[(2, 1)] + rew[(2, 2)]) / 2),
        "perseveration": stats(np.mean([y[(1, s, r)] - y[(2, s, r)] for s in (1, 2) for r in (1, 0)], 0)),
        "state_main": stats(np.mean([y[(a, 1, r)] - y[(a, 2, r)] for a in (1, 2) for r in (1, 0)], 0)),
        "graphs": {},
        "held_in_h": {f"{which} effect after {s}, r={r}": {
            "norm_mean": round(float(np.mean(h["norm"])), 3),
            "on_choice_readout_mean": round(float(np.mean(h["choice"])), 3),
            "shares": {kk: round(v / n, 3) for kk, v in sorted(h["share"].items(), key=lambda kv: -abs(kv[1]))[:args.top]
                       if abs(v / n) >= 0.01}}
            for (which, s, r), h in held.items()},
    }
    for (i, v), a in agg.items():
        node = {kk: round(val / n, 4) for kk, val in sorted(a["node"].items(), key=lambda kv: -abs(kv[1]))
                if abs(val / n) >= 1e-3}
        by_step = defaultdict(list)
        for nm, val in node.items():
            by_step[nm.split("/")[0]].append((nm.split("/")[1], val))
        result["graphs"].setdefault(i, {})[v] = {
            "total": round(float(np.mean(a["total"])), 3),
            "flow_by_source": {kk: round(val / n, 4) for kk, val in sorted(a["src"].items(), key=lambda kv: -abs(kv[1]))
                               if abs(val / n) >= 1e-3},
            "flow_through_step": {s: round(sum(val for _, val in fs), 3) for s, fs in sorted(by_step.items())},
            "top_features_by_step": {s: [{"feature": fi, "flow": val} for fi, val in fs[:args.top]]
                                     for s, fs in sorted(by_step.items())}}
    for key in ("credit_to_state", "credit_to_action", "reward_x_state_x_action", "perseveration", "state_main"):
        print(key, result[key], flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
