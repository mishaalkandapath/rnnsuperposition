"""The circuit of the inferred-value effect: how an outcome in one state changes the value response on reaching the
other state (value_circuit.py: arriving at a state believed bad gives -0.48 at the reveal, at one inferred good only
through an outcome in the other state -0.20).

Closed-loop episodes (reversal_flow.rollout). An event: trial t reaches state X and its outcome flips the ideal
observer's belief, so that the other state Y is now believed good; trial t+1 reaches Y. The window runs from trial
t-3 to (t+1).D2, inputs as recorded. The contrast swaps trial t's reward (shown at (t+1).D1) so that Y stays believed
bad (events where the swap does not do that are dropped). Targets, all straight lines through h so the flow stays
exact (contrast_flow.full_flow, features of all three gates):
  reveal   the value change when Y is revealed: value after (t+1).D2 minus value after (t+1).GO
  value    the value after (t+1).D2
Flow by step and the top features; and patching the top-k flow features per step (real GRU, gates = reconstruction +
held error, circuit_checks.run_patched with the value head) from the counterfactual run into the original: share of
the reveal difference carried, against random changed features.

    python -m circuit.rl_task.inferred_value_flow --out rl_features/rl_circuits/inferred_value_flow.json
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.circuit_checks import run_patched
from circuit.rl_task.contrast_flow import REW, full_flow, gru_run, load_model
from circuit.rl_task.reversal_flow import HAZARD, rollout

PHASES = ("D1", "GO", "D2")


def observer_episode(st, rw, ch):
    lam, out = 0.0, []
    for k in range(len(st)):
        out.append(lam)
        if ch[k] > 0 and st[k] > 0 and rw[k] in (0, 1):
            ev = np.log(9.0) * (1 if st[k] == 1 else -1) * (1 if rw[k] == 1 else -1)
            b = 1 / (1 + np.exp(-(lam + ev)))
            b = b * (1 - HAZARD) + (1 - b) * HAZARD
            lam = float(np.log(b / (1 - b)))
    return np.array(out)


def good(lam, s):
    return (lam > 0) == (s == 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--events", type=int, default=400)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    rnn, tcs, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    ch, st, rw = rec["choice"], rec["state"], rec["reward"]
    events = []
    for e in rng.permutation(ch.shape[0]):
        lam = observer_episode(st[e], rw[e], ch[e])
        for t in range(4, 98):
            if not ((ch[e, t - 3:t + 2] > 0).all() and (st[e, t - 3:t + 2] > 0).all() and rw[e, t] in (0, 1)):
                continue
            Xs, Y = st[e, t], st[e, t + 1]
            if Y == Xs or good(lam[t], Y) or not good(lam[t + 1], Y):
                continue                                                  # Y believed bad before, inferred good after
            rw_c = rw[e].copy(); rw_c[t] = 1 - rw_c[t]
            lam_c = observer_episode(st[e], rw_c, ch[e])
            if good(lam_c[t + 1], Y):
                continue                                                  # the swap must leave Y believed bad
            events.append((int(e), t))
            break
        if len(events) >= args.events:
            break
    wV = tracer.W_value
    agg = {tg: {"d": [], "step": defaultdict(float), "node": defaultdict(float), "src": defaultdict(float)} for tg in ("reveal", "value")}
    items, checks = [], []
    for e, t in events:
        lo = t - 3
        x = X[e, 3 * lo:3 * (t + 1) + 3].clone()                        # to (t+1).D2
        xc = x.clone(); xc[3 * 4, REW] = 1 - xc[3 * 4, REW]              # trial t's reward, shown at (t+1).D1
        acts = [tracer.run_forward_pass(gru_run(rnn, xx, Hp[e, 3 * lo])) for xx in (x, xc)]
        go, d2 = 3 * 4 + 1, 3 * 4 + 2
        f_go = full_flow(tracer, acts[0], acts[1], wV, go)
        f_d2 = full_flow(tracer, acts[0], acts[1], wV, d2)
        checks.append(max(abs(f_go["check"]), abs(f_d2["check"]), f_go["preact_check"], f_d2["preact_check"]))
        n_go = {nd: v for nd, v in zip(f_go["nodes"], f_go["flow"].tolist())}
        n_d2 = {nd: v for nd, v in zip(f_d2["nodes"], f_d2["flow"].tolist())}
        for tg, flows, d in (("reveal", {nd: n_d2.get(nd, 0) - n_go.get(nd, 0) for nd in set(n_go) | set(n_d2)},
                              f_d2["d_target"] - f_go["d_target"]),
                             ("value", n_d2, f_d2["d_target"])):
            agg[tg]["d"].append(d)
            for (kd, ts, fi), fl in flows.items():
                if fl != 0:
                    name = f"{'t' if ts // 3 == 3 else 't+1' if ts // 3 == 4 else f't{ts // 3 - 3:+d}'}.{PHASES[ts % 3]}"
                    agg[tg]["step"][name] += fl
                    agg[tg]["node"][f"{name}/{kd}:{fi}"] += fl
        for key in ("input", "error_r", "error_z", "error_n"):
            agg["reveal"]["src"][key] += float(f_d2["into"][key].sum() - f_go["into"][key].sum())
        items.append({"x": (x, xc), "h0": Hp[e, 3 * lo], "acts": acts,
                      "flows": {nd: n_d2.get(nd, 0) - n_go.get(nd, 0) for nd in set(n_go) | set(n_d2)},
                      "da": {nd: d for nd, d in zip(f_d2["nodes"], f_d2["da"].tolist())}})
    n = len(items)
    out = {"episodes": args.episodes, "n_events": n, "max_check": float(np.max(checks))}
    for tg, a in agg.items():
        out[tg] = {"difference_mean (inferred good minus still bad)": round(float(np.mean(a["d"])), 3),
                   "flow_through_step": {k: round(v / n, 3) for k, v in sorted(a["step"].items())},
                   "top_features": {k: round(v / n, 3) for k, v in sorted(a["node"].items(), key=lambda kv: -abs(kv[1]))[:args.top]}}
    out["reveal"]["flow_by_source"] = {k: round(v / n, 3) for k, v in agg["reveal"]["src"].items()}
    pat = defaultdict(list)
    for k in (1, 3, 5, 10):
        for rnd in (False, True):
            for it in items:
                by = defaultdict(list)
                for nd, fl in it["flows"].items():
                    if it["da"].get(nd, 0.0) != 0:
                        by[nd[1]].append((abs(fl), nd))
                chosen = []
                for lst in by.values():
                    lst.sort(key=lambda v: -v[0])
                    if rnd:
                        rest = [nd for _, nd in lst[k:]]
                        chosen += [rest[j] for j in rng.choice(len(rest), size=min(k, len(rest)), replace=False)] if rest else []
                    else:
                        chosen += [nd for _, nd in lst[:k]]
                T = it["x"][0].shape[0]
                mask = {g: torch.zeros(1, T, it["acts"][0][f"f_{g}"].shape[1], dtype=torch.bool) for g in "rzn"}
                for g, ts, fi in chosen:
                    mask[g][0, ts, fi] = True
                first = min(nd[1] for nd in it["da"] if it["da"][nd] != 0)
                steps = torch.zeros(1, T, dtype=torch.bool); steps[0, first:] = True
                value = {g: it["acts"][1][f"f_{g}"][None] for g in "rzn"}
                err = {g: it["acts"][0][f"e_{g}"][None] for g in "rzn"}
                V = run_patched(rnn, tcs, it["x"][0][None], it["h0"][None], mask, value, steps, err, head="value")[0]
                Vo = [float(it["acts"][0]["value"][j]) for j in (13, 14)]
                Vs = [float(it["acts"][1]["value"][j]) for j in (13, 14)]
                d_o, d_s, d_p = Vo[1] - Vo[0], Vs[1] - Vs[0], float(V[14] - V[13])
                if abs(d_o - d_s) >= 0.1:
                    pat[f"{'random' if rnd else 'top'}{k}"].append((d_o - d_p) / (d_o - d_s))
    out["patching_share_of_reveal_difference_carried_median"] = {k: round(float(np.median(v)), 3) for k, v in sorted(pat.items())}
    out["patching_n"] = {k: len(v) for k, v in pat.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
