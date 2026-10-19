"""Test 1 of belief_axis.py in the attribution graph: when the next trial visits the other state, does the earlier
outcome's evidence pass through the features that read the next outcome (one quantity), or around them (two
separate values)?

Events as in belief_axis.py (seed 17): trial u's reward swapped (shown at v.D1), target the choice logit at t.GO,
oriented so u's evidence counts as positive; exact contrastive flow (contrast_flow.full_flow). Grouped by the state
trial u reached (S1 / S2), whether trial v reached the same state or the other one, and whether v's outcome agreed
with u's evidence; common transitions on trial v only. Per group: the target change at t.GO, the flow through the
features at t.D1 (where v's outcome is read) and at v.D2, their share of the target change, and the top t.D1
features. Two separate values predict little flow through t.D1 features after an other-state visit (u's value is
not involved in reading v's outcome); one quantity predicts flow through them after both kinds of visit.

    python -m circuit.rl_task.test1_flow --out rl_features/rl_circuits/test1_flow.json [--events 3000]
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from circuit.rl_task.belief_axis import favours_s1
from circuit.rl_task.contrast_flow import REW, full_flow, gru_run, load_model
from circuit.rl_task.reversal_flow import rollout


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--events", type=int, default=3000)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    rnn, tcs, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    ch, st, rw = rec["choice"], rec["state"], rec["reward"]
    cand = [(e, u) for e in range(ch.shape[0]) for u in range(4, 97)
            if (ch[e, u - 2:u + 3] > 0).all() and (st[e, u - 2:u + 3] > 0).all() and rw[e, u] in (0, 1) and rw[e, u + 1] in (0, 1)
            and st[e, u + 1] == ch[e, u + 1]]                                  # trial v: common transition
    picks = [cand[i] for i in rng.permutation(len(cand))[:args.events]]
    agg = defaultdict(lambda: {"d": [], "d1": [], "d2": [], "feat": defaultdict(float)})
    for e, u in picks:
        lo, v = u - 2, u + 1
        x = X[e, 3 * lo:3 * (u + 2) + 2].clone()
        xc = x.clone(); xc[9, REW] = 1 - xc[9, REW]
        sign = 1.0 if favours_s1(st[e, u], rw[e, u]) else -1.0
        acts = [tracer.run_forward_pass(gru_run(rnn, xx, Hp[e, 3 * lo])) for xx in (x, xc)]
        f = full_flow(tracer, acts[0], acts[1], sign * tracer.W_choice, 13)
        same = "same state" if st[e, v] == st[e, u] else "other state"
        agree = "agreeing" if favours_s1(st[e, v], rw[e, v]) == (sign > 0) else "disagreeing"
        g = agg[(f"u at S{int(st[e, u])}", same, agree)]
        g["d"].append(f["d_target"])
        g["d1"].append(sum(fl for (kd, ts, fi), fl in zip(f["nodes"], f["flow"].tolist()) if ts == 12))
        g["d2"].append(sum(fl for (kd, ts, fi), fl in zip(f["nodes"], f["flow"].tolist()) if ts == 11))
        for (kd, ts, fi), fl in zip(f["nodes"], f["flow"].tolist()):
            if ts == 12 and fl != 0:
                g["feat"][f"{kd}:{fi}"] += fl
    out = {"events": len(picks), "groups": {}}
    for key in sorted(agg):
        g = agg[key]; n = len(g["d"])
        d, d1, d2 = (float(np.mean(g[k])) for k in ("d", "d1", "d2"))
        out["groups"][" / ".join(key)] = {
            "n": n, "target_change_mean": round(d, 3), "flow_through_tD1_features": round(d1, 3),
            "flow_through_vD2_features": round(d2, 3), "tD1_share_of_target": round(d1 / d, 3) if d else None,
            "top_tD1_features": {k: round(v / n, 3) for k, v in sorted(g["feat"].items(), key=lambda kv: -abs(kv[1]))[:args.top]}}
        print(" / ".join(key), out["groups"][" / ".join(key)], flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
