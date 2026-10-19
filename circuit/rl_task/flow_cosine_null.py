"""How meaningful are the S1-vs-S2 evidence flow cosines of belief_axis.py? Nulls, a noise ceiling and intervals.

Same events as belief_axis.py (seed 17): trial u's reward swapped, target the choice logit at t.GO, flow oriented so
the evidence counts as positive; per step the mean flow vector of S1-outcome events and of S2-outcome events (one
entry per feature of all three gates). Per step:
  observed       cosine of the two groups' mean flow vectors (as in belief_axis)
  bootstrap      95% interval, events resampled within each group
  chance         feature-permutation null: one group's mean flows shuffled across that step's features (sizes
                 kept, feature identity destroyed); share of permutations at least as high = p
  ceiling        split-half within a group: each group split at random into halves, cosine between the halves'
                 mean flow vectors (mean of the S1 and S2 values); the most sampling noise allows
  label shuffle  the S1 / S2 labels permuted across events; the cosine between the shuffled groups. If the observed
                 cosine is inside this distribution, S1 and S2 evidence use the features indistinguishably;
                 p_lower = share of shuffles at or below the observed

    python -m circuit.rl_task.flow_cosine_null --out rl_features/rl_circuits/flow_cosine_null.json [--events 1200]
"""
import argparse
import json
import os
import time

import numpy as np

from circuit.rl_task.belief_axis import STEP_NAMES, favours_s1
from circuit.rl_task.contrast_flow import REW, full_flow, gru_run, load_model
from circuit.rl_task.reversal_flow import rollout


def cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--events", type=int, default=1200)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    rnn, tcs, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    ch, st, rw = rec["choice"], rec["state"], rec["reward"]
    cand = [(e, u) for e in range(ch.shape[0]) for u in range(4, 97)
            if (ch[e, u - 2:u + 3] > 0).all() and (st[e, u - 2:u + 3] > 0).all() and rw[e, u] in (0, 1) and rw[e, u + 1] in (0, 1)]
    picks = [cand[i] for i in rng.permutation(len(cand))[:args.events]]
    F = {k: tracer.n_feats[k] if hasattr(tracer, "n_feats") else None for k in "rzn"}
    rows, groups = {ts: [] for ts in STEP_NAMES}, []
    keyidx = {}
    for e, u in picks:
        lo = u - 2
        x = X[e, 3 * lo:3 * (u + 2) + 2].clone()
        xc = x.clone(); xc[9, REW] = 1 - xc[9, REW]
        sign = 1.0 if favours_s1(st[e, u], rw[e, u]) else -1.0
        acts = [tracer.run_forward_pass(gru_run(rnn, xx, Hp[e, 3 * lo])) for xx in (x, xc)]
        f = full_flow(tracer, acts[0], acts[1], sign * tracer.W_choice, 13)
        vec = {ts: {} for ts in STEP_NAMES}
        for (kd, ts, fi), fl in zip(f["nodes"], f["flow"].tolist()):
            if ts in STEP_NAMES and fl != 0:
                k = f"{kd}:{fi}"
                keyidx.setdefault(k, len(keyidx))
                vec[ts][k] = fl
        for ts in STEP_NAMES:
            rows[ts].append(vec[ts])
        groups.append(0 if st[e, u] == 1 else 1)
    groups = np.array(groups)
    print(f"{len(groups)} events ({time.time() - t0:.0f}s)", flush=True)
    out = {"events": len(groups), "n_S1": int((groups == 0).sum()), "n_S2": int((groups == 1).sum()),
           "n_perm": args.n_perm, "steps": {}}
    for ts, sn in STEP_NAMES.items():
        M = np.zeros((len(groups), len(keyidx)))
        for i, v in enumerate(rows[ts]):
            for k, fl in v.items():
                M[i, keyidx[k]] = fl
        M = M[:, np.abs(M).sum(0) > 0]
        g1, g2 = M[groups == 0], M[groups == 1]
        obs = cos(g1.mean(0), g2.mean(0))
        boot = [cos(g1[rng.integers(0, len(g1), len(g1))].mean(0), g2[rng.integers(0, len(g2), len(g2))].mean(0))
                for _ in range(args.n_perm)]
        m2 = g2.mean(0)
        chance = [cos(g1.mean(0), rng.permutation(m2)) for _ in range(args.n_perm)]
        ceil = []
        for _ in range(args.n_perm):
            c = []
            for g in (g1, g2):
                p = rng.permutation(len(g)); h = len(g) // 2
                c.append(cos(g[p[:h]].mean(0), g[p[h:]].mean(0)))
            ceil.append(np.mean(c))
        shuf = []
        for _ in range(args.n_perm):
            lab = rng.permutation(groups)
            shuf.append(cos(M[lab == 0].mean(0), M[lab == 1].mean(0)))
        q = lambda a: [round(float(np.quantile(a, 0.025)), 3), round(float(np.quantile(a, 0.975)), 3)]
        out["steps"][sn] = {"observed": round(obs, 3), "bootstrap_95": q(boot),
                            "chance_95": q(chance), "chance_p": round(float(np.mean(np.array(chance) >= obs)), 4),
                            "split_half_ceiling_median": round(float(np.median(ceil)), 3), "split_half_ceiling_95": q(ceil),
                            "label_shuffle_95": q(shuf), "label_shuffle_p_lower": round(float(np.mean(np.array(shuf) <= obs)), 4)}
        print(sn, out["steps"][sn], flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
