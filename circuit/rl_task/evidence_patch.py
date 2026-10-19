"""Causal check of the shared evidence route: do the features that carry both S1 and S2 evidence in the attribution
graph (belief_axis.py) actually carry both, in the real GRU?

Events as in belief_axis.py: closed-loop episodes; trial u's reward swapped (shown at v.D1, v = u + 1); target the
choice logit at t.GO (t = u + 2). Group by the state trial u reached (S1 outcome / S2 outcome). Exact contrastive
flow per event (contrast_flow.full_flow), oriented so the evidence counts as positive.

Shared carriers: at each of v.D2, t.D1 and t.GO, the features among the top-k by mean flow in BOTH groups, chosen on
half of the events. On the other half, in the real GRU (circuit_checks.run_patched, gates = reconstruction + held
error from v.D2 on):
  necessity    run the original; the carriers set to their values in the swapped run: share of the target change
               removed, (L_o - L_patched) / (L_o - L_s)
  sufficiency  run the swapped input; the carriers set to their original values: share restored,
               (L_patched - L_s) / (L_o - L_s)
Also each group's own top-k features per step (S1 top, S2 top: where does each kind of evidence go?). Control: as
many random features per step that changed between the runs (not in the tested set). Medians per group, over
events whose target change is at least 0.5. The evidence also reaches t.GO through the carried memory and through
features that are not carriers, so shares below 1 are expected; the test is whether the shared carriers matter
for S1 and S2 evidence alike, and far more than random features.

    python -m circuit.rl_task.evidence_patch --out rl_features/rl_circuits/evidence_patch.json [--events 1200]
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.belief_axis import favours_s1
from circuit.rl_task.circuit_checks import run_patched
from circuit.rl_task.contrast_flow import REW, full_flow, gru_run, load_model
from circuit.rl_task.reversal_flow import rollout

STEPS = {11: "v.D2", 12: "t.D1", 13: "t.GO"}
TAU = 13


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--events", type=int, default=1200)
    ap.add_argument("--top", type=int, default=8)
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
    items = []
    for e, u in picks:
        lo = u - 2
        x = X[e, 3 * lo:3 * (u + 2) + 2].clone()
        xc = x.clone(); xc[9, REW] = 1 - xc[9, REW]
        sign = 1.0 if favours_s1(st[e, u], rw[e, u]) else -1.0
        acts = [tracer.run_forward_pass(gru_run(rnn, xx, Hp[e, 3 * lo])) for xx in (x, xc)]
        f = full_flow(tracer, acts[0], acts[1], sign * tracer.W_choice, TAU)
        flow = {nd: fl for nd, fl in zip(f["nodes"], f["flow"].tolist()) if nd[1] in STEPS}
        changed = {nd for nd, da in zip(f["nodes"], f["da"].tolist()) if nd[1] in STEPS and da != 0}
        items.append({"group": "S1 outcome" if st[e, u] == 1 else "S2 outcome", "x": (x, xc), "h0": Hp[e, 3 * lo],
                      "acts": acts, "flow": flow, "changed": changed})
    print(f"{len(items)} events, flows done ({time.time() - t0:.0f}s)", flush=True)
    order = rng.permutation(len(items))
    sel, test = [items[i] for i in order[: len(items) // 2]], [items[i] for i in order[len(items) // 2:]]
    # shared carriers, chosen on the selection half
    mean = {g: defaultdict(float) for g in ("S1 outcome", "S2 outcome")}
    n = defaultdict(int)
    for it in sel:
        n[it["group"]] += 1
        for nd, fl in it["flow"].items():
            mean[it["group"]][nd] += fl
    carriers, top_s1, top_s2 = [], [], []
    for ts in STEPS:
        tops = [set(sorted((nd for nd in mean[g] if nd[1] == ts), key=lambda nd: -abs(mean[g][nd]))[:args.top])
                for g in mean]
        carriers += sorted(tops[0] & tops[1]); top_s1 += sorted(tops[0]); top_s2 += sorted(tops[1])
    name = lambda nd: f"{STEPS[nd[1]]}/{nd[0]}:{nd[2]}"
    res = defaultdict(list)
    for it in test:
        x, xc = it["x"]
        T = x.shape[0]
        Lo, Ls = (float(a["choice"][TAU]) for a in it["acts"])
        if abs(Lo - Ls) < 0.5:
            continue
        def random_like(nodes):
            per_step, pool, out_ = defaultdict(int), defaultdict(list), []
            for nd in nodes:
                per_step[nd[1]] += 1
            for nd in it["changed"]:
                if nd not in nodes:
                    pool[nd[1]].append(nd)
            for ts, k in per_step.items():
                p = pool[ts]
                out_ += [p[j] for j in rng.choice(len(p), size=min(k, len(p)), replace=False)] if p else []
            return out_
        for label, nodes in (("shared carriers", carriers), ("random features", random_like(carriers)),
                             ("S1 top features", top_s1), ("S2 top features", top_s2),
                             ("random features (top-set size)", random_like(top_s1))):
            mask = {g: torch.zeros(1, T, it["acts"][0][f"f_{g}"].shape[1], dtype=torch.bool) for g in "rzn"}
            for g, ts, fi in nodes:
                mask[g][0, ts, fi] = True
            steps = torch.zeros(1, T, dtype=torch.bool); steps[0, min(STEPS):] = True
            for test_name, src, dst in (("necessity", 0, 1), ("sufficiency", 1, 0)):
                value = {g: it["acts"][dst][f"f_{g}"][None] for g in "rzn"}
                err = {g: it["acts"][src][f"e_{g}"][None] for g in "rzn"}
                L = float(run_patched(rnn, tcs, it["x"][src][None], it["h0"][None], mask, value, steps, err)[0, TAU])
                share = (Lo - L) / (Lo - Ls) if test_name == "necessity" else (L - Ls) / (Lo - Ls)
                res[(it["group"], label, test_name)].append(share)
    out = {"events": len(items), "selection": len(sel), "test": len(test), "top_k": args.top,
           "shared_carriers": [name(nd) for nd in carriers],
           "S1_top_features": [name(nd) for nd in top_s1], "S2_top_features": [name(nd) for nd in top_s2],
           "carrier_mean_flow_selection": {name(nd): {g: round(mean[g][nd] / max(n[g], 1), 3) for g in mean} for nd in carriers},
           "median_share": {f"{g} / {lab} / {tn}": {"median": round(float(np.median(v)), 3), "n": len(v)}
                            for (g, lab, tn), v in sorted(res.items())}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
