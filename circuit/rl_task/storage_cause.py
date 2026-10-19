"""Follow-up to storage_asymmetry.py: is the S2 evidence held on the choice read-out between choices what produces
its delayed effect, and why is its immediate effect small?

Events as in storage_asymmetry.py (seed 17). Per event, oriented so trial u's evidence counts as positive:
  E1, E2       the effect on the choice logit at v's choice and at t's choice (a trial later)
  a_vGO, a_vD2, a_tD1   the evidence's component on the choice read-out after v.GO (= E1), v.D2 and t.D1, in logits
               (difference in h between the runs, times the choice read-out weights)
1. Removal. In the ORIGINAL run, after v.D2 and t.D1, h's component along a direction is set to the swapped run's
   (that part of the evidence removed between the choices); the later effect left, E2_removed, as a share of E2.
   Directions: the choice read-out; the belief decoder; 3 random unit directions (mean).
2. Event by event. Correlation across events of E2 with E1, with a_vD2 and with a_tD1; mean read-out component at
   each step; mean amount added on the read-out at v.D2 (a_vD2 - a_vGO).
3. The small immediate effect: E1 against how strongly the agent already preferred an action at v's choice
   (|choice logit| in the original run, binned), for S1 and S2 outcomes (saturation?).

    python -m circuit.rl_task.storage_cause --out rl_features/rl_circuits/storage_cause.json [--events 3000]
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.belief_axis import favours_s1
from circuit.rl_task.contrast_flow import REW, load_model
from circuit.rl_task.reversal_flow import observer, rollout

STEPS = (11, 12)


@torch.no_grad()
def run(rnn, x, h0, u=None, ref=None):
    gru, readout = rnn.layers[0], rnn.layers[-1]
    h, hs, out = h0, [], []
    for t in range(x.shape[0]):
        xt = x[t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        h = (1 - z) * h + z * n
        if u is not None and t in STEPS:
            h = h + ((ref[t] - h) @ u) * u
        hs.append(h)
        o = readout(h)
        out.append(o[1] - o[2])
    return torch.stack(hs), torch.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--events", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    rnn, _, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    ch, st, rw = rec["choice"], rec["state"], rec["reward"]
    w = tracer.W_choice.detach().float(); wu = w / w.norm()
    b = observer(rec)
    lam = np.log(np.clip(b, 1e-6, 1) / np.clip(1 - b, 1e-6, 1))
    h_after = torch.cat([Hp[:, 1:], Hp[:, -1:]], 1)
    Hs = h_after[:, 0::3].reshape(-1, 48).numpy()
    wdec = np.linalg.lstsq(np.column_stack([Hs, np.ones(len(Hs))]), lam.reshape(-1), rcond=None)[0][:48]
    bdir = torch.tensor(wdec / np.linalg.norm(wdec), dtype=torch.float32)
    rand = [torch.tensor(v / np.linalg.norm(v), dtype=torch.float32) for v in rng.standard_normal((3, 48))]
    cand = [(e, u) for e in range(ch.shape[0]) for u in range(4, 97)
            if (ch[e, u - 2:u + 3] > 0).all() and (st[e, u - 2:u + 3] > 0).all() and rw[e, u] in (0, 1) and rw[e, u + 1] in (0, 1)]
    picks = [cand[i] for i in rng.permutation(len(cand))[:args.events]]
    rows = defaultdict(lambda: defaultdict(list))
    for e, u in picks:
        lo = u - 2
        x = X[e, 3 * lo:3 * (u + 2) + 2].clone()
        xc = x.clone(); xc[9, REW] = 1 - xc[9, REW]
        sign = 1.0 if favours_s1(st[e, u], rw[e, u]) else -1.0
        h0 = Hp[e, 3 * lo]
        ho, Lo = run(rnn, x, h0)
        hs, Ls = run(rnn, xc, h0)
        g = "S1 outcome" if st[e, u] == 1 else "S2 outcome"
        r_ = rows[g]
        E1, E2 = sign * float(Lo[10] - Ls[10]), sign * float(Lo[13] - Ls[13])
        dh = sign * (ho - hs)
        r_["E1"].append(E1); r_["E2"].append(E2)
        for ts, nm in ((10, "a_vGO"), (11, "a_vD2"), (12, "a_tD1")):
            r_[nm].append(float(dh[ts] @ w))
        r_["prior |logit| at v's choice"].append(abs(float(Lo[10])))
        if abs(E2) > 0.3:
            for nm, dirs in (("choice read-out", [wu]), ("belief decoder", [bdir]), ("random", rand)):
                left = np.mean([sign * float(run(rnn, x, h0, u_, hs)[1][13] - Ls[13]) for u_ in dirs])
                r_[f"E2 left after removing {nm}"].append(left / E2)
    out = {"events": len(picks), "groups": {}}
    for g, r_ in sorted(rows.items()):
        A = {k: np.array(v) for k, v in r_.items()}
        res = {"n": len(A["E1"]),
               "median E1": round(float(np.median(A["E1"])), 3), "median E2": round(float(np.median(A["E2"])), 3),
               "mean read-out component after v.GO / v.D2 / t.D1": [round(float(A[k].mean()), 3) for k in ("a_vGO", "a_vD2", "a_tD1")],
               "mean added on the read-out at v.D2": round(float((A["a_vD2"] - A["a_vGO"]).mean()), 3),
               "corr(E2, E1)": round(float(np.corrcoef(A["E2"], A["E1"])[0, 1]), 3),
               "corr(E2, read-out after v.D2)": round(float(np.corrcoef(A["E2"], A["a_vD2"])[0, 1]), 3),
               "corr(E2, read-out after t.D1)": round(float(np.corrcoef(A["E2"], A["a_tD1"])[0, 1]), 3)}
        for nm in ("choice read-out", "belief decoder", "random"):
            k = f"E2 left after removing {nm}"
            res[f"median share of E2 left after removing {nm} between choices"] = round(float(np.median(A[k])), 3)
        bins = [0, 3, 6, 9, 100]
        P = A["prior |logit| at v's choice"]
        res["median E1 by prior |logit| at v's choice"] = {
            f"{bins[i]}-{bins[i + 1]}": [round(float(np.median(A["E1"][(P >= bins[i]) & (P < bins[i + 1])])), 3) if ((P >= bins[i]) & (P < bins[i + 1])).any() else None,
                                         int(((P >= bins[i]) & (P < bins[i + 1])).sum())] for i in range(len(bins) - 1)}
        out["groups"][g] = res
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
