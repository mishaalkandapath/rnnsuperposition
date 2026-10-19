"""Does the asymmetric storage of S1 and S2 evidence explain the agent's deviations from the ideal observer?

Events as in belief_axis.py (seed 17): trial u's reward swapped (shown at v.D1, v = u + 1); effects oriented so u's
evidence counts as positive. Grouped by the state trial u reached (S1 / S2 outcome), whether trial v reached the
same or the other state, and whether v's outcome agreed with u's evidence.

1. Behaviour. Per group, for the agent (choice logit), the ideal observer (belief logit) and the fitted Bayesian
   model (policy logit; symmetric in S1 / S2 apart from its choice bias and perseveration): the effect right after
   u's outcome (at v's choice) E1, the effect a trial later (at t's choice) E2, and survival E2 / E1 (medians).
   The deviation to explain is where the agent treats S1 and S2 outcomes differently and the models do not.
2. Causal. In the real GRU, after trial v's D2 and t's D1 (between the two choices, where S2 evidence sits on the
   choice read-out and S1 evidence does not), the swapped run's hidden state is set equal to the original's along
   one direction; the agent's survival is measured again. Directions: the choice read-out; the ideal observer's
   belief decoder (h after D1); 3 random unit directions (mean of the three). If the storage causes the deviation,
   clamping the read-out should cut S2 evidence's survival much more than S1's and remove the asymmetry.

    python -m circuit.rl_task.storage_asymmetry --out rl_features/rl_circuits/storage_asymmetry.json [--events 3000]
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.bayes_vs_mb import bayes_model
from circuit.rl_task.belief_axis import favours_s1, model_params, observer_two
from circuit.rl_task.contrast_flow import REW, load_model
from circuit.rl_task.reversal_flow import observer, rollout

CLAMP_STEPS = (11, 12)                                                    # after v.D2 and t.D1


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
        if u is not None and t in CLAMP_STEPS:
            h = h + ((ref[t] - h) @ u) * u
        hs.append(h)
        o = readout(h)
        out.append(o[1] - o[2])
    return torch.stack(hs), torch.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--events", type=int, default=3000)
    ap.add_argument("--fits", default="rl_features/rl_circuits/bayes_vs_mb.json")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    rnn, _, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    ch, st, rw = rec["choice"], rec["state"], rec["reward"]
    _, p_by = model_params(json.load(open(args.fits))["models"])
    wch = tracer.W_choice.detach().float(); wch = wch / wch.norm()
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
    res = defaultdict(lambda: defaultdict(list))
    for e, u in picks:
        lo, v, t = u - 2, u + 1, u + 2
        x = X[e, 3 * lo:3 * t + 2].clone()
        xc = x.clone(); xc[9, REW] = 1 - xc[9, REW]
        sign = 1.0 if favours_s1(st[e, u], rw[e, u]) else -1.0
        h0 = Hp[e, 3 * lo]
        ho, Lo = run(rnn, x, h0)
        _, Ls = run(rnn, xc, h0)
        E1, E2 = sign * float(Lo[10] - Ls[10]), sign * float(Lo[13] - Ls[13])
        rw_c = rw[e].copy(); rw_c[u] = 1 - rw_c[u]
        lo_, ls_ = (observer_two(st[e], r_, ch[e], t) for r_ in (rw[e], rw_c))
        O1, O2 = sign * (lo_[v] - ls_[v]), sign * (lo_[t] - ls_[t])
        S, C = torch.tensor(np.stack([st[e], st[e]])), torch.tensor(np.stack([ch[e], ch[e]]))
        R = torch.tensor(np.stack([rw[e], rw_c])).clamp(min=0).float()
        V = (C > 0) & (S > 0) & torch.tensor(np.stack([rw[e] >= 0, rw_c >= 0]))
        with torch.no_grad():
            byl, _ = bayes_model(p_by, S, R, C, V)
        B1, B2 = sign * float(byl[0, v] - byl[1, v]), sign * float(byl[0, t] - byl[1, t])
        g = "S1 outcome" if st[e, u] == 1 else "S2 outcome"
        same = "same state" if st[e, v] == st[e, u] else "other state"
        agree = "agreeing" if favours_s1(st[e, v], rw[e, v]) == (sign > 0) else "disagreeing"
        keys = [(g, "all"), (g, same), (g, f"{same}, {agree}")]
        clamp = {}
        if abs(E1) > 0.5:
            for nm, dirs in (("clamp choice read-out", [wch]), ("clamp belief decoder", [bdir]), ("clamp random", rand)):
                clamp[nm] = float(np.mean([sign * float(run(rnn, xc, h0, u_, ho)[1][13] - Ls[13]) for u_ in dirs])) / E1
        for k in keys:
            r_ = res[k]
            r_["agent E1"].append(E1); r_["agent E2"].append(E2); r_["observer E1"].append(O1); r_["observer E2"].append(O2)
            r_["Bayesian E1"].append(B1); r_["Bayesian E2"].append(B2)
            if abs(E1) > 0.5:
                r_["agent survival"].append(E2 / E1)
                for nm, val in clamp.items():
                    r_[f"agent survival, {nm}"].append(val)
            if abs(O1) > 1e-3:
                r_["observer survival"].append(O2 / O1)
            if abs(B1) > 1e-3:
                r_["Bayesian survival"].append(B2 / B1)
    out = {"events": len(picks), "groups": {}}
    for (g, k), d in sorted(res.items()):
        out["groups"][f"{g} / {k}"] = {m: round(float(np.median(v)), 3) for m, v in d.items()}
        out["groups"][f"{g} / {k}"]["n"] = len(d["agent E1"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
