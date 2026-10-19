"""Is evidence from S2 outcomes held in the memory's "S2 signal" channel (why S2 evidence is carried and S1 evidence
re-processed)?

At D2 the state reached is written mostly by the S2 features (credit_factorial.held_in_h): "reached S2" is an
active signal in h, "reached S1" mostly its absence. The lead: an S2 outcome's evidence is stored along that
channel and carried; an S1 outcome has no active channel, so its evidence is re-created by features each step.

State-signal direction d: mean h after D2 when S2 was reached minus when S1 was reached (same choice, averaged
over the two choices), closed-loop episodes. Events as in belief_axis.py (seed 17): trial u's reward swapped
(shown at v.D1), evidence oriented so it counts as positive; grouped by the state trial u reached.
  alignment     cosine of the evidence's change in h after v.D1 (original minus swapped, oriented) with d
  persistence   the change in h after v.GO, v.D2 and t.D1, projected on its direction after v.D1, over its size
                after v.D1 (how much of the written evidence is still there, unchanged in direction)
  causal        in the real GRU, from v.D1 on, after every step the swapped run's h is set equal to the
                original's along one direction: share of the choice change at t.GO removed. Directions: d; the
                belief decoder of the ideal observer (h after D1); 3 random unit directions (mean)

    python -m circuit.rl_task.state_channel --out rl_features/rl_circuits/state_channel.json [--events 1200]
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


@torch.no_grad()
def run(rnn, x, h0, clamp=None):
    """Real GRU; clamp = (from_step, unit direction u, reference hs (T, H)): after each step >= from_step, h's
    component along u is set to the reference's. Returns h after every step (T, H) and the choice logit (T,)."""
    gru, readout = rnn.layers[0], rnn.layers[-1]
    h, hs, out = h0, [], []
    for t in range(x.shape[0]):
        xt = x[t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        h = (1 - z) * h + z * n
        if clamp is not None and t >= clamp[0]:
            u, ref = clamp[1], clamp[2][t]
            h = h + ((ref - h) @ u) * u
        hs.append(h)
        o = readout(h)
        out.append(o[1] - o[2])
    return torch.stack(hs), torch.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--events", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    rnn, _, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    ch, st, rw = rec["choice"], rec["state"], rec["reward"]
    N, T = ch.shape
    h_after = torch.cat([Hp[:, 1:], Hp[:, -1:]], 1)                      # h after each step
    # state-signal direction: h after D2, S2 minus S1 reached, within each choice
    hd2 = h_after[:, 2::3]
    diffs = []
    for a in (1, 2):
        m1, m2 = (ch == a) & (st == 1), (ch == a) & (st == 2)
        diffs.append(hd2[torch.tensor(m2)].mean(0) - hd2[torch.tensor(m1)].mean(0))
    d = sum(diffs) / 2
    d = d / d.norm()
    # belief decoder direction (h after D1 -> ideal observer's belief logit)
    b = observer(rec)
    lam = np.log(np.clip(b, 1e-6, 1) / np.clip(1 - b, 1e-6, 1))
    Hs = h_after[:, 0::3].reshape(-1, 48).numpy()
    wdec = np.linalg.lstsq(np.column_stack([Hs, np.ones(len(Hs))]), lam.reshape(-1), rcond=None)[0][:48]
    bdir = torch.tensor(wdec / np.linalg.norm(wdec), dtype=torch.float32)
    rand = [torch.tensor(v / np.linalg.norm(v), dtype=torch.float32) for v in rng.standard_normal((3, 48))]
    wch = tracer.W_choice.detach().float()
    cand = [(e, u) for e in range(N) for u in range(4, 97)
            if (ch[e, u - 2:u + 3] > 0).all() and (st[e, u - 2:u + 3] > 0).all() and rw[e, u] in (0, 1) and rw[e, u + 1] in (0, 1)]
    picks = [cand[i] for i in rng.permutation(len(cand))[:args.events]]
    res = defaultdict(list)
    for e, u in picks:
        lo = u - 2
        x = X[e, 3 * lo:3 * (u + 2) + 2].clone()
        xc = x.clone(); xc[9, REW] = 1 - xc[9, REW]
        sign = 1.0 if favours_s1(st[e, u], rw[e, u]) else -1.0
        g = "S1 outcome" if st[e, u] == 1 else "S2 outcome"
        h0 = Hp[e, 3 * lo]
        ho, Lo = run(rnn, x, h0)
        hs, Ls = run(rnn, xc, h0)
        dL = float(Lo[13] - Ls[13])
        dh = sign * (ho - hs)                                            # oriented evidence in h, per step
        e9 = dh[9]
        res[(g, "alignment with state signal")].append(float(e9 @ d / e9.norm()))
        res[(g, "alignment with choice read-out")].append(float(e9 @ wch / (e9.norm() * wch.norm())))
        for ts, nm in ((10, "v.GO"), (11, "v.D2"), (12, "t.D1")):
            res[(g, f"persistence at {nm}")].append(float(dh[ts] @ e9 / (e9 @ e9)))
        if abs(dL) < 0.5:
            continue
        for nm, dirs in (("state signal", [d]), ("belief decoder", [bdir]), ("random direction", rand)):
            vals = []
            for u_ in dirs:
                _, Lc = run(rnn, xc, h0, clamp=(9, u_, ho))
                vals.append((float(Lc[13]) - float(Ls[13])) / dL)
            res[(g, f"removed: {nm}")].append(float(np.mean(vals)))
    out = {"events": len(picks), "cos_state_signal_vs_choice_readout": round(float(d @ wch / wch.norm()), 3),
           "cos_state_signal_vs_belief_decoder": round(float(d @ bdir), 3),
           "results": {f"{g} / {k}": {"median": round(float(np.median(v)), 3), "n": len(v)} for (g, k), v in sorted(res.items())}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
