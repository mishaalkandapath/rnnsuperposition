"""Does the agent act on the belief information in its memory, now or later?

The ideal observer's belief can be read from the hidden state (choice_lag.belief_subspace: four straight-line
read-outs, one per step of a trial), but just before a choice, matching that part between two runs barely changes
the choice (choice_lag.py). Is that information used later (it feeds the "which action" memory a trial on), or never?

Closed-loop episodes (reversal_flow.rollout). A recipient (episode e, trial t) and a donor (another episode, trial t')
with the same choices, states and rewards on the two trials before and ideal beliefs at least 2 belief logits apart.
At one moment (entering t.D1, before the previous reward is read; or entering t.GO, just before the choice) one part
of the recipient's hidden state is replaced by the donor's, and the real GRU runs on the recipient's own inputs to
(t+2).GO. Parts: the belief subspace; the choice read-out direction outside it; a random subspace of the same size
outside both (3 draws); the whole state (the most a swap can change). Measured: the change of the choice logit at
t.GO, (t+1).GO and (t+2).GO, oriented towards the donor's belief, and as a share of the whole-state swap's change;
and the belief read-out's change, to see whether the swapped belief information persists.

    python -m circuit.rl_task.belief_swap --out rl_features/rl_circuits/belief_swap.json
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.choice_lag import belief_subspace
from circuit.rl_task.contrast_flow import load_model
from circuit.rl_task.reversal_flow import observer, rollout


@torch.no_grad()
def run_from(rnn, x, h):
    gru, readout = rnn.layers[0], rnn.layers[-1]
    L, H = [], []
    for t in range(x.shape[0]):
        xt = x[t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        h = (1 - z) * h + z * n
        o = readout(h)
        L.append(float(o[1] - o[2])); H.append(h)
    return L, torch.stack(H)


def pairs_for(rec, lam, rng, n_max):
    ch, st, rw = rec["choice"], rec["state"], rec["reward"]
    N, T = ch.shape
    key = lambda e, t: tuple((int(ch[e, k]), int(st[e, k]), int(rw[e, k])) for k in range(t - 2, t))
    idx = defaultdict(list)
    for e in range(N):
        for t in range(5, T - 3):
            if (ch[e, t - 2:t + 3] > 0).all() and (rw[e, t - 2:t + 2] >= 0).all():
                idx[key(e, t)].append((e, t))
    pairs = []
    keys = list(idx)
    for i in rng.permutation(sum(len(v) for v in idx.values())):
        k = keys[int(rng.integers(len(keys)))]
        lst = idx[k]
        e, t = lst[int(rng.integers(len(lst)))]
        cand = [(e2, t2) for e2, t2 in lst if e2 != e and abs(lam[e2, t2] - lam[e, t]) >= 2]
        if cand:
            pairs.append((e, t) + cand[int(rng.integers(len(cand)))])
        if len(pairs) >= n_max:
            break
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    rnn, tcs, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    b = observer(rec)
    lam = np.log(np.clip(b, 1e-6, 1) / np.clip(1 - b, 1e-6, 1))
    B, r2 = belief_subspace(Hp, lam)
    Pb = B @ B.T
    w = tracer.W_choice.detach().numpy()
    wr = w - Pb @ w; wr /= np.linalg.norm(wr)
    Pr = np.outer(wr, wr)
    outside = np.eye(48) - Pb - Pr
    parts = {"belief subspace": Pb, "choice read-out (outside the belief subspace)": Pr}
    for j in range(3):
        Rq, _ = np.linalg.qr(outside @ np.random.default_rng(400 + j).normal(size=(48, B.shape[1])))
        parts[f"random subspace {j}"] = Rq @ Rq.T
    parts["whole state"] = np.eye(48)
    P_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in parts.items()}
    # belief read-out after each step (to follow the swapped information)
    dec = {}
    for ph in range(3):
        Hs = torch.cat([Hp[:, 1:], Hp[:, -1:]], 1)[:, ph::3].reshape(-1, 48).numpy()
        A = np.column_stack([Hs, np.ones(len(Hs))])
        dec[ph] = np.linalg.lstsq(A, lam.reshape(-1), rcond=None)[0]
    pairs = pairs_for(rec, lam, rng, args.pairs)
    res = {m: defaultdict(lambda: defaultdict(list)) for m in ("entering t.D1", "entering t.GO")}
    for e, t, e2, t2 in pairs:
        sign = 1.0 if lam[e2, t2] > lam[e, t] else -1.0                  # towards the donor's belief (S1 +)
        for moment, off in (("entering t.D1", 0), ("entering t.GO", 1)):
            s0 = 3 * t + off
            x = X[e, s0:3 * (t + 2) + 2]                                  # to (t+2).GO
            h_r, h_d = Hp[e, s0], Hp[e2, 3 * t2 + off]
            base, Hb = run_from(rnn, x, h_r)
            go = [1 - off, 4 - off, 7 - off]                              # positions of t.GO, (t+1).GO, (t+2).GO
            whole = None
            for k in ["whole state"] + [k for k in P_t if k != "whole state"]:
                hp = h_r - P_t[k] @ (h_r - h_d)
                Lp, Hpat = run_from(rnn, x, hp)
                d = [sign * (Lp[g] - base[g]) for g in go]
                if k == "whole state":
                    whole = d
                for i_, g in enumerate(go):
                    tag = ["t.GO", "(t+1).GO", "(t+2).GO"][i_]
                    res[moment][k][f"logit change at {tag}"].append(d[i_])
                    if abs(whole[i_]) >= 1:
                        res[moment][k][f"share of the whole-state swap at {tag}"].append(d[i_] / whole[i_])
                    ph = g % 3 if off == 0 else (g + 1) % 3
                    wv = dec[1][:48]                                      # read-out after GO
                    res[moment][k][f"belief read-out change after {tag}"].append(
                        sign * float((Hpat[g] - Hb[g]).numpy() @ wv))
    out = {"episodes": args.episodes, "n_pairs": len(pairs), "belief_readout_r2": r2,
           "ideal_belief_difference_mean_abs": round(float(np.mean([abs(lam[e2, t2] - lam[e, t]) for e, t, e2, t2 in pairs])), 3)}
    for moment, d in res.items():
        out[moment] = {k: {m: round(float(np.median(v)), 3) for m, v in dd.items()} for k, dd in d.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    for moment in res:
        print(moment)
        for k, v in out[moment].items():
            print("  ", k, v)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
