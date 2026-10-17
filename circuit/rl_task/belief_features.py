"""Which features hold the belief, and which compute its update?

"Belief" elsewhere is a linear read-out of the ideal observer's belief logit (hazard 0.025) from the hidden state h.
Here: the features behind it. Closed-loop episodes (reversal_flow.rollout); read-outs fitted per step of a trial
(the state after D1, after GO, after D2), each to the belief before this trial's choice (which includes the reward
shown at this trial's D1).

track    per feature (all three gates) and step of the trial: the correlation of its activation with the ideal
         belief; the top features each way, and how much of the belief the top features predict together. Also the
         partial correlation beyond this trial's choice, its state reached and the previous outcome, which the
         belief shares with the agent's behaviour: features that hold the belief itself, not just the choice or
         the last outcome.
write    the belief read-out after the D1 step (where the previous trial's reward is shown) split exactly:
         read(h_after) = read((1 - z) h_before) (the old state the update gate keeps) + sum_f read(z d_f) a_f (each
         candidate feature's write at this D1) + read(z (b + e)) (bias and error write) + the read-out's intercept.
         Every term centred on its average over all D1 steps (uncentred, two terms of about +-20 cancel). Averaged
         per kind of previous outcome (state reached, rewarded or not), oriented so that positive = towards the state
         the outcome is evidence for; with the ideal belief before and after that outcome.
graph    the reward contrast (contrast_flow.load_windows: the core trial's reward swapped): exact contrastive flow,
         features of all three gates as nodes, with the belief read-out after (c+1).D1 as the target; top features
         and flow by source.

    python -m circuit.rl_task.belief_features --out rl_features/rl_circuits/belief_features.json
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.contrast_flow import edit, full_flow, gru_run, load_model, load_windows
from circuit.rl_task.reversal_flow import observer, rollout

PHASES = ("D1", "GO", "D2")
GATE_REC = {"r": "reset", "z": "update", "n": "hidden"}
TYPES = ("S1 rewarded", "S1 unrewarded", "S2 rewarded", "S2 unrewarded")


REFINED = {"r": "reset", "z": "update", "n": "candidate"}


def label(kind, f):
    for p in (f"rl_features/rl_refined/{REFINED[kind]}_{f}.json", f"rl_features/rl_autointerp/{GATE_REC[kind]}_{f}.json"):
        if os.path.exists(p):
            d = json.load(open(p))
            return (d.get("words") or d.get("final_label") or d.get("label") or "")[:140]
    return ""


@torch.no_grad()
def acts_all(rnn, tcs, X, Hp):
    """Activations of all features at every step (N, 3T, F) per gate, and z, n, h before / after."""
    gru = rnn.layers[0]
    x, h = X, Hp
    r = torch.sigmoid(gru.input_to_reset(x) + gru.hidden_to_reset(h))
    z = torch.sigmoid(gru.input_to_update(x) + gru.hidden_to_update(h))
    n = torch.tanh(gru.input_to_new(x) + gru.hidden_to_new(r * h))
    a = {"r": tcs["reset"].act(tcs["reset"].input_to_features(torch.cat([h, x], -1))),
         "z": tcs["update"].act(tcs["update"].input_to_features(torch.cat([h, x], -1))),
         "n": tcs["candidate"].act(tcs["candidate"].input_to_features(torch.cat([r * h, x], -1)))}
    return a, z, n, (1 - z) * h + z * n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--n_windows", type=int, default=60, help="graph: windows per core-trial pattern")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rnn, tcs, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    b = observer(rec)
    lam = np.log(np.clip(b, 1e-6, 1) / np.clip(1 - b, 1e-6, 1))           # (N, T): belief before trial t's choice
    a, z, n, h_after = acts_all(rnn, tcs, X, Hp)
    N, S = X.shape[:2]
    # read-outs of the belief from the state after each step (after D1 and later, the reward of t-1 is in)
    dec = {}
    for ph in range(3):
        H = h_after[:, ph::3].reshape(-1, 48).numpy()
        A = np.column_stack([H, np.ones(len(H))])
        w = np.linalg.lstsq(A, lam.reshape(-1), rcond=None)[0]
        r2 = 1 - ((lam.reshape(-1) - A @ w) ** 2).sum() / ((lam.reshape(-1) - lam.mean()) ** 2).sum()
        dec[ph] = (w[:48], float(w[48]), round(float(r2), 3))
    result = {"episodes": args.episodes, "readout_r2_after_each_step": {PHASES[p]: dec[p][2] for p in range(3)}}
    # track
    track = {}
    for ph in range(3):
        rows = []
        y = lam.reshape(-1)
        for g in "rzn":
            A_ = a[g][:, ph::3].reshape(-1, a[g].shape[-1]).numpy()
            on = A_.std(0) > 1e-6
            c = np.zeros(A_.shape[1])
            c[on] = [np.corrcoef(A_[:, f], y)[0, 1] for f in np.flatnonzero(on)]
            for f in np.flatnonzero(on):
                rows.append((float(c[f]), g, int(f), float((A_[:, f] > 0).mean())))
        rows.sort(key=lambda r_: -abs(r_[0]))
        top = rows[:args.top]
        cols = np.column_stack([a[g][:, ph::3].reshape(-1, a[g].shape[-1]).numpy()[:, f] for _, g, f, _ in top] + [np.ones(len(y))])
        bta = np.linalg.lstsq(cols, y, rcond=None)[0]
        track[f"at {PHASES[ph]}"] = {
            "top_features": [{"feature": f"{g}:{f}", "corr_with_belief": round(c_, 3), "active_share": round(act, 3),
                              "label": label(g, f)} for c_, g, f, act in top],
            "r2_of_belief_from_these_features": round(float(1 - ((y - cols @ bta) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3)}
    # partial correlation: beyond this trial's choice, the previous outcome (state reached, rewarded or not) and
    # this trial's state reached (which the agent's choices and recent outcomes share with the belief)
    ch_t, st_t = rec["choice"], rec["state"]
    prev_typ = np.full(lam.shape, -1)
    pt = np.where((ch_t[:, :-1] > 0) & (st_t[:, :-1] > 0) & (rec["reward"][:, :-1] >= 0),
                  (st_t[:, :-1] - 1) * 2 + (1 - np.clip(rec["reward"][:, :-1], 0, 1)).astype(int), -1)
    prev_typ[:, 1:] = pt
    C = np.column_stack([np.ones(lam.size), (ch_t.reshape(-1) == 1), (st_t.reshape(-1) == 1)]
                        + [(prev_typ.reshape(-1) == k) for k in range(4)]).astype(float)
    resid = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    y_r = resid(lam.reshape(-1))
    for ph in range(3):
        rows = []
        for g in "rzn":
            A_ = a[g][:, ph::3].reshape(-1, a[g].shape[-1]).numpy()
            for f in np.flatnonzero(A_.std(0) > 1e-6):
                xr = resid(A_[:, f])
                if xr.std() > 1e-6:
                    rows.append((float(np.corrcoef(xr, y_r)[0, 1]), g, int(f)))
        rows.sort(key=lambda r_: -abs(r_[0]))
        track[f"at {PHASES[ph]}"]["top_features_beyond_choice_and_outcomes"] = [
            {"feature": f"{g}:{f}", "partial_corr_with_belief": round(c_, 3), "label": label(g, f)} for c_, g, f in rows[:args.top]]
        track[f"at {PHASES[ph]}"]["belief_variance_left_after_choice_and_outcomes"] = round(float(y_r.var() / lam.var()), 3)
    result["track"] = track
    print("track", json.dumps({k: [(r_["feature"], r_["corr_with_belief"]) for r_ in v["top_features"]] for k, v in track.items()}),
          {k: v["r2_of_belief_from_these_features"] for k, v in track.items()}, f"({time.time() - t0:.0f}s)", flush=True)
    # write: the belief read-out's change over D1 (steps 3t, t >= 1), split exactly
    w1 = torch.tensor(dec[0][0], dtype=torch.float32)                     # read-out after D1
    D = tcs["candidate"].features_to_outputs
    Dw, Db = D.weight.detach(), D.bias.detach()
    d1 = torch.arange(3, S, 3)
    zz, aa, hb = z[:, d1], a["n"][:, d1], Hp[:, d1]
    nn_ = n[:, d1]
    n_hat = aa @ Dw.T + Db
    err = nn_ - n_hat
    per_feat = aa * torch.einsum("ntu,fu->ntf", zz, Dw.T * w1[None, :])   # read(z d_f) a_f
    bias_err = (zz * (Db + err)) @ w1
    st, rw, ch = rec["state"][:, :-1], rec["reward"][:, :-1], rec["choice"][:, :-1]
    typ = np.where((ch > 0) & (st > 0) & (rw >= 0), (st - 1) * 2 + (1 - np.clip(rw, 0, 1)).astype(int), -1)
    sign = np.where((typ == 0) | (typ == 3), 1.0, -1.0)                   # towards the state the outcome favours (S1: +)
    write = {}
    pf = per_feat.numpy()
    # every term centred on its mean over all valid D1 steps: the split then explains the belief's deviation from
    # its average (uncentred, the kept and bias / error terms are about +-20 and cancel)
    ok = typ >= 0
    cen = lambda v: v - v[ok].mean(0)
    carried = cen((((1 - zz) * hb) @ w1).numpy())
    after = cen((h_after[:, d1] @ w1).numpy())
    pf = cen(pf)
    be = cen(bias_err.numpy())
    lam_now, lam_before = lam[:, 1:], lam[:, :-1]
    for t_, name in enumerate(TYPES):
        k = typ == t_
        s_ = sign[k]
        contrib = (pf[k] * s_[:, None]).mean(0)
        top = np.argsort(-np.abs(contrib))[:args.top]
        write[name] = {"n": int(k.sum()),
                       "belief_after_D1 (read-out, minus its average)": round(float((after[k] * s_).mean()), 3),
                       "  = kept from the old state": round(float((carried[k] * s_).mean()), 3),
                       "  + this D1's candidate feature writes": round(float((pf[k].sum(-1) * s_).mean()), 3),
                       "  + bias and error write": round(float((be[k] * s_).mean()), 3),
                       "ideal belief before / after this outcome": [round(float((lam_before[k] * s_).mean()), 3),
                                                                    round(float((lam_now[k] * s_).mean()), 3)],
                       "top_feature_writes": [{"feature": f"n:{int(f)}", "write": round(float(contrib[f]), 3),
                                               "label": label("n", int(f))} for f in top]}
    result["write"] = write
    print("write", json.dumps({k: {kk: v_ for kk, v_ in v.items() if kk != "top_feature_writes"}
                               | {"top": [(r_["feature"], r_["write"]) for r_ in v["top_feature_writes"][:6]]} for k, v in write.items()}),
          f"({time.time() - t0:.0f}s)", flush=True)
    # graph: reward contrast, target = belief read-out after (c+1).D1
    windows = load_windows("rl_features/rl_trace_sequences", args.n_windows, args.seed)
    node, src, checks, dt = defaultdict(float), defaultdict(float), [], []
    for w in windows:
        c = w["c"]
        tau = 3 * c + 3
        acts = [tracer.run_forward_pass(gru_run(rnn, x, w["h0"])) for x in (w["x"], edit(w, "reward"))]
        ev_sign = 1.0 if ((w["state"] == 1) == (w["reward"] == 1)) else -1.0   # original outcome favours S1: +
        f = full_flow(tracer, acts[0], acts[1], ev_sign * w1, tau)
        checks.append(max(abs(f["check"]), f["preact_check"])); dt.append(f["d_target"])
        for (kd, ts, fi), fl in zip(f["nodes"], f["flow"].tolist()):
            if fl != 0:
                node[f"{kd}:{fi}"] += fl
        for key, v in f["into"].items():
            src[f"{key} -> features"] += float(v.sum())
        for key, v in f["target"].items():
            src[key.replace("direct_", "") + " -> read-out directly"] += v
    nw = len(windows)
    result["graph"] = {"n_windows": nw, "max_check": float(np.max(checks)),
                       "belief_change_from_the_reward_swap_mean": round(float(np.mean(dt)), 3),
                       "flow_by_source": {k: round(v / nw, 3) for k, v in sorted(src.items(), key=lambda kv: -abs(kv[1])) if abs(v) > 1e-3},
                       "top_features": [{"feature": k, "flow": round(v / nw, 3), "label": label(k[0], int(k[2:]))}
                                        for k, v in sorted(node.items(), key=lambda kv: -abs(kv[1]))[:args.top]]}
    print("graph", json.dumps({k: v for k, v in result["graph"].items() if k != "top_features"}),
          [(r_["feature"], r_["flow"]) for r_ in result["graph"]["top_features"]], flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
