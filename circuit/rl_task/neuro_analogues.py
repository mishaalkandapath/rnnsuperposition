"""Mechanistic checks behind four systems-neuroscience parallels (closed-loop episodes, reversal_flow.rollout).

coincidence  How do the next-D1 detectors combine the reward with the remembered state? A candidate feature's
             preactivation splits exactly into an input part (W_x x: the reward and the fixation inputs at D1), a
             memory part (W_h (r h): the remembered state and history) and its bias. Per previous outcome (state
             reached, rewarded or not), each detector's mean parts, its threshold, and how often the input part alone
             or the memory part alone (each with the bias, the other at its mean over all D1 steps) would cross the
             threshold: an AND made by adding then thresholding (coincidence detection). The reset gate also sees the
             reward input, so the memory part is also given with the reset gate computed as if the reward had been
             the other one (the reward's gating of which memory is read).
persistence  How long the state reached stays readable from h after it is shown at D2 (an eligibility trace for
             credit assignment): held-out accuracy of a linear read-out of the state at D2, the next D1, GO, D2 and
             the D1 after, on rare-transition trials only (on common ones the choice gives the state away).
cross_time   Static or dynamic codes: linear read-outs of three variables trained on h after one step of a trial
             (D1, GO, D2) and tested after each (held-out correlation of prediction and target): the belief information (ideal observer's belief
             logit), the action preference (the choice logit at this trial's GO) and the previous reward. And for
             each variable, the overlap of the features most correlated with it at different steps (Jaccard, top
             10 of each gate; candidate features are mostly step-specific, so this is expected to be low).
run_length   Choice history: at the D1 that reads a missed or received reward, the "stay anyway" reader (candidate
             220) and the "switch" readers (candidates 208, 11, reset 4), and the pull of the choice logit towards
             the previous choice beyond the belief information, by how many trials the agent has kept that choice,
             after a reward and after a miss.

    python -m circuit.rl_task.neuro_analogues --out rl_features/rl_circuits/neuro_analogues.json
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.bayes_vs_mb import basis
from circuit.rl_task.contrast_flow import load_model
from circuit.rl_task.reversal_flow import observer, rollout
from circuit.rl_task.trace_role import keep_out_basis

TYPES = ("S1 rewarded", "S1 unrewarded", "S2 rewarded", "S2 unrewarded")
DETECTORS = (77, 18, 215, 208, 66, 118, 165, 220, 11)


def r2_heldout(Xtr, ytr, Xte, yte):
    """Held-out correlation of a linear read-out's prediction with the target (insensitive to an offset or scale
    change between the training and test steps)."""
    A = np.column_stack([Xtr, np.ones(len(Xtr))]); w = np.linalg.lstsq(A, ytr, rcond=None)[0]
    p = np.column_stack([Xte, np.ones(len(Xte))]) @ w
    return float(np.corrcoef(p, yte)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    rnn, tcs, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    N = X.shape[0]
    gru, tc = rnn.layers[0], tcs["candidate"]
    ch, st, rw, L = rec["choice"], rec["state"], rec["reward"], rec["logit"]
    out = {"episodes": args.episodes}
    # ---- coincidence
    with torch.no_grad():
        d1 = torch.arange(3, X.shape[1], 3)                               # D1 of trials 1..99 (reads trial j-1's outcome)
        x, h = X[:, d1], Hp[:, d1]
        r = torch.sigmoid(gru.input_to_reset(x) + gru.hidden_to_reset(h))
        Wh, Wx = tracer.W_n_h.detach(), tracer.W_n_x.detach()
        bias = tc.input_to_features.bias.detach()
        thr = torch.exp(tc.act.threshold.detach())                        # stored as a log-threshold
        p_in = x @ Wx.T
        p_mem = (r * h) @ Wh.T
        # the reset gate sees the reward input too: the memory part with the reset gate computed as if the reward
        # had been the other one (how much of the memory part's dependence on the reward comes through that gating)
        xs = x.clone(); xs[..., 6] = 1 - xs[..., 6]
        r_sw = torch.sigmoid(gru.input_to_reset(xs) + gru.hidden_to_reset(h))
        p_mem_sw = (r_sw * h) @ Wh.T
    p_in, p_mem, bias, thr, p_mem_sw = p_in.numpy(), p_mem.numpy(), bias.numpy(), thr.numpy(), p_mem_sw.numpy()
    prev = np.where((ch[:, :-1] > 0) & (st[:, :-1] > 0) & (rw[:, :-1] >= 0),
                    (st[:, :-1] - 1) * 2 + (1 - np.clip(rw[:, :-1], 0, 1)).astype(int), -1)
    ok = prev >= 0
    m_in, m_mem = p_in[ok].mean(0), p_mem[ok].mean(0)
    coinc = {}
    for f in DETECTORS:
        rows = {}
        for k, name in enumerate(TYPES):
            sel = prev == k
            tot = p_in[sel][:, f] + p_mem[sel][:, f] + bias[f]
            rows[name] = {"input_part": round(float(p_in[sel][:, f].mean()), 3), "memory_part": round(float(p_mem[sel][:, f].mean()), 3),
                          "memory_part_with_the_reset_gate_of_the_other_reward": round(float(p_mem_sw[sel][:, f].mean()), 3),
                          "fires_with_the_reset_gate_of_the_other_reward": round(float((p_in[sel][:, f] + p_mem_sw[sel][:, f] + bias[f] > thr[f]).mean()), 3),
                          "fires": round(float((tot > thr[f]).mean()), 3),
                          "would_fire_on_input_alone": round(float((p_in[sel][:, f] + m_mem[f] + bias[f] > thr[f]).mean()), 3),
                          "would_fire_on_memory_alone": round(float((p_mem[sel][:, f] + m_in[f] + bias[f] > thr[f]).mean()), 3)}
        coinc[f"n:{f}"] = {"bias": round(float(bias[f]), 3), "threshold": round(float(thr[f]), 3), "by_previous_outcome": rows}
    out["coincidence"] = coinc
    print("coincidence", json.dumps({k: {t_: (v_["fires"], v_["would_fire_on_input_alone"], v_["would_fire_on_memory_alone"])
                                         for t_, v_ in v["by_previous_outcome"].items()} for k, v in coinc.items()}), flush=True)
    # ---- persistence of the state reached
    h_after = torch.cat([Hp[:, 1:], Hp[:, -1:]], 1).numpy()            # h after each step
    perm = rng.permutation(N); tr, te = perm[:N // 2], perm[N // 2:]
    pers = {}
    for off, name in ((2, "after its D2 (state shown)"), (3, "after the next D1"), (4, "after the next GO"),
                      (5, "after the next D2"), (6, "after the D1 after")):
        rows_t = np.arange(0, 96)
        Hs = h_after[:, 3 * rows_t + off]                                 # (N, 96, 48)
        y = (st[:, rows_t] == 1).astype(float)
        # rare transitions only: there the state reached is not implied by the choice, so a read-out must use the
        # state's own memory (on common trials the choice and later behaviour give it away)
        vm = (st[:, rows_t] > 0) & (ch[:, rows_t] > 0) & (st[:, rows_t] != ch[:, rows_t])
        A_tr, y_tr = Hs[tr][vm[tr]], y[tr][vm[tr]]
        A_te, y_te = Hs[te][vm[te]], y[te][vm[te]]
        w = np.linalg.lstsq(np.column_stack([A_tr, np.ones(len(A_tr))]), y_tr * 2 - 1, rcond=None)[0]
        acc = float(((np.column_stack([A_te, np.ones(len(A_te))]) @ w > 0) == (y_te > 0.5)).mean())
        # beyond what the current choice and belief already say: accuracy among trials whose next choice is the same
        # control: the same read-out of which action was chosen (which on these trials is the other state)
        pers[name] = {"accuracy_rare_trials": round(acc, 3), "n_test": int(len(y_te))}
    out["persistence_of_state_reached"] = pers
    print("persistence", json.dumps(pers), flush=True)
    # ---- cross-temporal generalisation
    b = observer(rec)
    lam = np.log(np.clip(b, 1e-6, 1) / np.clip(1 - b, 1e-6, 1))
    prev_r = np.concatenate([np.zeros((N, 1)), np.clip(rw[:, :-1], 0, 1)], 1)
    vars_ = {"belief information": lam, "action preference (this trial's choice logit)": L, "previous reward": prev_r}
    valid = (ch > 0); valid[:, :2] = False
    cross = {}
    for vn, y in vars_.items():
        mat = {}
        for ph_tr, n_tr in ((0, "D1"), (1, "GO"), (2, "D2")):
            Htr = h_after[tr][:, ph_tr::3][valid[tr]]; ytr = y[tr][valid[tr]]
            row = {}
            for ph_te, n_te in ((0, "D1"), (1, "GO"), (2, "D2")):
                Hte = h_after[te][:, ph_te::3][valid[te]]; yte = y[te][valid[te]]
                row[f"tested after {n_te}"] = round(r2_heldout(Htr, ytr, Hte, yte), 3)
            mat[f"trained after {n_tr}"] = row
        cross[vn] = mat
    # feature overlap across steps
    with torch.no_grad():
        acts = {"r": tcs["reset"].act(tcs["reset"].input_to_features(torch.cat([Hp, X], -1))),
                "z": tcs["update"].act(tcs["update"].input_to_features(torch.cat([Hp, X], -1)))}
        rr = torch.sigmoid(gru.input_to_reset(X) + gru.hidden_to_reset(Hp))
        acts["n"] = tc.act(tc.input_to_features(torch.cat([rr * Hp, X], -1)))
    overlap = {}
    for vn, y in vars_.items():
        tops = {}
        for ph, pn in ((0, "D1"), (1, "GO"), (2, "D2")):
            yv = y[valid]
            sel = set()
            for g in "rzn":
                A = acts[g][:, ph::3].numpy()[valid]
                c = np.array([abs(np.corrcoef(A[:, f], yv)[0, 1]) if A[:, f].std() > 1e-6 else 0 for f in range(A.shape[1])])
                sel |= {f"{g}:{f}" for f in np.argsort(-c)[:10]}
            tops[pn] = sel
        overlap[vn] = {f"{a} vs {b_}": round(len(tops[a] & tops[b_]) / len(tops[a] | tops[b_]), 3)
                       for a, b_ in (("D1", "GO"), ("GO", "D2"), ("D1", "D2"))}
    out["cross_time"] = {"read_out_correlation": cross, "top_feature_overlap_jaccard": overlap}
    print("cross_time", json.dumps(out["cross_time"]), flush=True)
    # ---- run length
    Q, wdec, b0 = keep_out_basis(tracer, Hp, rec, with_decoder=True)
    dec = Hp[:, 2::3].numpy() @ wdec + b0
    streak = np.zeros(ch.shape)
    for t in range(2, ch.shape[1]):
        streak[:, t] = np.where(ch[:, t - 1] == ch[:, t - 2], streak[:, t - 1] + 1, 1)
    prevc = np.concatenate([np.zeros((N, 1)), np.where(ch[:, :-1] == 1, 1.0, -1.0)], 1)
    m = valid.copy(); m[:, :6] = False
    xb, yb = dec[m], L[m]
    B = basis(xb, np.percentile(xb, [10, 30, 50, 70, 90]))
    res = np.zeros(L.shape); res[m] = yb - B @ np.linalg.lstsq(B, yb, rcond=None)[0]
    a_d1 = {f: acts["n"][:, 3::3, f].numpy() for f in (220, 208, 11)}
    a_d1["r:4"] = acts["r"][:, 3::3, 4].numpy()
    bins = ((1, 3), (3, 6), (6, 11), (11, 21), (21, 101))
    rl = {}
    for name, rwd in (("after a miss", 0), ("after a reward", 1)):
        rows = {}
        for lo, hi in bins:
            sel = m[:, 1:] & (prev_r[:, 1:] == rwd) & (streak[:, 1:] >= lo) & (streak[:, 1:] < hi) & (ch[:, :-1] > 0)
            if sel.sum() < 50:
                continue
            pull = (res[:, 1:] * prevc[:, 1:])[sel]
            row = {"n": int(sel.sum()), "pull_towards_previous_choice": round(float(pull.mean()), 3)}
            for k_, v_ in a_d1.items():
                row[f"{k_ if isinstance(k_, str) else 'n:' + str(k_)} at this D1"] = round(float(v_[:, :sel.shape[1]][sel].mean()), 3)
            rows[f"kept the choice {lo}-{hi - 1} trials"] = row
        rl[name] = rows
    out["run_length"] = rl
    print("run_length", json.dumps(rl), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
