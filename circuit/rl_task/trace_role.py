"""What is the "was I just rewarded" trace for? (rare_and_trace.py: it carries most of the phase-portrait bands.)

The trace: the top 3 directions of the change in h after a trial's GO when the previous trial's reward is swapped
(oriented by the reward), with the belief direction (a linear decoder of the ideal observer's belief) and the choice
read-out projected out, estimated on closed-loop events as in rare_and_trace.py. Three tests:

ablate   closed-loop play (reversal_flow.rollout) with the trace removed on every trial: after each GO, h's
         component along the trace directions is set to its mean there. Compared with the intact agent and with
         random 3-dimensional subspaces (also clear of the belief direction and the read-out), same seeds: reward
         per trial, agreement with the ideal observer, P(new good action) and the choice logit after reversals,
         the stay table, how much of the phase-portrait spread the previous trial explains (the bands), and how
         well the value predicts the discounted return.
         Cleaner version ("conditional"): the trace is set not to its mean but to what the belief predicts it to be
         (a flexible fit on two belief readings of h after GO, the choice logit and the decoded belief, which the
         edit leaves unchanged), so only the part of the trace the belief cannot explain is removed.
belief_update  the agent's belief update per trial (the decoded belief after GO, trial t to t+1) against the ideal
         observer's, by whether the outcome agreed with the current belief and by confidence; and after reversals,
         how often the ideal belief, the belief read from h and the choice favour the new good action.
inertia  closed-loop play where, before each GO, h's component along the choice read-out direction (made orthogonal
         to the belief decoder, so the belief is kept) is set to its mean: does the choice follow the belief faster
         after reversals? Controls: random directions (clear of the belief direction and the read-out).
value    does the trace feed the value head? The band contrast (previous trial's reward swapped): the value
         difference between the two runs at the next steps, and the share of it removed when the trace part of
         the difference is removed after the current GO (vs random subspaces).
choice   how often do the bands change the choice itself? Per trial of the intact agent, the logit at the next GO
         against the logit its input type's average curve predicts: how often the two disagree in sign, and the
         change in P(a1) the band makes.

    python -m circuit.rl_task.trace_role --out rl_features/rl_circuits/trace_role.json
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.band_flow import BACK, curve_fits, make_pair, observer_logits
from circuit.rl_task.bayes_vs_mb import bands, mask_for, trial_table
from circuit.rl_task.contrast_flow import REW, gru_run, load_model
from circuit.rl_task.reversal_flow import behaviour, observer, rollout, settled_reversals
from circuit.rl_task.value_circuit import returns


def keep_out_basis(tracer, Hp, rec, with_decoder=False):
    """Orthonormal basis (48, 2) of the belief decoder direction (h after GO) and the choice read-out (and the
    decoder's weights and intercept)."""
    b = observer(rec)
    lam = np.log(np.clip(b, 1e-6, 1) / np.clip(1 - b, 1e-6, 1))
    Hgo = Hp[:, 2::3].numpy().reshape(-1, 48)
    wfull = np.linalg.lstsq(np.column_stack([Hgo, np.ones(len(Hgo))]), lam.reshape(-1), rcond=None)[0]
    wdec = wfull[:48]
    keep = np.stack([wdec / np.linalg.norm(wdec), tracer.W_choice.detach().numpy() / tracer.W_choice.norm().item()])
    Q, _ = np.linalg.qr(keep.T)
    return (Q, wdec, wfull[48]) if with_decoder else Q


def belief_basis(L, dec, kL, kd):
    """Flexible functions of the two belief readings of h (choice logit L, decoded belief dec): numpy or torch."""
    mod = torch if torch.is_tensor(L) else np
    relu = (lambda v: torch.clamp(v, min=0)) if torch.is_tensor(L) else (lambda v: np.maximum(v, 0))
    L, dec = L / 10.0, dec / 4.0                                          # about unit scale
    kL, kd = [k / 10.0 for k in kL], [k / 4.0 for k in kd]
    cols = [mod.ones_like(L), L, dec, L * dec] + [relu(L - k) for k in kL] + [relu(dec - k) for k in kd]
    return mod.stack(cols, -1)


def belief_update(rec, Hp, wdec, b0):
    """The agent's belief update per trial (belief read from h after GO, trial t to t+1) against the ideal
    observer's, by whether trial t's outcome agreed with the current belief and how confident the belief was."""
    b = observer(rec)
    lam = np.log(np.clip(b, 1e-6, 1) / np.clip(1 - b, 1e-6, 1))
    dec = Hp[:, 2::3].numpy() @ wdec + b0                                 # belief after each trial's GO
    tt = trial_table(rec)
    m = mask_for(tt)
    st, rw = rec["state"], rec["reward"]
    ev_s1 = ((st == 1) & (rw == 1)) | ((st == 2) & (rw == 0))            # trial t's outcome is evidence for S1
    agree = ev_s1 == (lam > 0)
    conf = np.abs(lam)
    d_agent = np.concatenate([dec[:, 1:] - dec[:, :-1], np.zeros((dec.shape[0], 1))], 1) * np.sign(lam)
    d_ideal = np.concatenate([lam[:, 1:] - lam[:, :-1], np.zeros((lam.shape[0], 1))], 1) * np.sign(lam)
    out = {}
    for name, sel in (("agreeing", agree), ("disagreeing", ~agree)):
        rows = {}
        for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4)):
            k = m & sel & (conf >= lo) & (conf < hi)
            if k.sum() >= 50:
                rows[f"confidence {lo}-{hi}"] = {"n": int(k.sum()), "agent": round(float(d_agent[k].mean()), 3),
                                                 "ideal": round(float(d_ideal[k].mean()), 3)}
        out[name] = rows
    k = m & ~agree & (conf >= 3)
    out["disagreeing_at_high_confidence_agent_over_ideal"] = round(float(d_agent[k].mean() / d_ideal[k].mean()), 3)
    out["note"] = "belief change towards the currently believed state (negative = towards the other), in belief logits"
    return out


def band_events(rnn, X, Hp, rec, rng, n):
    """Band-contrast events (previous trial's reward swapped): states after the current GO in both runs."""
    tt = trial_table(rec)
    m = mask_for(tt); m[:, :BACK + 1] = False
    m[:, 1:] &= tt["valid"][:, :-1]; m[:, 2:] &= tt["valid"][:, :-2]
    cand = np.argwhere(m[:, :-1])
    ev = []
    for e, t in cand[rng.permutation(len(cand))[:n]]:
        x, xc, k = make_pair(X, rec, e, t, "prev_reward")
        h0 = Hp[e, 3 * (t - BACK)]
        a = [gru_run(rnn, xx, h0) for xx in (x, xc)]
        tau1 = 3 * k + 1
        h_after = [(1 - it["z_ts"][tau1]) * it["h_prevs"][tau1] + it["z_ts"][tau1] * it["h_new_ts"][tau1] for it in a]
        rsign = 1.0 if x[3 * k, REW] == 1 else -1.0
        ev.append({"x": x, "xc": xc, "k": k, "h": h_after, "h0": h0, "dh": (rsign * (h_after[0] - h_after[1])).numpy()})
    return ev


@torch.no_grad()
def run_from(rnn, x, h):
    gru, readout = rnn.layers[0], rnn.layers[-1]
    L, V = [], []
    for t in range(x.shape[0]):
        xt = x[t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        h = (1 - z) * h + z * n
        o = readout(h)
        L.append(float(o[1] - o[2])); V.append(float(o[3]))
    return L, V


def belief_vs_choice(rec, Hp, wdec, b0):
    """After settled reversals, by trials since: the share of trials whose ideal belief, belief read from h (after GO,
    where the decoder was fit) and choice logit favour the new good action, and their means (towards the new one)."""
    b = observer(rec)
    lam = np.log(np.clip(b, 1e-6, 1) / np.clip(1 - b, 1e-6, 1))
    dec = Hp[:, 2::3].numpy() @ wdec + b0
    L = rec["logit"]
    rows = defaultdict(list)
    for e, R, old_a, new_a in settled_reversals(rec):
        s_ = 1 if new_a == 1 else -1
        for k in range(-1, 7):
            rows[k].append((s_ * lam[e, R + k], s_ * dec[e, R + k], s_ * L[e, R + k]))
    out = {}
    for k, r in sorted(rows.items()):
        r = np.array(r)
        out[str(k)] = {"share_favouring_new": {"ideal belief": round(float((r[:, 0] > 0).mean()), 3),
                                               "belief read from h": round(float((r[:, 1] > 0).mean()), 3),
                                               "choice": round(float((r[:, 2] > 0).mean()), 3)},
                       "mean": {"ideal belief": round(float(r[:, 0].mean()), 2),
                                "belief read from h": round(float(r[:, 1].mean()), 2),
                                "choice logit": round(float(r[:, 2].mean()), 2)}}
    return out


def summary(rec, X, Hp):
    ch, rw = rec["choice"], rec["reward"]
    valid = ch > 0
    b = observer(rec)
    ideal = np.where(b > 0.5, 1, 2)
    beh = behaviour(rec, settled_reversals(rec))
    tt = trial_table(rec)
    bd = bands(tt, mask_for(tt))
    V, R = rec["value"], rec["step_reward"]
    G = returns(R)
    return {"reward_per_trial": round(float(np.clip(rw[valid], 0, None).mean()), 4),
            "invalid_trial_share": round(float(1 - valid.mean()), 4),
            "agreement_with_ideal_observer": round(float((ch[valid] == ideal[valid]).mean()), 4),
            "p_new_action_after_reversal": {k: v["agent"] for k, v in beh["p_new_action_by_trials_since_reversal"].items()
                                            if 1 <= int(k) <= 6},
            "choice_logit_towards_new_action": {k: v["mean"] for k, v in beh["choice_logit_towards_new_action"].items()
                                                if -1 <= int(k) <= 6},
            "stay_table": beh["stay_table"],
            "mean_abs_choice_logit": round(float(np.abs(rec["logit"][valid]).mean()), 3),
            "bands_share_explained_by_previous_input_type": bd["added"]["previous input type"]["share_of_spread_explained"],
            "value_rmse_vs_return_by_step": round(float(np.sqrt(((V.mean(0) - G.mean(0)) ** 2).mean())), 3),
            "value_return_correlation": round(float(np.corrcoef(V.ravel(), G.ravel())[0, 1]), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--events", type=int, default=1200)
    ap.add_argument("--dims", type=int, default=3)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    rnn, tcs, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    Q, wdec, b0 = keep_out_basis(tracer, Hp, rec, with_decoder=True)
    proj_out = np.eye(48) - Q @ Q.T
    ev = band_events(rnn, X, Hp, rec, rng, args.events)
    half = len(ev) // 2
    D = np.stack([proj_out @ v["dh"] for v in ev[:half]])
    Vt = np.linalg.svd(D, full_matrices=False)[2]
    subspaces = {"trace": Vt[:args.dims].T}
    for j in range(3):
        B, _ = np.linalg.qr(proj_out @ np.random.default_rng(200 + j).normal(size=(48, args.dims)))
        subspaces[f"random_{j}"] = B
    mu = Hp[:, 2::3].reshape(-1, 48).mean(0)                              # mean state after GO
    result = {"episodes": args.episodes, "dims": args.dims, "intact": summary(rec, X, Hp), "ablated": {}}
    print("intact", json.dumps(result["intact"]), f"({time.time() - t0:.0f}s)", flush=True)
    for name, B in subspaces.items():
        P = torch.tensor(B @ B.T, dtype=torch.float32)
        hook = lambda ph, h, P=P: h - (h - mu) @ P if ph == 1 else h       # after each GO: trace part set to its mean
        Xa, Ha, ra = rollout(rnn, args.episodes, args.seed, hook=hook)
        result["ablated"][name] = summary(ra, Xa, Ha)
        print(name, json.dumps(result["ablated"][name]), f"({time.time() - t0:.0f}s)", flush=True)
    # cleaner test: set the trace to what the belief predicts it to be (only the part the belief cannot explain is
    # removed). Belief readings of h after GO: the choice logit and the decoded belief (both untouched by the edit).
    Hg = Hp[:, 2::3].reshape(-1, 48)
    Lg = Hg @ tracer.W_choice.detach() + float(tracer.b_choice)
    dg = Hg @ torch.tensor(wdec, dtype=torch.float32) + float(b0)
    kL = np.percentile(Lg.numpy(), [20, 40, 60, 80]).tolist(); kd = np.percentile(dg.numpy(), [20, 40, 60, 80]).tolist()
    Phi = belief_basis(Lg.double(), dg.double(), kL, kd)
    result["conditional"] = {}
    for name, B in subspaces.items():
        Bt = torch.tensor(B, dtype=torch.float32)
        C = ((Hg - mu) @ Bt).double()
        beta = torch.linalg.solve(Phi.T @ Phi + 1e-3 * len(Phi) * torch.eye(Phi.shape[1], dtype=Phi.dtype), Phi.T @ C)
        r2 = 1 - ((C - Phi @ beta) ** 2).sum(0) / ((C - C.mean(0)) ** 2).sum(0)
        wd, bt = torch.tensor(wdec, dtype=torch.float32), beta.float()
        def hook(ph, h, Bt=Bt, bt=bt, wd=wd):
            if ph != 1:
                return h
            L_ = h @ tracer.W_choice.detach() + float(tracer.b_choice)
            d_ = h @ wd + float(b0)
            c_hat = belief_basis(L_, d_, kL, kd) @ bt
            return h - ((h - mu) @ Bt - c_hat) @ Bt.T
        Xa, Ha, ra = rollout(rnn, args.episodes, args.seed, hook=hook)
        result["conditional"][name] = summary(ra, Xa, Ha)
        result["conditional"][name]["share_of_variance_predicted_by_belief"] = [round(float(v), 3) for v in r2]
        print("conditional", name, json.dumps(result["conditional"][name]), f"({time.time() - t0:.0f}s)", flush=True)
    result["belief_update"] = belief_update(rec, Hp, wdec, b0)
    result["belief_vs_choice_after_reversal"] = belief_vs_choice(rec, Hp, wdec, b0)
    print("belief_vs_choice", json.dumps(result["belief_vs_choice_after_reversal"]), flush=True)
    # choice inertia: before each GO (after D1), set the component of h along the choice read-out direction that is
    # orthogonal to the belief decoder to its mean (the previous choice's carried trace), keep the belief
    w = tracer.W_choice.detach().numpy()
    wd = wdec / np.linalg.norm(wdec)
    u = w - (w @ wd) * wd
    u = torch.tensor(u / np.linalg.norm(u), dtype=torch.float32)
    mu1 = Hp[:, 1::3].reshape(-1, 48).mean(0)                             # mean state after D1 (entering GO)
    result["inertia"] = {}
    for name, vec in [("choice read-out, belief kept", u)] + [
            (f"random direction {j}", None) for j in range(3)]:
        if vec is None:
            r_ = proj_out @ np.random.default_rng(300 + int(name[-1])).normal(size=48)
            vec = torch.tensor(r_ / np.linalg.norm(r_), dtype=torch.float32)
        hook = lambda ph, h, v=vec: h - ((h - mu1) @ v)[..., None] * v if ph == 0 else h
        Xa, Ha, ra = rollout(rnn, args.episodes, args.seed, hook=hook)
        result["inertia"][name] = summary(ra, Xa, Ha)
        result["inertia"][name]["belief_vs_choice_after_reversal"] = belief_vs_choice(ra, Ha, wdec, b0)
        print("inertia", name, json.dumps(result["inertia"][name]), f"({time.time() - t0:.0f}s)", flush=True)
    print("belief_update", json.dumps(result["belief_update"]), flush=True)
    # value: does the trace feed the value head?
    val = defaultdict(list)
    for name, B in subspaces.items():
        P = torch.tensor(B @ B.T, dtype=torch.float32)
        for v in ev[half:]:
            tau1 = 3 * v["k"] + 1
            Lo, Vo = run_from(rnn, v["x"][tau1 + 1:], v["h"][0])
            Ls, Vs = run_from(rnn, v["x"][tau1 + 1:], v["h"][1])
            Lp, Vp = run_from(rnn, v["x"][tau1 + 1:], v["h"][0] - P @ (v["h"][0] - v["h"][1]))
            for j, step in ((0, "D2"), (1, "next D1"), (2, "next GO")):
                d = Vo[j] - Vs[j]
                val[f"{name}/{step}/value_difference"].append(abs(d))
                if abs(d) > 0.02:
                    val[f"{name}/{step}/share_removed"].append(1 - (Vp[j] - Vs[j]) / d)
    result["value"] = {k: round(float(np.median(v)), 3) for k, v in sorted(val.items())}
    print("value", json.dumps(result["value"]), flush=True)
    # choice: how often do the bands change the choice itself?
    tt = trial_table(rec)
    m = mask_for(tt)
    curve = curve_fits(tt, m)
    L, dL, T = tt["L"][m], tt["dL"][m], tt["typ"][m]
    pred = np.array([l + curve(ty, l) for l, ty in zip(L, T)])
    actual = L + dL
    sig = lambda z: 1 / (1 + np.exp(-z))
    result["choice"] = {"n_trials": int(m.sum()),
                        "share_where_band_flips_the_preferred_action": round(float(np.mean(np.sign(pred) != np.sign(actual))), 4),
                        "mean_abs_change_in_P(a1)": round(float(np.mean(np.abs(sig(actual) - sig(pred)))), 4),
                        "share_with_P(a1)_change_over_0.1": round(float(np.mean(np.abs(sig(actual) - sig(pred)) > 0.1)), 4),
                        "mean_abs_band_offset_logits": round(float(np.mean(np.abs(actual - pred))), 3)}
    print("choice", json.dumps(result["choice"]), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
