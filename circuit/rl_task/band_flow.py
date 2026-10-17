"""The bands of the phase portrait, traced with contrastive attribution graphs (bayes_vs_mb.py found them
statistically: at the same logit L(t) and input type, the logit change dL = L(t+1) - L(t) depends on the trial before).

Closed-loop episodes of the agent (reversal_flow.rollout). For a trial t (t-1, t, t+1 valid), the window runs from
the state carried into trial t-3 to the GO of trial t+1, inputs as recorded (later choices kept). Contrasts move a
point between bands while trial t's own input stays the same:

  prev_reward   trial t-1's reward (shown at t.D1) swapped 1 <-> 0
  prev_state    trial t-1's state (shown at (t-1).D2) swapped S1 <-> S2
  prev2_reward  trial t-2's reward swapped (the weaker, two-back bands)

Each pair is oriented so that the original's history favours a1 more than the edited one's (by the ideal observer),
so the logit change at t is positive. The observer's belief logit before t and before t+1 under both histories gives
the Bayesian reference: how much of the change a Bayesian keeps one trial later. Exact contrastive flow
(contrast_flow.full_flow, all three gates' features as nodes) for three targets, all linear in h so they stay exact:
  L(t)       the choice logit at t.GO
  L(t+1)     the logit at (t+1).GO
  offset     the band offset: each run's distance from its input type's mean curve dL(L) (piecewise linear,
             fitted on all trials), original minus edited. It equals d(dL) - s d(L(t)) exactly when s is the
             secant slope of the curve between the two runs' L(t), (f(L_o) - f(L_s)) / (L_o - L_s), a constant per
             pair, so it stays linear in h: its flow = flow(L(t+1)) - (1 + s) flow(L(t)). (A first version used the
             local slope at L(t); over logit changes of about 5 that counted the curve's bending as band.)
Per feature (kind, step relative to t, id): its flow into L(t), into L(t+1) and into the offset; for features at or
before t.GO the part of their flow into L(t) that does not reach L(t+1) ("lost": the transient). Patching (real GRU,
gates = reconstruction + held error, circuit_checks.run_patched): the top-k offset features per step set to their
values in the other run, sufficiency (in the edited run) and necessity (in the original), against random changed
features: the share of the offset carried.

    python -m circuit.rl_task.band_flow --out rl_features/rl_circuits/band_flow.json [--events 400]
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.bayes_vs_mb import TYPES, basis, mask_for, trial_table
from circuit.rl_task.circuit_checks import run_patched
from circuit.rl_task.contrast_flow import REW, S1, S2, full_flow, gru_run, load_model
from circuit.rl_task.reversal_flow import HAZARD, rollout

BACK = 3                                                                  # context trials before t-? (window starts at t-3)
PHASES = ("D1", "GO", "D2")


def curve_slopes(tt, m):
    """Slope dL/dL(t) of each input type's mean curve (piecewise linear in L) at every trial."""
    x, y, T = tt["L"][m], tt["dL"][m], tt["typ"][m]
    knots = np.percentile(x, [10, 30, 50, 70, 90])
    slopes = np.zeros(tt["L"].shape)
    for t in range(4):
        k = T == t
        beta = np.linalg.lstsq(basis(x[k], knots), y[k], rcond=None)[0]
        Lall = tt["L"]
        s = beta[1] + sum(beta[2 + j] * (Lall > kn) for j, kn in enumerate(knots))
        slopes = np.where(tt["typ"] == t, s, slopes)
    return slopes


def curve_fits(tt, m):
    """Each input type's mean curve dL(L) (piecewise linear in L), as a function f(type, L)."""
    x, y, T = tt["L"][m], tt["dL"][m], tt["typ"][m]
    knots = np.percentile(x, [10, 30, 50, 70, 90])
    betas = {t: np.linalg.lstsq(basis(x[T == t], knots), y[T == t], rcond=None)[0] for t in range(4)}
    return lambda typ, L: float((basis(np.array([L]), knots) @ betas[int(typ)])[0])


def make_pair(X, rec, e, t, contrast):
    lo = t - BACK
    x = X[e, 3 * lo:3 * (t + 1) + 2].clone()                              # up to (t+1).GO
    xc = x.clone()
    k = t - lo                                                            # trial t's index in the window
    if contrast == "prev_reward":
        xc[3 * k, REW] = 1 - xc[3 * k, REW]                               # t-1's reward, shown at t.D1
    elif contrast == "prev2_reward":
        xc[3 * (k - 1), REW] = 1 - xc[3 * (k - 1), REW]                   # t-2's reward, shown at (t-1).D1
    elif contrast == "prev_state":
        xc[3 * (k - 1) + 2, [S1, S2]] = xc[3 * (k - 1) + 2, [S2, S1]]     # t-1's state, shown at (t-1).D2
    return x, xc, k


def observer_logits(rec, e, t, x, k):
    """The ideal observer's belief logit before trial t's choice and before trial t+1's, with the window's inputs x
    (trials before the window as recorded; in the window the states and rewards are read from x)."""
    st, rw, ch = rec["state"][e].copy(), rec["reward"][e].copy(), rec["choice"][e]
    lo = t - BACK
    for j in range(k + 1):                                                # trials lo .. t: state at D2, reward at next D1
        trial = lo + j
        if 3 * j + 2 < x.shape[0]:
            st[trial] = 1 if x[3 * j + 2, S1] == 1 else 2 if x[3 * j + 2, S2] == 1 else 0
        if 3 * (j + 1) < x.shape[0]:
            rw[trial] = float(x[3 * (j + 1), REW])
    lam, out = 0.0, {}
    for trial in range(t + 1):
        if trial == t:
            out[0] = lam
        s_, r_ = st[trial], rw[trial]
        if ch[trial] > 0 and s_ > 0 and r_ in (0.0, 1.0):
            ev = np.log(9.0) * (1 if s_ == 1 else -1) * (1 if r_ == 1 else -1)
            b = 1 / (1 + np.exp(-(lam + ev)))
            b = b * (1 - HAZARD) + (1 - b) * HAZARD
            lam = float(np.log(b / (1 - b)))
    out[1] = lam
    return out


def step_name(ts, k):
    off = ts // 3 - k
    return f"t{'' if off == 0 else f'{off:+d}'}.{PHASES[ts % 3]}"


DETECTORS = [77, 66, 196, 118, 36, 18, 114, 183, 208, 165, 215, 8, 95, 145]
MATCHED = ((0, 6, 10), (1, 6, 10), (2, -10, -6), (3, 6, 10), (0, 8, 9), (0, 9, 10), (0, 10, 11))


@torch.no_grad()
def matched_logit(rnn, tcs, X, Hp, rec, tt):
    """At the same logit L(t) and the same input type, by the previous trial's input type: the next logit change,
    the belief read from h after t.GO (linear decoder of the ideal observer's belief logit), the ideal observer's
    belief, and the candidate detectors' activations at (t+1).D1, which reads trial t's outcome."""
    from circuit.rl_task.bayes_vs_mb import TYPES
    from circuit.rl_task.reversal_flow import observer
    m = mask_for(tt)
    b = observer(rec)
    lam = np.log(np.clip(b, 1e-6, 1) / np.clip(1 - b, 1e-6, 1))
    Hgo = Hp[:, 2::3].numpy()
    A = np.column_stack([Hgo.reshape(-1, 48), np.ones(Hgo.shape[0] * Hgo.shape[1])])
    w_ = np.linalg.lstsq(A, lam.reshape(-1), rcond=None)[0]
    dec = (Hgo.reshape(-1, 48) @ w_[:48] + w_[48]).reshape(lam.shape)
    gru, tc = rnn.layers[0], tcs["candidate"]
    idx = np.arange(3, X.shape[1], 3)
    x, h = X[:, idx], Hp[:, idx]
    r = torch.sigmoid(gru.input_to_reset(x) + gru.hidden_to_reset(h))
    act = tc.act(tc.input_to_features(torch.cat([r * h, x], -1))).numpy()   # [:, j]: D1 of trial j + 1
    out = {}
    T = act.shape[1]
    for cur, lo, hi in MATCHED:
        base = m[:, :T] & (tt["typ"][:, :T] == cur) & (tt["L"][:, :T] > lo) & (tt["L"][:, :T] < hi)
        rows = {}
        for pt in range(4):
            k = base & (tt["prev_typ"][:, :T] == pt)
            if k.sum() < 50:
                continue
            rows[f"previous {TYPES[pt]}"] = {
                "n": int(k.sum()), "logit": round(float(tt["L"][:, :T][k].mean()), 2),
                "next_change": round(float(tt["dL"][:, :T][k].mean()), 2),
                "belief_read_from_h": round(float(dec[:, :T][k].mean()), 2),
                "ideal_observer_belief": round(float(lam[:, :T][k].mean()), 2),
                "detectors_next_D1": {str(f): round(float(act[k][:, f].mean()), 2) for f in DETECTORS
                                      if act[base][:, f].mean() > 0.03}}
        out[f"{TYPES[cur]}, logit {lo} to {hi}"] = rows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--events", type=int, default=400)
    ap.add_argument("--contrasts", nargs="+", default=["prev_reward", "prev_state", "prev2_reward"])
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    rnn, tcs, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    tt = trial_table(rec)
    m = mask_for(tt)
    m[:, :BACK + 1] = False
    m[:, 1:] &= tt["valid"][:, :-1]
    m[:, 2:] &= tt["valid"][:, :-2]
    curve = curve_fits(tt, mask_for(tt))
    slopes = curve_slopes(tt, mask_for(tt))
    cand = np.argwhere(m[:, :-1])
    result = {"episodes": args.episodes, "window": "trials t-3 .. t+1 (to its GO)", "contrasts": {},
              "matched_logit": matched_logit(rnn, tcs, X, Hp, rec, tt)}
    w = tracer.W_choice
    for contrast in args.contrasts:
        picks = cand[rng.permutation(len(cand))[:args.events]]
        agg = {k: defaultdict(float) for k in ("L_t", "L_t1", "offset", "lost")}
        stat = defaultdict(list)
        by_type = defaultdict(lambda: defaultdict(list))
        items = []
        for e, t in picks:
            x, xc, k = make_pair(X, rec, e, t, contrast)
            acts = [tracer.run_forward_pass(gru_run(rnn, xx, Hp[e, 3 * (t - BACK)])) for xx in (x, xc)]
            tau1, tau2 = 3 * k + 1, 3 * (k + 1) + 1
            # orient: +1 if the original's history favours a1 more than the edited one's (by the observer)
            lam = [observer_logits(rec, e, t, xx, k) for xx in (x, xc)]
            sign = 1.0 if lam[0][0] >= lam[1][0] else -1.0
            stat["observer_d_lam_t"].append(sign * (lam[0][0] - lam[1][0]))
            stat["observer_d_lam_t1"].append(sign * (lam[0][1] - lam[1][1]))
            f1 = full_flow(tracer, acts[0], acts[1], sign * w, tau1)
            f2 = full_flow(tracer, acts[0], acts[1], sign * w, tau2)
            Lo_t, Ls_t = float(acts[0]["choice"][tau1]), float(acts[1]["choice"][tau1])
            ty = int(tt["typ"][e, t])
            s = ((curve(ty, Lo_t) - curve(ty, Ls_t)) / (Lo_t - Ls_t) if abs(Lo_t - Ls_t) > 1e-3
                 else float(slopes[e, t]))
            dL_t, dL_t1 = f1["d_target"], f2["d_target"]
            off = dL_t1 - (1 + s) * dL_t
            stat["d_L_t"].append(dL_t); stat["d_L_t1"].append(dL_t1); stat["d_dL"].append(dL_t1 - dL_t)
            stat["offset"].append(off)
            stat["check"].append(max(abs(f1["check"]), abs(f2["check"]), f1["preact_check"], f2["preact_check"]))
            by_type[TYPES[int(tt["typ"][e, t])]]["offset"].append(off)
            n1 = {nd: v for nd, v in zip(f1["nodes"], f1["flow"].tolist())}
            n2 = {nd: v for nd, v in zip(f2["nodes"], f2["flow"].tolist())}
            flows = {}
            for nd in set(n1) | set(n2):
                a, b = n1.get(nd, 0.0), n2.get(nd, 0.0)
                name = f"{step_name(nd[1], k)}/{nd[0]}:{nd[2]}"
                agg["L_t"][name] += a; agg["L_t1"][name] += b
                o = b - (1 + s) * a
                agg["offset"][name] += o
                flows[nd] = o
                if nd[1] <= tau1:
                    agg["lost"][name] += a - b
            for key, v in f1["into"].items():
                stat[f"src_L_t/{key}"].append(float(v.sum()))
            for key, v in f2["into"].items():
                stat[f"src_L_t1/{key}"].append(float(v.sum()))
            items.append({"x": (x, xc), "h0": Hp[e, 3 * (t - BACK)], "acts": acts, "tau": (tau1, tau2), "s": s, "sign": sign,
                          "offset": off, "flows": flows, "da": {nd: d for nd, d in zip(f2["nodes"], f2["da"].tolist())}})
        n = len(items)
        mean = lambda d_: {k_: round(v / n, 4) for k_, v in sorted(d_.items(), key=lambda kv: -abs(kv[1]))[:args.top]}
        by_step = defaultdict(float)
        for k_, v in agg["offset"].items():
            by_step[k_.split("/")[0]] += v / n
        lost_step = defaultdict(float)
        for k_, v in agg["lost"].items():
            lost_step[k_.split("/")[0]] += v / n
        out = {"n": n, "max_check": float(np.max(stat["check"])),
               "mean": {k_: round(float(np.mean(stat[k_])), 3) for k_ in ("d_L_t", "d_L_t1", "d_dL", "offset")},
               "kept_share_of_the_logit_change": {
                   "agent": round(float(np.median(np.array(stat["d_L_t1"]) / np.array(stat["d_L_t"]))), 3),
                   "ideal_observer_belief_logit": round(float(np.median(np.array(stat["observer_d_lam_t1"])
                                                                        / np.maximum(np.array(stat["observer_d_lam_t"]), 1e-6))), 3)},
               "mean_abs_offset": round(float(np.mean(np.abs(stat["offset"]))), 3),
               "offset_by_input_type_at_t": {k_: round(float(np.mean(v["offset"])), 3) for k_, v in sorted(by_type.items())},
               "flow_by_source": {k_: round(float(np.mean(v)), 3) for k_, v in stat.items() if k_.startswith("src")
                                  and abs(np.mean(v)) >= 1e-3},
               "offset_flow_by_step": {k_: round(v, 3) for k_, v in sorted(by_step.items())},
               "lost_flow_by_step": {k_: round(v, 3) for k_, v in sorted(lost_step.items())},
               "top_offset_features": mean(agg["offset"]), "top_lost_features": mean(agg["lost"]),
               "top_L_t_features": mean(agg["L_t"]), "top_L_t1_features": mean(agg["L_t1"])}
        # patching the top-k offset features per step
        pat = defaultdict(list)
        for kk in (1, 3, 5, 10):
            for rnd in (False, True):
                for it in items:
                    if abs(it["offset"]) < 0.5:
                        continue
                    by = defaultdict(list)
                    for nd, o in it["flows"].items():
                        if it["da"].get(nd, 0.0) != 0:
                            by[nd[1]].append((abs(o), nd))
                    chosen = []
                    for lst in by.values():
                        lst.sort(key=lambda v: -v[0])
                        if rnd:
                            rest = [nd for _, nd in lst[kk:]]
                            chosen += [rest[j] for j in rng.choice(len(rest), size=min(kk, len(rest)), replace=False)] if rest else []
                        else:
                            chosen += [nd for _, nd in lst[:kk]]
                    T = it["tau"][1] + 1
                    mask = {g: torch.zeros(1, T, it["acts"][0][f"f_{g}"].shape[1], dtype=torch.bool) for g in "rzn"}
                    for g, ts, fi in chosen:
                        mask[g][0, ts, fi] = True
                    first = min(nd[1] for nd in it["da"] if it["da"][nd] != 0)
                    steps = torch.zeros(1, T, dtype=torch.bool); steps[0, first:] = True
                    tau1, tau2 = it["tau"]
                    for mode, run, src in (("sufficiency", 1, 0), ("necessity", 0, 1)):
                        value = {g: it["acts"][src][f"f_{g}"][None] for g in "rzn"}
                        err = {g: it["acts"][run][f"e_{g}"][None] for g in "rzn"}
                        y = run_patched(rnn, tcs, it["x"][run][None], it["h0"][None], mask, value, steps, err)[0]
                        L = [float(a["choice"][tt_]) for a in it["acts"] for tt_ in (tau1, tau2)]   # o: L_t, L_t1; s: ...
                        L = [it["sign"] * v for v in L]
                        Lp_t, Lp_t1 = it["sign"] * float(y[tau1]), it["sign"] * float(y[tau2])
                        s = it["s"]
                        if mode == "sufficiency":                         # edited run patched towards o
                            got = (Lp_t1 - L[3]) - (1 + s) * (Lp_t - L[2])
                        else:                                             # original patched towards s
                            got = (L[1] - Lp_t1) - (1 + s) * (L[0] - Lp_t)
                        pat[f"{'random' if rnd else 'top'}{kk}/{mode}"].append(got / it["offset"])
        out["patching_offset_share_median"] = {k_: round(float(np.median(v)), 3) for k_, v in sorted(pat.items())}
        result["contrasts"][contrast] = out
        print(contrast, json.dumps(out), f"({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
