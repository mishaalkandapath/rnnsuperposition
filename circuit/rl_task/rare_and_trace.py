"""Two follow-ups to belief_axis.py and band_flow.py (closed-loop episodes, reversal_flow.rollout).

rare   An outcome's effect on the choice lasts much longer when the next trial reaches the other state by a rare
       transition (belief_axis.py: 0.99 kept vs 0.44 by a common one). Why? Events: trial u's reward swapped
       (oriented so the original favours S1), trial v = u + 1 reached the other state; split by v's transition.
       (1) Is v's own outcome weighted less after a rare transition? Its effect on the choice at t.GO (t = u + 2)
           when v's reward is swapped instead.
       (2) The effect of u's swap right after it (v.GO) and two trials on (t.GO), separately (medians), the share
           of events where trial u was rewarded and where the agent stayed at v (chose the action leading commonly to
           u's state; a rare transition to the other state means it stayed), and the exact
           attribution graph to t.GO: flow by step and the features whose flow differs most between rare and
           common cases (the extra part that survives).
trace  Does the "rewarded or not" trace in h carry the bands? The band contrast of band_flow.py (trial t-1's reward
       swapped, trial t's input kept; oriented). The trace subspace: the top principal directions of the change in h
       after t.GO across events (on half of them), with the belief decoder direction and the choice readout projected
       out. On the other half: after t.GO, when L(t) is already fixed, the trace part of the difference between the
       two runs is removed from the original's h; the real GRU runs on to (t+1).GO. If the trace carries the band,
       the band offset should vanish. The offset here is exact: each run's distance from its input type's mean
       curve, original minus edited (band_flow.py's linear-slope version is compared with it). Controls: random subspaces of the same size, also orthogonal
       to the belief direction and the readout. Where the rest is: the same removal for everything except the belief
       direction and the readout (46 dimensions), for only those two, and for the trace plus those two. Preference-
       protected version: of the difference along each subspace, only the part that the agent's action preference
       (its choice logit after t.GO) and the belief read-out cannot explain is removed.

    python -m circuit.rl_task.rare_and_trace --out rl_features/rl_circuits/rare_and_trace.json
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.band_flow import BACK, curve_fits, curve_slopes, make_pair, observer_logits
from circuit.rl_task.bayes_vs_mb import mask_for, trial_table
from circuit.rl_task.belief_axis import favours_s1
from circuit.rl_task.contrast_flow import REW, full_flow, gru_run, load_model
from circuit.rl_task.reversal_flow import observer, rollout

PHASES = ("D1", "GO", "D2")


@torch.no_grad()
def run_from(rnn, x, h):
    """Real GRU from state h on inputs x (T, 8); the choice logit after each step."""
    gru, readout = rnn.layers[0], rnn.layers[-1]
    out = []
    for t in range(x.shape[0]):
        xt = x[t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        h = (1 - z) * h + z * n
        o = readout(h)
        out.append(float(o[1] - o[2]))
    return out


def rare_test(rnn, tracer, X, Hp, rec, rng, n_events, top):
    ch, st, rw = rec["choice"], rec["state"], rec["reward"]
    cand = [(e, u) for e in range(ch.shape[0]) for u in range(4, 97)
            if (ch[e, u - 2:u + 3] > 0).all() and (st[e, u - 2:u + 3] > 0).all() and rw[e, u] in (0, 1)
            and rw[e, u + 1] in (0, 1) and st[e, u + 1] != st[e, u]]
    picks = [cand[i] for i in rng.permutation(len(cand))[:n_events]]
    stat = defaultdict(lambda: defaultdict(list))
    flow = {g: defaultdict(float) for g in ("rare", "common")}
    step_flow = {g: defaultdict(float) for g in ("rare", "common")}
    n_g = defaultdict(int)
    w = tracer.W_choice
    for e, u in picks:
        v, t, lo = u + 1, u + 2, u - 2
        x = X[e, 3 * lo:3 * t + 2].clone()
        g = "common" if st[e, v] == ch[e, v] else "rare"
        conf = "agreeing" if favours_s1(st[e, v], rw[e, v]) == favours_s1(st[e, u], rw[e, u]) else "disagreeing"
        sign = 1.0 if favours_s1(st[e, u], rw[e, u]) else -1.0
        # (2) u's swap
        xc = x.clone(); xc[9, REW] = 1 - xc[9, REW]
        acts = [tracer.run_forward_pass(gru_run(rnn, xx, Hp[e, 3 * lo])) for xx in (x, xc)]
        a_v = sign * (float(acts[0]["choice"][10]) - float(acts[1]["choice"][10]))
        a_t = sign * (float(acts[0]["choice"][13]) - float(acts[1]["choice"][13]))
        stat[g]["effect_right_after (v.GO)"].append(a_v); stat[g]["effect_two_trials_on (t.GO)"].append(a_t)
        stat[g]["share_trial_u_rewarded"].append(float(rw[e, u] == 1))
        stat[g]["share_stayed (chose u's state's action at v)"].append(float(ch[e, v] == st[e, u]))
        stat[g]["kept"].append(a_t / a_v if abs(a_v) > 1e-3 else np.nan)
        stat[f"{g}, {conf}"]["kept"].append(a_t / a_v if abs(a_v) > 1e-3 else np.nan)
        f = full_flow(tracer, acts[0], acts[1], sign * w, 13)
        n_g[g] += 1
        for (kd, ts, fi), fl in zip(f["nodes"], f["flow"].tolist()):
            if fl != 0 and ts >= 9:
                name = f"{['v', 't'][ts // 3 - 3]}.{PHASES[ts % 3]}"
                flow[g][f"{name}/{kd}:{fi}"] += fl
                step_flow[g][name] += fl
        # (1) v's own outcome: its reward swapped (shown at t.D1), effect at t.GO, oriented by its own evidence
        xv = x.clone(); xv[12, REW] = 1 - xv[12, REW]
        sv = 1.0 if favours_s1(st[e, v], rw[e, v]) else -1.0
        b_o, b_s = run_from(rnn, x, Hp[e, 3 * lo]), run_from(rnn, xv, Hp[e, 3 * lo])
        stat[g]["effect_of_v's_own_outcome (t.GO)"].append(sv * (b_o[13] - b_s[13]))
    out = {"n": dict(n_g), "by_transition_of_v": {}}
    for g, d in stat.items():
        out["by_transition_of_v"][g] = {k: round(float(np.nanmean(v) if k.startswith("share") else np.nanmedian(v)), 3)
                                        for k, v in d.items()}
    diff = {k: flow["rare"].get(k, 0) / n_g["rare"] - flow["common"].get(k, 0) / n_g["common"]
            for k in set(flow["rare"]) | set(flow["common"])}
    out["flow_by_step"] = {g: {k: round(v / n_g[g], 3) for k, v in sorted(d.items())} for g, d in step_flow.items()}
    out["features_rare_minus_common"] = [{"node": k, "rare": round(flow["rare"].get(k, 0) / n_g["rare"], 3),
                                          "common": round(flow["common"].get(k, 0) / n_g["common"], 3), "difference": round(v, 3)}
                                         for k, v in sorted(diff.items(), key=lambda kv: -abs(kv[1]))[:top]]
    return out


def trace_test(rnn, tracer, X, Hp, rec, rng, n_events, dims=(1, 3, 5)):
    tt = trial_table(rec)
    m = mask_for(tt); m[:, :BACK + 1] = False
    m[:, 1:] &= tt["valid"][:, :-1]; m[:, 2:] &= tt["valid"][:, :-2]
    slopes = curve_slopes(tt, mask_for(tt))
    curve = curve_fits(tt, mask_for(tt))
    cand = np.argwhere(m[:, :-1])
    picks = cand[rng.permutation(len(cand))[:n_events]]
    # belief decoder direction (h after GO) and choice readout, to keep out of the trace subspace
    b = observer(rec)
    lam = np.log(np.clip(b, 1e-6, 1) / np.clip(1 - b, 1e-6, 1))
    Hgo = Hp[:, 2::3].numpy().reshape(-1, 48)                            # state entering D2 = after GO
    wdec = np.linalg.lstsq(np.column_stack([Hgo, np.ones(len(Hgo))]), lam.reshape(-1), rcond=None)[0][:48]
    keep_out = np.stack([wdec / np.linalg.norm(wdec), tracer.W_choice.detach().numpy() / tracer.W_choice.norm().item()])
    Q, _ = np.linalg.qr(keep_out.T)                                      # (48, 2) orthonormal
    proj_out = np.eye(48) - Q @ Q.T
    ev = []
    for e, t in picks:
        x, xc, k = make_pair(X, rec, e, t, "prev_reward")
        lams = [observer_logits(rec, e, t, xx, k) for xx in (x, xc)]
        sign = 1.0 if lams[0][0] >= lams[1][0] else -1.0
        h0 = Hp[e, 3 * (t - BACK)]
        a = [gru_run(rnn, xx, h0) for xx in (x, xc)]
        tau1 = 3 * k + 1
        h_after = [(1 - it["z_ts"][tau1]) * it["h_prevs"][tau1] + it["z_ts"][tau1] * it["h_new_ts"][tau1] for it in a]
        Lo, Ls = run_from(rnn, x, h0), run_from(rnn, xc, h0)
        rsign = 1.0 if x[3 * k, REW] == 1 else -1.0                      # orientation by the reward itself
        ev.append({"x": x, "k": k, "sign": sign, "s": float(slopes[e, t]), "h": h_after, "Lo": Lo, "Ls": Ls,
                   "typ": int(tt["typ"][e, t]),
                   "dh_by_reward": (rsign * (h_after[0] - h_after[1])).numpy()})
    half = len(ev) // 2
    D = np.stack([proj_out @ v["dh_by_reward"] for v in ev[:half]])
    _, _, Vt = np.linalg.svd(D - 0 * D.mean(0), full_matrices=False)    # uncentred: directions of the change itself
    var_share = (np.linalg.svd(D, compute_uv=False) ** 2)
    var_share = var_share / var_share.sum()
    out = {"n_events": len(ev), "trace_variance_share_top5": np.round(var_share[:5], 3).tolist(), "results": {}}
    def offset_of(v, Lt1_o):
        """Exact band offset: each run's distance from its input type's curve, original minus edited (oriented).
        (band_flow.py's flows use the linearised version, d(dL) - s d(L(t)), which is linear in h.)"""
        tau1, tau2 = 3 * v["k"] + 1, 3 * (v["k"] + 1) + 1
        res_o = (Lt1_o - v["Lo"][tau1]) - curve(v["typ"], v["Lo"][tau1])
        res_s = (v["Ls"][tau2] - v["Ls"][tau1]) - curve(v["typ"], v["Ls"][tau1])
        return v["sign"] * (res_o - res_s)

    def offset_linear(v):
        tau1, tau2 = 3 * v["k"] + 1, 3 * (v["k"] + 1) + 1
        dLt = v["sign"] * (v["Lo"][tau1] - v["Ls"][tau1])
        return v["sign"] * (v["Lo"][tau2] - v["Ls"][tau2]) - (1 + v["s"]) * dLt
    ex = np.array([offset_of(v, v["Lo"][3 * (v["k"] + 1) + 1]) for v in ev])
    li = np.array([offset_linear(v) for v in ev])
    out["exact_vs_linearised_offset"] = {"corr": round(float(np.corrcoef(ex, li)[0, 1]), 3),
                                         "mean_abs_exact": round(float(np.abs(ex).mean()), 3),
                                         "mean_abs_linearised": round(float(np.abs(li).mean()), 3),
                                         "mean_abs_difference": round(float(np.abs(ex - li).mean()), 3)}
    subspaces = [(k_, d_) for k_ in ["trace"] + [f"random_{j}" for j in range(5)] for d_ in dims]
    subspaces += [("all_but_belief_and_readout", 46), ("belief_and_readout", 2), ("trace_top5_and_belief_and_readout", 7)]
    for kind, kd in subspaces:
        if True:
            if kind == "trace":
                B = Vt[:kd].T
            elif kind == "all_but_belief_and_readout":
                B = np.linalg.svd(proj_out)[0][:, :46]                   # orthonormal basis of the complement
            elif kind == "belief_and_readout":
                B = Q
            elif kind == "trace_top5_and_belief_and_readout":
                B, _ = np.linalg.qr(np.column_stack([Vt[:5].T, Q]))
            else:
                R = proj_out @ np.random.default_rng(100 + int(kind[-1]) * 10 + kd).normal(size=(48, kd))
                B, _ = np.linalg.qr(R)
            P = torch.tensor(B @ B.T, dtype=torch.float32)
            shares, offs, offp = [], [], []
            for v in ev[half:]:
                tau1, tau2 = 3 * v["k"] + 1, 3 * (v["k"] + 1) + 1
                off = offset_of(v, v["Lo"][tau2])
                if abs(off) < 0.5:
                    continue
                hp = v["h"][0] - P @ (v["h"][0] - v["h"][1])
                Lp = run_from(rnn, v["x"][tau1 + 1:], hp)
                offp_ = offset_of(v, Lp[-1])
                shares.append(1 - offp_ / off); offs.append(off); offp.append(offp_)
            out["results"][f"{kind}/{kd}d"] = {"offset_removed_share_median": round(float(np.median(shares)), 3),
                                              "mean_abs_offset_before": round(float(np.mean(np.abs(offs))), 3),
                                              "mean_abs_offset_after": round(float(np.mean(np.abs(offp))), 3), "n": len(shares)}
    # preference-protected removal: of the difference along each subspace, remove only the part the agent's action
    # preference (the choice logit after t.GO) and belief read-out cannot explain (trace_role.belief_basis fit on all
    # after-GO states), so information about the previous reward that the preference reflects is kept
    from circuit.rl_task.trace_role import belief_basis
    Hg = Hp[:, 2::3].reshape(-1, 48)
    wd_t = torch.tensor(wdec, dtype=torch.float32)
    b0 = float(np.mean(lam) - Hgo.mean(0) @ wdec)
    readings = lambda H: (H @ tracer.W_choice.detach() + float(tracer.b_choice), H @ wd_t + b0)
    Lg, dg = readings(Hg)
    kL = np.percentile(Lg.numpy(), [20, 40, 60, 80]).tolist(); kd_ = np.percentile(dg.numpy(), [20, 40, 60, 80]).tolist()
    Phi = belief_basis(Lg.double(), dg.double(), kL, kd_)
    mu = Hg.mean(0)
    out["preference_protected"] = {}
    for kind, kd in [(k_, d_) for k_ in ["trace"] + [f"random_{j}" for j in range(5)] for d_ in dims]:
        if kind == "trace":
            B = Vt[:kd].T
        else:
            B, _ = np.linalg.qr(proj_out @ np.random.default_rng(100 + int(kind[-1]) * 10 + kd).normal(size=(48, kd)))
        Bt = torch.tensor(B, dtype=torch.float32)
        C = ((Hg - mu) @ Bt).double()
        beta = torch.linalg.solve(Phi.T @ Phi + 1e-3 * len(Phi) * torch.eye(Phi.shape[1], dtype=Phi.dtype), Phi.T @ C).float()
        pred = lambda h: belief_basis(*readings(h[None]), kL, kd_)[0] @ beta
        shares, offs, offp, unexpl = [], [], [], []
        for v in ev[half:]:
            tau1, tau2 = 3 * v["k"] + 1, 3 * (v["k"] + 1) + 1
            off = offset_of(v, v["Lo"][tau2])
            if abs(off) < 0.5:
                continue
            ho, hs = v["h"]
            dc = (ho - hs) @ Bt                                           # trace difference
            dc_pred = pred(ho) - pred(hs)                                 # the part the preference / belief explain
            hp = ho - (dc - dc_pred) @ Bt.T
            Lp = run_from(rnn, v["x"][tau1 + 1:], hp)
            offp_ = offset_of(v, Lp[-1])
            shares.append(1 - offp_ / off); offs.append(off); offp.append(offp_)
            unexpl.append(float((dc - dc_pred).norm() / max(dc.norm(), 1e-9)))
        out["preference_protected"][f"{kind}/{kd}d"] = {
            "offset_removed_share_median": round(float(np.median(shares)), 3),
            "mean_abs_offset_before": round(float(np.mean(np.abs(offs))), 3),
            "mean_abs_offset_after": round(float(np.mean(np.abs(offp))), 3),
            "share_of_the_difference_removed (norm)": round(float(np.median(unexpl)), 3), "n": len(shares)}
    rnd = defaultdict(list)
    for k_, v in out["results"].items():
        if k_.startswith("random"):
            rnd[k_.split("/")[1]].append(v["offset_removed_share_median"])
    out["random_summary"] = {k_: {"mean": round(float(np.mean(v)), 3), "max": round(float(np.max(v)), 3)} for k_, v in rnd.items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--rare_events", type=int, default=2000)
    ap.add_argument("--trace_events", type=int, default=1200)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    rnn, tcs, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    result = {"episodes": args.episodes}
    result["rare"] = rare_test(rnn, tracer, X, Hp, rec, rng, args.rare_events, args.top)
    print("rare", json.dumps(result["rare"]), f"({time.time() - t0:.0f}s)", flush=True)
    result["trace"] = trace_test(rnn, tracer, X, Hp, rec, rng, args.trace_events)
    print("trace", json.dumps(result["trace"]), f"({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
