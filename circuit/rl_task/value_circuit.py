"""The value circuit of the two-step-task GRU: what the value head computes, how its clock is made, and whether it
touches the choice circuit.

The critic was trained on full-episode discounted returns (A2C, gamma = 0.97 per step, terminal at the end of the
100-trial episode); the network gets no trial-number input. Closed-loop episodes of the real agent
(reversal_flow.rollout).

what      V against the discounted return-to-go G under the agent's own policy (gamma 0.97), by trial; the TD
          error delta = r + gamma V' - V by trial and, at the reward step, by outcome; V beyond its time profile
          (mean V at each of the 300 steps): its dependence on the ideal observer's confidence |b - 0.5|, the
          expected next reward under the belief and the agent's choice, and the previous reward; V and G around
          reversals (time profile removed); the inferred-value test of Bromberg-Martin et al. (2010) / Wang et al.
          (2018): the value change when a state is revealed, after its belief flipped through an outcome in the other
          state, against when it is still believed bad.
clock     a time direction in h: trial number decoded linearly from h at D1 (fit on half the episodes, scored on
          the other half); the projection's per-trial increment and how much the last outcome changes it; V as a
          function of the projection. Clock features: candidate features whose mean activation rises over the
          episode and starts at trial 50 or later; their preactivation against the projection.
time      contrastive flow (contrast_flow.full_flow) where only the episode time stored in h differs: a late window
          (trials 80-95) and an early one (trials 30-50, another episode) with the same good state and the same
          choice, state and reward on the previous 3 trials; both run on the late window's inputs for 2 trials, from
          their own carried states. Target: the value at the second trial's GO, and the choice there. Flow by source
          (the carried state's change, read directly or through features) and the features carrying it. Patching
          the top-k value-flow features per step from the early run into the late one (gates: reconstruction +
          held error, circuit_checks.run_patched): how much of the value change they carry, and what they do to
          the choice.
overlap   the features carrying the time contrast's value flow vs those carrying the credit contrasts' choice flow
          (contrast_flow.json); behaviour by stage of the episode (agreement with the observer, |choice logit|,
          stay table early vs late).

    python -m circuit.rl_task.value_circuit --out rl_features/rl_circuits/value_circuit.json [--episodes 3000]
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.circuit_checks import run_patched
from circuit.rl_task.contrast_flow import full_flow, gru_run, load_model
from circuit.rl_task.reversal_flow import observer, rollout, settled_reversals

GAMMA = 0.97
PHASES = ("D1", "GO", "D2")


def returns(R, gamma=GAMMA):
    G, run = np.zeros_like(R), np.zeros(R.shape[0])
    for t in range(R.shape[1] - 1, -1, -1):
        run = R[:, t] + gamma * run
        G[:, t] = run
    return G


def r2(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()), beta[1:].tolist()


def what(rec):
    V, R = rec["value"], rec["step_reward"]
    N, S = V.shape
    G = returns(R)
    vt, gt = V.reshape(N, 100, 3).mean((0, 2)), G.reshape(N, 100, 3).mean((0, 2))
    Vn = np.concatenate([V[:, 1:], np.zeros((N, 1))], 1)
    delta = R + GAMMA * Vn - V
    d2 = delta.reshape(N, 100, 3)[:, :, 2]                               # reward step (D2)
    rw = rec["reward"]
    out = {"gamma": GAMMA, "V_by_trial": np.round(vt, 3).tolist(), "G_by_trial": np.round(gt, 3).tolist(),
           "rmse_mean_V_vs_mean_G_by_step": round(float(np.sqrt(((V.mean(0) - G.mean(0)) ** 2).mean())), 3),
           "corr_V_G_single_steps": round(float(np.corrcoef(V.ravel(), G.ravel())[0, 1]), 3),
           "td_error_by_trial": np.round(delta.reshape(N, 100, 3).mean((0, 2)), 3).tolist(),
           "td_error_at_reward_step": {"rewarded": round(float(d2[rw == 1].mean()), 3),
                                       "unrewarded": round(float(d2[rw == 0].mean()), 3)},
           "value_change_D2_to_next_D1": {k: round(float((V[:, 3::3] - V[:, 2:-1:3])[(rw[:, :-1] == r)].mean()), 3)
                                          for k, r in (("rewarded", 1), ("unrewarded", 0))}}
    # V beyond time, at GO steps
    b = observer(rec)
    base = V.mean(0)
    res = (V - base)[:, 1::3]
    ch = rec["choice"]
    p_s1 = np.where(ch == 1, 0.8, 0.2)
    er = p_s1 * (0.9 * b + 0.1 * (1 - b)) + (1 - p_s1) * (0.1 * b + 0.9 * (1 - b))
    prev_r = np.concatenate([np.zeros((N, 1)), rw[:, :-1]], 1)
    ok = (ch > 0)
    ok[:, 0] = False
    cues = {"observer confidence |b - 0.5|": np.abs(b - 0.5), "expected next reward (belief, choice)": er,
            "previous reward": prev_r}
    out["V_beyond_time_at_GO"] = {"sd_of_V_beyond_time": round(float(res[ok].std()), 3),
                                  "sd_of_V_time_profile": round(float(base.std()), 3)}
    for k, c in cues.items():
        rr, beta = r2(res[ok], c[ok])
        out["V_beyond_time_at_GO"][k] = {"r2": round(rr, 3), "slope": round(beta[0], 3)}
    rr, _ = r2(res[ok], np.column_stack([c[ok] for c in cues.values()]))
    out["V_beyond_time_at_GO"]["all three"] = {"r2": round(rr, 3)}
    # at D2 the state reached is shown: does V jump with it (expected immediate reward), as a TD error would?
    good_now = (rec["state"] == np.where(b > 0.5, 1, 2)) & ok
    bad_now = (rec["state"] != np.where(b > 0.5, 1, 2)) & ok & (rec["state"] > 0)
    res_d2 = (V - base)[:, 2::3]
    delta_go = delta.reshape(N, 100, 3)[:, :, 1]                         # r_GO + gamma V_D2 - V_GO
    out["at_D2_by_state_reached"] = {
        name: {"V_beyond_time_at_D2": round(float(res_d2[m].mean()), 3),
               "td_error_GO_to_D2": round(float(delta_go[m].mean()), 3), "n": int(m.sum())}
        for name, m in (("believed-good state", good_now), ("believed-bad state", bad_now))}
    # inferred value (as in Bromberg-Martin et al. 2010 / Wang et al. 2018, Simulation 3): the value change when the
    # state reached is revealed (GO -> D2, beyond its time profile), on arriving at a state Y the trial after an
    # outcome in the other state flipped the ideal belief so that Y is now believed good (Y not visited since), against
    # arriving at Y while it is still believed bad
    jump = (V - base)[:, 2::3] - (V - base)[:, 1::3]
    gb = lambda bb, s_: (bb > 0.5) == (s_ == 1)
    inf = {"still believed bad": [], "now inferred good": [], "believed good (any)": []}
    chv, stv, rwv = rec["choice"], rec["state"], rec["reward"]
    for e in range(N):
        for t in range(1, 99):
            if not (chv[e, t] > 0 and chv[e, t + 1] > 0 and stv[e, t] > 0 and stv[e, t + 1] > 0 and rwv[e, t] >= 0):
                continue
            Y = stv[e, t + 1]
            if gb(b[e, t + 1], Y):
                inf["believed good (any)"].append(jump[e, t + 1])
            if Y == stv[e, t] or gb(b[e, t], Y):
                continue
            inf["now inferred good" if gb(b[e, t + 1], Y) else "still believed bad"].append(jump[e, t + 1])
    out["inferred_value_at_reveal"] = {k: {"n": len(v), "value_change_beyond_time": round(float(np.mean(v)), 3)}
                                       for k, v in inf.items()}
    Gb = (G - G.mean(0))[:, 1::3]
    around = defaultdict(lambda: ([], []))
    for e, R_, _, _ in settled_reversals(rec):
        for k in range(-3, 10):
            around[k][0].append(res[e, R_ + k]); around[k][1].append(Gb[e, R_ + k])
    out["around_reversals_beyond_time"] = {k: {"V": round(float(np.mean(v)), 3), "G": round(float(np.mean(g)), 3)}
                                           for k, (v, g) in sorted(around.items())}
    return out


def clock(rnn, tcs, X, Hp, rec, rng):
    N = X.shape[0]
    h = Hp[:, 0::3].numpy()                                              # state entering each D1
    trial = np.broadcast_to(np.arange(100), (N, 100))
    half = rng.permutation(N)
    tr_i, te_i = half[:N // 2], half[N // 2:]
    A = np.column_stack([h[tr_i].reshape(-1, 48), np.ones(len(tr_i) * 100)])
    w = np.linalg.lstsq(A, trial[tr_i].ravel(), rcond=None)[0]
    pred = h[te_i].reshape(-1, 48) @ w[:48] + w[48]
    yt = trial[te_i].ravel()
    p = (h.reshape(-1, 48) @ w[:48] + w[48]).reshape(N, 100)
    inc = p[:, 1:] - p[:, :-1]
    rw, ch, st = rec["reward"], rec["choice"], rec["state"]
    rare = (st != ch) & (ch > 0)
    rr, beta = r2(inc.ravel(), np.column_stack([rw[:, :-1].ravel(), rare[:, :-1].ravel()]))
    out = {"decode_trial_from_h_r2_test": round(float(1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()), 3),
           "decode_trial_mae_test": round(float(np.abs(yt - pred).mean()), 2),
           "projection_by_trial": np.round(p.mean(0), 2).tolist(),
           "projection_sd_by_trial": np.round(p.std(0), 2).tolist(),
           "increment_mean_sd": [round(float(inc.mean()), 3), round(float(inc.std()), 3)],
           "increment_r2_from_last_outcome": round(rr, 3),
           "increment_slopes_reward_rare": [round(b_, 3) for b_ in beta]}
    V_d1 = rec["value"][:, 0::3]
    bins = np.quantile(p, np.linspace(0, 1, 21))
    idx = np.clip(np.digitize(p, bins) - 1, 0, 19)
    curve = np.array([V_d1[idx == k].mean() for k in range(20)])
    out["V_at_D1_by_projection_bin"] = {"bin_centre": np.round((bins[:-1] + bins[1:]) / 2, 1).tolist(),
                                        "V": np.round(curve, 3).tolist()}
    out["V_at_D1_r2_from_projection_bins"] = round(float(1 - ((V_d1 - curve[idx]) ** 2).sum()
                                                         / ((V_d1 - V_d1.mean()) ** 2).sum()), 3)
    out["V_at_D1_r2_from_trial_profile"] = round(float(1 - ((V_d1 - V_d1.mean(0)) ** 2).sum()
                                                       / ((V_d1 - V_d1.mean()) ** 2).sum()), 3)
    # clock features: candidate activations on the rollouts
    gru, tc = rnn.layers[0], tcs["candidate"]
    sub = rng.choice(N, size=min(N, 600), replace=False)
    with torch.no_grad():
        x = X[sub].reshape(-1, 8); hh = Hp[sub].reshape(-1, 48)
        r = torch.sigmoid(gru.input_to_reset(x) + gru.hidden_to_reset(hh))
        pre = tc.input_to_features(torch.cat([r * hh, x], -1))
        act = tc.act(pre)
    pre = pre.numpy().reshape(len(sub), 300, -1); act = act.numpy().reshape(len(sub), 300, -1)
    ps = np.repeat(p[sub], 3, axis=1)                                    # projection at each step (from its trial's D1)
    feats = []
    for f in range(act.shape[2]):
        prof = act[:, :, f].reshape(len(sub), 100, 3).mean((0, 2))
        if prof.max() <= 0 or np.corrcoef(np.arange(100), prof)[0, 1] <= 0.6:
            continue
        onset = int(np.argmax(prof > 0.1 * prof.max()))
        if onset < 50:
            continue
        phase = int(np.argmax(act[:, :, f].reshape(len(sub), 100, 3).mean((0, 1))))
        pf, pp = pre[:, phase::3, f].ravel(), ps[:, phase::3].ravel()
        feats.append({"feature": f, "onset_trial": onset, "main_phase": PHASES[phase],
                      "corr_preact_with_time_projection": round(float(np.corrcoef(pf, pp)[0, 1]), 3),
                      "corr_preact_with_trial": round(float(np.corrcoef(pf, np.repeat(np.arange(100)[None], len(sub), 0).ravel())[0, 1]), 3)})
    out["clock_features"] = sorted(feats, key=lambda d: d["onset_trial"])
    return out, w


def matched_pairs(X, Hp, rec, rng, n_max):
    """(late episode, late trial, early episode, early trial) with the same good state and the same choice, state
    and reward on the 3 trials before."""
    ch, st, rw, good = rec["choice"], rec["state"], rec["reward"], rec["good"]
    def key(e, t):
        return (int(good[e, t]),) + tuple((int(ch[e, k]), int(st[e, k]), int(rw[e, k])) for k in range(t - 3, t))
    early = defaultdict(list)
    for e in range(ch.shape[0]):
        for t in range(30, 51):
            if (ch[e, t - 3:t + 2] > 0).all():
                early[key(e, t)].append((e, t))
    pairs = []
    for e in rng.permutation(ch.shape[0]):
        t = int(rng.integers(80, 96))
        if not (ch[e, t - 3:t + 2] > 0).all():
            continue
        cand = [c for c in early.get(key(e, t), []) if c[0] != e]
        if cand:
            pairs.append((int(e), t) + cand[int(rng.integers(len(cand)))])
        if len(pairs) >= n_max:
            break
    return pairs


def time_contrast(rnn, tcs, tracer, X, Hp, rec, pairs, top):
    K = 2
    tau = 3 * (K - 1) + 1
    agg = {tg: {"d": [], "src": defaultdict(float), "node": defaultdict(float), "checks": []} for tg in ("value", "choice")}
    items = []
    for e, t, e2, t2 in pairs:
        x = X[e, 3 * t:3 * t + tau + 1]
        acts = [tracer.run_forward_pass(gru_run(rnn, x, h0)) for h0 in (Hp[e, 3 * t], Hp[e2, 3 * t2])]
        flows = {}
        for tg, w_T in (("value", tracer.W_value), ("choice", tracer.W_choice)):
            f = full_flow(tracer, acts[0], acts[1], w_T, tau)
            flows[tg] = f
            a = agg[tg]
            a["d"].append(f["d_target"]); a["checks"].append(max(abs(f["check"]), f["preact_check"]))
            for key, v in f["into"].items():
                if float(v.abs().sum()) > 0:
                    a["src"][f"{key} -> features"] += float(v.sum())
            for key, v in f["target"].items():
                a["src"][key.replace("direct_", "") + " -> target directly"] += v
            for (kd, ts, fi), fl in zip(f["nodes"], f["flow"].tolist()):
                if fl != 0:
                    a["node"][f"{PHASES[ts % 3]}{ts // 3}/{kd}:{fi}"] += fl
        items.append((x, acts, flows["value"], (e, t, e2, t2)))
    n = len(items)
    out = {"n_pairs": n, "K_trials": K, "target_step": "GO of the second trial"}
    for tg, a in agg.items():
        node = sorted(a["node"].items(), key=lambda kv: -abs(kv[1]))
        out[tg] = {"d_target_mean": round(float(np.mean(a["d"])), 3), "d_target_sd": round(float(np.std(a["d"])), 3),
                   "max_check": float(np.max(a["checks"])),
                   "flow_by_source": {k: round(v / n, 3) for k, v in sorted(a["src"].items(), key=lambda kv: -abs(kv[1]))},
                   "top_features": [{"node": k, "flow": round(v / n, 3)} for k, v in node[:top]],
                   "features_by_total_flow": {}}
        tot = defaultdict(float)
        for k, v in a["node"].items():
            tot[k.split("/")[1]] += v / n
        out[tg]["features_by_total_flow"] = {k: round(v, 3) for k, v in sorted(tot.items(), key=lambda kv: -abs(kv[1]))[:top]}
    # patching the top-k value-flow features per step, early values into the late run
    res = defaultdict(list)
    for k in (1, 3, 5, 10, 20):
        for x, acts, f, _ in items:
            by_step = defaultdict(list)
            for nd, fl, da in zip(f["nodes"], f["flow"].tolist(), f["da"].tolist()):
                if da != 0:
                    by_step[nd[1]].append((abs(fl), nd))
            T = tau + 1
            mask = {g: torch.zeros(1, T, acts[0][f"f_{g}"].shape[1], dtype=torch.bool) for g in "rzn"}
            for lst in by_step.values():
                for _, (g, ts, fi) in sorted(lst, key=lambda v: -v[0])[:k]:
                    mask[g][0, ts, fi] = True
            value = {g: acts[1][f"f_{g}"][None] for g in "rzn"}
            err = {g: acts[0][f"e_{g}"][None] for g in "rzn"}
            steps = torch.ones(1, T, dtype=torch.bool)
            v_o, v_s = float(acts[0]["value"][tau]), float(acts[1]["value"][tau])
            c_o, c_s = float(acts[0]["choice"][tau]), float(acts[1]["choice"][tau])
            # the carried state is not patched: the late run keeps its own h0 (only features change)
            v = float(run_patched(rnn, tcs, x[None], acts[0]["h_prevs"][0][None], mask, value, steps, err, "value")[0, -1])
            c = float(run_patched(rnn, tcs, x[None], acts[0]["h_prevs"][0][None], mask, value, steps, err, "choice")[0, -1])
            if abs(v_s - v_o) >= 0.5:
                res[f"top{k}/value_share"].append((v - v_o) / (v_s - v_o))
            res[f"top{k}/choice_change"].append(c - c_o)
            res[f"top{k}/choice_change_relative_to_its_logit"].append(abs(c - c_o) / max(abs(c_o), 1e-3))
            if k == 1:
                res["unpatched/choice_difference_late_vs_early"].append(c_s - c_o)
    out["patching"] = {k: round(float(np.median(v)), 3) for k, v in sorted(res.items())}
    return out


def overlap(time_out, top):
    cf = json.load(open("rl_features/rl_circuits/contrast_flow.json"))["contrasts"]
    credit = defaultdict(float)
    for c in ("state", "reward"):
        for g in cf[c].values():
            for s, fs in g["top_features_by_step"].items():
                for x in fs:
                    credit[x["feature"]] += abs(x["flow"]) * g["n"]
    credit_top = [k for k, _ in sorted(credit.items(), key=lambda kv: -kv[1])[:top]]
    val = time_out["value"]["features_by_total_flow"]
    shared = [k for k in val if k in credit_top]
    tot = sum(abs(v) for v in val.values())
    return {"credit_top_features": credit_top, "value_time_top_features": list(val),
            "shared": shared, "share_of_value_time_flow_in_shared_features": round(sum(abs(val[k]) for k in shared) / tot, 3)}


def stages(rec):
    b = observer(rec)
    ch, st, rw, lg = rec["choice"], rec["state"], rec["reward"], rec["logit"]
    ideal = np.where(b > 0.5, 1, 2)
    out = {}
    for name, lo, hi in (("trials 10-49", 10, 50), ("trials 50-79", 50, 80), ("trials 80-99", 80, 100)):
        sl = slice(lo, hi)
        v = ch[:, sl] > 0
        stay = ch[:, lo:hi] == ch[:, lo - 1:hi - 1]
        ok = v & (ch[:, lo - 1:hi - 1] > 0)
        common = st[:, lo - 1:hi - 1] == ch[:, lo - 1:hi - 1]
        r = rw[:, lo - 1:hi - 1]
        out[name] = {"agree_with_observer": round(float((ch[:, sl][v] == ideal[:, sl][v]).mean()), 3),
                     "mean_abs_choice_logit": round(float(np.abs(lg[:, sl]).mean()), 2),
                     "stay": {f"{'common' if c else 'rare'} {'rewarded' if rr else 'unrewarded'}":
                              round(float(stay[ok & (common == c) & (r == rr)].mean()), 3) for c in (1, 0) for rr in (1, 0)}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    rnn, tcs, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    result = {"episodes": args.episodes, "what": what(rec)}
    print("what", json.dumps({k: v for k, v in result["what"].items() if "by_trial" not in k}), flush=True)
    result["clock"], _ = clock(rnn, tcs, X, Hp, rec, rng)
    print("clock", json.dumps({k: v for k, v in result["clock"].items() if "by_trial" not in k and "bin" not in k}), flush=True)
    pairs = matched_pairs(X, Hp, rec, rng, args.pairs)
    result["time"] = time_contrast(rnn, tcs, tracer, X, Hp, rec, pairs, args.top)
    print("time", json.dumps(result["time"]), flush=True)
    result["overlap"] = overlap(result["time"], args.top)
    result["stages"] = stages(rec)
    print("overlap", json.dumps(result["overlap"]), "\nstages", json.dumps(result["stages"]), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
