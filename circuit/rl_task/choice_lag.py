"""Where does the choice's lag behind the agent's own belief come from?

After a reversal the agent's belief (read from its hidden state) flips about a trial before its choice does
(trace_role.py). Closed-loop episodes (reversal_flow.rollout). "Belief" below is the linear read-out of the ideal
observer's belief logit from h after GO (trace_role.keep_out_basis); "outcome" is the state reached and whether it was
rewarded.

predictors  the choice logit at GO against what the belief predicts (a piecewise-linear fit of the logit on the
            belief): which history explains the rest? The previous choice, the previous choice's streak length,
            the previous choice x the previous reward, the trial in the episode. And how often the choice disagrees
            in sign with the agent's own belief, and how often it then repeats the previous choice.
graph       matched pairs from different episodes: trial t where the choice disagrees with the agent's own belief
            and repeats the previous choice (a "stick"), and trial t' where the choice follows the belief, with the
            same choices, states and rewards on the 2 trials before, the same choice-input history, and an ideal
            belief within 0.3. Both run on the stick trial's inputs for trial t-1 and t from their own carried states
            (so the only difference is what each carried in). Exact contrastive flow (contrast_flow.full_flow, the
            carried state's change as source) into the logit at t.GO, oriented towards the previous choice: by
            source, by step, top features, and the top D1 features' activations in the two runs. How the pairs
            differ: the previous choice's streak length, the ideal and read-out beliefs, the trial in the episode;
            their most common histories.
split       how much of the difference goes through the belief and how much around it: the starting difference
            split into its belief-subspace part (linear read-outs of the ideal observer's belief at each step of a
            trial), its part along the choice read-out outside that subspace, and the rest; exact flow of each to
            the logit; and a real-GRU swap of each part of the state entering t.GO.
patching    the top-k flow features per step from the follow-the-belief run into the stick run (gates =
            reconstruction + held error, circuit_checks.run_patched): share of the logit difference carried;
            random changed features as control.

    python -m circuit.rl_task.choice_lag --out rl_features/rl_circuits/choice_lag.json
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.bayes_vs_mb import basis
from circuit.rl_task.circuit_checks import run_patched
from circuit.rl_task.contrast_flow import full_flow, gru_run, load_model
from circuit.rl_task.reversal_flow import observer, rollout
from circuit.rl_task.trace_role import keep_out_basis

PHASES = ("D1", "GO", "D2")


def history(rec):
    ch, rw = rec["choice"], rec["reward"]
    N, T = ch.shape
    prev = np.concatenate([np.zeros((N, 1)), np.where(ch[:, :-1] == 1, 1.0, -1.0)], 1)
    streak = np.zeros((N, T))
    for t in range(2, T):
        streak[:, t] = np.where(ch[:, t - 1] == ch[:, t - 2], streak[:, t - 1] + 1, 1)
    prev_r = np.concatenate([np.zeros((N, 1)), np.clip(rw[:, :-1], 0, 1)], 1)
    return prev, streak, prev_r


def predictors(rec, dec):
    L, ch = rec["logit"], rec["choice"]
    N, T = L.shape
    m = ch > 0
    m[:, :6] = False
    for k in range(1, 6):
        m[:, 6:] &= (ch[:, 6 - k:T - k] > 0)
    x, y = dec[m], L[m]
    B = basis(x, np.percentile(x, [10, 30, 50, 70, 90]))
    res = y - B @ np.linalg.lstsq(B, y, rcond=None)[0]
    prev, streak, prev_r = history(rec)
    feats = {"previous choice": prev[m], "previous choice x streak length (capped at 10)": prev[m] * np.minimum(streak[m], 10),
             "previous choice x previous reward": prev[m] * prev_r[m],
             "trial in episode / 100": np.broadcast_to(np.arange(T), (N, T))[m] / 100}
    out = {"r2_logit_from_belief": round(float(1 - res.var() / y.var()), 3), "residual_sd": round(float(res.std()), 3),
           "alone": {}}
    for k, v in feats.items():
        A = np.column_stack([np.ones_like(v), v]); b = np.linalg.lstsq(A, res, rcond=None)[0]
        out["alone"][k] = {"slope": round(float(b[1]), 3),
                           "r2": round(float(1 - ((res - A @ b) ** 2).sum() / ((res - res.mean()) ** 2).sum()), 3)}
    A = np.column_stack([np.ones(len(res))] + list(feats.values())); b = np.linalg.lstsq(A, res, rcond=None)[0]
    out["together"] = {"r2": round(float(1 - ((res - A @ b) ** 2).sum() / ((res - res.mean()) ** 2).sum()), 3),
                       "slopes": {k: round(float(v), 3) for k, v in zip(feats, b[1:])}}
    pull = b[1] + b[3] * np.array([0.0, 1.0])
    out["pull_towards_previous_choice_beyond_belief"] = {"after an unrewarded trial": round(float(pull[0]), 3),
                                                        "after a rewarded trial": round(float(pull[1]), 3),
                                                        "note": "streak and time terms at zero"}
    lag = np.sign(x) != np.sign(y)
    out["share_choice_disagrees_with_own_belief"] = round(float(lag.mean()), 4)
    out["of_those_choice_repeats_previous"] = round(float((np.sign(y[lag]) == prev[m][lag]).mean()), 3)
    return out


def matched(rec, dec, lam, rng, n_max):
    ch, st, rw, L = rec["choice"], rec["state"], rec["reward"], rec["logit"]
    prev, streak, _ = history(rec)
    N, T = ch.shape
    key = lambda e, t: tuple((int(ch[e, k]), int(st[e, k]), int(rw[e, k])) for k in range(t - 2, t)) + (int(ch[e, t]),)
    follow = defaultdict(list)
    sticks = []
    for e in range(N):
        for t in range(6, T):
            if (ch[e, t - 2:t + 1] <= 0).any() or rw[e, t - 1] < 0 or rw[e, t - 2] < 0:
                continue
            agree = np.sign(L[e, t]) == np.sign(dec[e, t])
            if agree:
                follow[key(e, t)].append((e, t))
            elif np.sign(L[e, t]) == prev[e, t]:
                sticks.append((e, t))
    pairs = []
    for i in rng.permutation(len(sticks)):
        e, t = sticks[i]
        cand = [(e2, t2) for e2, t2 in follow.get(key(e, t), []) if e2 != e and abs(lam[e2, t2] - lam[e, t]) < 0.3]
        if cand:
            pairs.append((e, t) + cand[int(rng.integers(len(cand)))])
        if len(pairs) >= n_max:
            break
    return pairs


def belief_subspace(Hp, lam):
    """Orthonormal basis of linear read-outs of the ideal observer's belief from h entering each step of a trial
    (D1, GO, D2), each fitted to the belief before this trial's choice and, for D1 (whose last reward is not shown
    yet), also to the belief before the previous trial's; with each read-out's R^2."""
    N, T = lam.shape
    lam_prev = np.concatenate([np.zeros((N, 1)), lam[:, :-1]], 1)
    cols, r2 = [], {}
    for ph, name in ((0, "D1"), (1, "GO"), (2, "D2")):
        H = Hp[:, ph::3].numpy().reshape(-1, 48)
        for tgt_name, tgt in (("this trial's belief", lam), ("previous trial's belief", lam_prev)):
            if ph > 0 and tgt_name != "this trial's belief":
                continue
            A = np.column_stack([H, np.ones(len(H))])
            w = np.linalg.lstsq(A, tgt.reshape(-1), rcond=None)[0]
            p = A @ w
            r2[f"entering {name}, {tgt_name}"] = round(float(1 - ((tgt.reshape(-1) - p) ** 2).sum()
                                                       / ((tgt.reshape(-1) - tgt.mean()) ** 2).sum()), 3)
            cols.append(w[:48])
    B, _ = np.linalg.qr(np.stack(cols, 1))
    return B, r2


def split(rnn, tracer, Hp, rec, lam, items):
    """How much of the stick-vs-follow logit difference goes through the belief and how much around it. Parts of a
    state difference: its component in the belief subspace, its component along the choice read-out direction
    outside that subspace, and the rest.
    graph  the pair's starting difference (the carried-in memory) split into those parts; exact flow of each part
           to the logit at t.GO (the parts' flows add up to the whole).
    swap   the state entering t.GO in the stick run with one part replaced by the follow run's; the real GO step
           computes the logit: share of the stick-vs-follow difference removed."""
    B, r2 = belief_subspace(Hp, lam)
    Pb = B @ B.T
    w = tracer.W_choice.detach().numpy()
    wr = w - Pb @ w
    wr = wr / np.linalg.norm(wr)
    Pr = np.outer(wr, wr)
    parts = {"belief subspace": Pb, "choice read-out (outside the belief subspace)": Pr, "the rest": np.eye(48) - Pb - Pr}
    P_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in parts.items()}
    graph, swap = defaultdict(list), defaultdict(list)
    share_norm = defaultdict(list)
    for it in items:
        ao, asw = it["acts"]
        d0 = ao["h_prevs"][0] - asw["h_prevs"][0]
        tot = it["f"]["d_target"]
        for k, P in P_t.items():
            f = full_flow(tracer, ao, asw, it["sign"] * tracer.W_choice, 4, init_delta=P @ d0)
            graph[k].append(float(f["into"]["init"].sum()) + f["target"]["direct_init"])
            share_norm[k].append(float((P @ d0).norm() / d0.norm()))
        graph["whole"].append(tot)
        hs, hf = ao["h_prevs"][4], asw["h_prevs"][4]                      # entering t.GO
        dg = hs - hf
        x_go = it["x"][4:5]
        L_s = it["sign"] * float(ao["choice"][4]); L_f = it["sign"] * float(asw["choice"][4])
        for k, P in list(P_t.items()) + [("belief subspace and choice read-out", P_t["belief subspace"]
                                           + P_t["choice read-out (outside the belief subspace)"])]:
            g = gru_run(rnn, x_go, hs - P @ dg)
            h1 = (1 - g["z_ts"][0]) * g["h_prevs"][0] + g["z_ts"][0] * g["h_new_ts"][0]
            L_p = it["sign"] * float(h1 @ tracer.W_choice + tracer.b_choice)
            if abs(L_s - L_f) >= 1:
                swap[k].append((L_s - L_p) / (L_s - L_f))
        for k, P in P_t.items():
            share_norm[f"entering t.GO: {k}"].append(float((P @ dg).norm() / dg.norm()))
    return {"belief_readout_r2": r2, "belief_subspace_dims": int(B.shape[1]),
            "graph_flow_of_each_part_mean": {k: round(float(np.mean(v)), 3) for k, v in graph.items()},
            "swap_share_of_difference_removed_median": {k: round(float(np.median(v)), 3) for k, v in swap.items()},
            "share_of_the_state_difference_norm": {k: round(float(np.mean(v)), 3) for k, v in share_norm.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    rnn, tcs, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    Q, wdec, b0 = keep_out_basis(tracer, Hp, rec, with_decoder=True)
    dec = Hp[:, 2::3].numpy() @ wdec + b0
    b = observer(rec)
    lam = np.log(np.clip(b, 1e-6, 1) / np.clip(1 - b, 1e-6, 1))
    result = {"episodes": args.episodes, "predictors": predictors(rec, dec)}
    print("predictors", json.dumps(result["predictors"]), flush=True)
    pairs = matched(rec, dec, lam, rng, args.pairs)
    prev, streak, _ = history(rec)
    agg = {"src": defaultdict(float), "step": defaultdict(float), "node": defaultdict(float)}
    diffs = defaultdict(list)
    items, checks = [], []
    ch_, st_, rw_ = rec["choice"], rec["state"], rec["reward"]
    hist = defaultdict(int)
    for e, t, e2, t2 in pairs:
        x = X[e, 3 * (t - 1):3 * t + 2]                                  # trial t-1 D1 .. trial t GO
        tau = 4
        acts = [tracer.run_forward_pass(gru_run(rnn, x, h0)) for h0 in (Hp[e, 3 * (t - 1)], Hp[e2, 3 * (t2 - 1)])]
        sign = float(prev[e, t])                                         # towards the previous choice
        f = full_flow(tracer, acts[0], acts[1], sign * tracer.W_choice, tau)
        checks.append(max(abs(f["check"]), f["preact_check"]))
        diffs["logit difference (stick minus follow), towards the previous choice"].append(f["d_target"])
        diffs["streak length of the previous choice: stick"].append(streak[e, t])
        diffs["streak length of the previous choice: follow"].append(streak[e2, t2])
        diffs["ideal belief towards the previous choice: stick"].append(sign * lam[e, t])
        diffs["ideal belief towards the previous choice: follow"].append(sign * lam[e2, t2])
        diffs["read-out belief towards the previous choice: stick"].append(sign * dec[e, t])
        diffs["read-out belief towards the previous choice: follow (its own run)"].append(sign * dec[e2, t2])
        diffs["trial in episode: stick"].append(t); diffs["trial in episode: follow"].append(t2)
        for key, v in f["into"].items():
            agg["src"][f"{key} -> features"] += float(v.sum())
        for key, v in f["target"].items():
            agg["src"][key.replace("direct_", "") + " -> logit directly"] += v
        for (kd, ts, fi), fl in zip(f["nodes"], f["flow"].tolist()):
            if fl != 0:
                name = f"{'t-1' if ts < 3 else 't'}.{PHASES[ts % 3]}"
                agg["step"][name] += fl
                agg["node"][f"{name}/{kd}:{fi}"] += fl
        items.append({"x": x, "acts": acts, "f": f, "h0": Hp[e, 3 * (t - 1)], "sign": sign})
        hist[tuple(int(v) for v in (ch_[e, t - 2], st_[e, t - 2], rw_[e, t - 2], ch_[e, t - 1], st_[e, t - 1], rw_[e, t - 1], ch_[e, t]))] += 1
    n = len(items)
    result["graph"] = {
        "n_pairs": n, "max_check": float(np.max(checks)),
        "pair_differences_mean": {k: round(float(np.mean(v)), 3) for k, v in diffs.items()},
        "flow_by_source": {k: round(v / n, 3) for k, v in sorted(agg["src"].items(), key=lambda kv: -abs(kv[1]))},
        "flow_through_step": {k: round(v / n, 3) for k, v in sorted(agg["step"].items())},
        "top_features": {k: round(v / n, 3) for k, v in sorted(agg["node"].items(), key=lambda kv: -abs(kv[1]))[:args.top]}}
    result["graph"]["most_common_histories (choice, state, reward at t-2 and t-1; choice at t)"] = [
        [list(k), v] for k, v in sorted(hist.items(), key=lambda kv: -kv[1])[:4]]
    # the top D1 features' activations in the two runs, at the D1s that read trial t-2's and t-1's outcomes
    d1_feats = sorted({k.split("/")[1] for k in result["graph"]["top_features"] if ".D1/" in k})
    result["graph"]["d1_feature_activations_mean"] = {
        fk: {"reads trial t-2's outcome (t-1.D1)": {"stick": round(float(np.mean([it["acts"][0][f"f_{fk[0]}"][0, int(fk[2:])] for it in items])), 3),
                                                     "follow": round(float(np.mean([it["acts"][1][f"f_{fk[0]}"][0, int(fk[2:])] for it in items])), 3)},
             "reads trial t-1's outcome (t.D1)": {"stick": round(float(np.mean([it["acts"][0][f"f_{fk[0]}"][3, int(fk[2:])] for it in items])), 3),
                                                  "follow": round(float(np.mean([it["acts"][1][f"f_{fk[0]}"][3, int(fk[2:])] for it in items])), 3)}}
        for fk in d1_feats}
    print("graph", json.dumps(result["graph"]), flush=True)
    result["split"] = split(rnn, tracer, Hp, rec, lam, items)
    print("split", json.dumps(result["split"]), flush=True)
    pat = defaultdict(list)
    for k in (1, 3, 5, 10):
        for rnd in (False, True):
            for it in items:
                f = it["f"]
                by = defaultdict(list)
                for nd, fl, da in zip(f["nodes"], f["flow"].tolist(), f["da"].tolist()):
                    if da != 0:
                        by[nd[1]].append((abs(fl), nd))
                chosen = []
                for lst in by.values():
                    lst.sort(key=lambda v: -v[0])
                    if rnd:
                        rest = [nd for _, nd in lst[k:]]
                        chosen += [rest[j] for j in rng.choice(len(rest), size=min(k, len(rest)), replace=False)] if rest else []
                    else:
                        chosen += [nd for _, nd in lst[:k]]
                T_ = 5
                mask = {g: torch.zeros(1, T_, it["acts"][0][f"f_{g}"].shape[1], dtype=torch.bool) for g in "rzn"}
                for g, ts, fi in chosen:
                    mask[g][0, ts, fi] = True
                value = {g: it["acts"][1][f"f_{g}"][None] for g in "rzn"}
                err = {g: it["acts"][0][f"e_{g}"][None] for g in "rzn"}
                steps = torch.ones(1, T_, dtype=torch.bool)
                y = it["sign"] * float(run_patched(rnn, tcs, it["x"][None], it["h0"][None], mask, value, steps, err)[0, -1])
                y_o, y_s = (it["sign"] * float(a["choice"][4]) for a in it["acts"])
                if abs(y_o - y_s) >= 1:
                    pat[f"{'random' if rnd else 'top'}{k}"].append((y_o - y) / (y_o - y_s))
    result["patching_share_of_difference_carried_median"] = {k: round(float(np.median(v)), 3) for k, v in sorted(pat.items())}
    print("patching", json.dumps(result["patching_share_of_difference_carried_median"]), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
