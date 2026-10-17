"""One belief or two values? Where the two-step-task GRU keeps the evidence of an S1 outcome and of an S2 outcome.

A Bayesian agent keeps one variable (how likely S1 is the good state): a reward in S1 and no reward in S2 are evidence
about the same thing. Model-based RL with two values keeps Q(S1) and Q(S2): an outcome updates only the value of the
state it happened in, and the two meet only at the choice. Decoders cannot tell these apart here (the belief and
Q(S1) - Q(S2) are both decoded from h with R^2 0.98; bayes_vs_mb.py), so this script asks the network directly.

Closed-loop episodes (reversal_flow.rollout). An event: trial u (the evidence), trial v = u + 1 (the next trial),
trial t = u + 2. The window runs from trial u - 2 to t.GO, inputs as recorded; the contrast swaps trial u's reward
(shown at v.D1). Pairs are oriented so the original's outcome is the one that favours S1 (and a1).

survival  how much of the swap's effect on the choice logit at v.GO is still there at t.GO, after trial v's outcome
          (median), split by whether trial v visited the same state as trial u or the other one, and whether trial
          v's outcome confirmed or opposed trial u's. Two-value MB-RL predicts better survival after a visit to the
          other state (that visit leaves the first value alone); a Bayesian predicts no dependence on which state,
          only on confirming vs opposing (saturation). Also split by trial v's transition (common / rare), since a
          settled agent reaches the other state mostly by a rare transition. The same numbers for the ideal observer (belief logit) and for
          the MB-RL and Bayesian models fitted in bayes_vs_mb.py (their policy logits).
graph     exact contrastive flow into the logit at t.GO (contrast_flow.full_flow, features of all three gates).
          For evidence from S1 outcomes and from S2 outcomes separately: the mean flow through each feature at each
          step; per step, the cosine similarity of the two groups' flow vectors (1 = the same features carry both;
          0 = separate features) and their top features. Also the change in h after each step (mean, oriented) for
          the two groups: its cosine between them, its component along a linear decoder of the ideal observer's
          belief (fitted on all steps of that phase) and along the action preference direction (the choice
          read-out minus its overlap with the belief subspace of choice_lag.belief_subspace), and the cosine of the remainder between the groups when
          oriented by the evidence and when oriented by the reward itself (a "rewarded or not" memory).

    python -m circuit.rl_task.belief_axis --out rl_features/rl_circuits/belief_axis.json [--events 1200]
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.bayes_vs_mb import bayes_model, mb_model
from circuit.rl_task.contrast_flow import REW, full_flow, gru_run, load_model
from circuit.rl_task.reversal_flow import HAZARD, rollout

PHASES = ("D1", "GO", "D2")
STEP_NAMES = {9: "v.D1 (u's reward shown)", 10: "v.GO", 11: "v.D2", 12: "t.D1 (v's reward shown)", 13: "t.GO"}


def favours_s1(s, r):
    return (s == 1 and r == 1) or (s == 2 and r == 0)


def observer_two(st, rw, ch, upto):
    """Ideal-observer belief logit before each trial's choice, trials 0 .. upto (one episode)."""
    lam, out = 0.0, []
    for k in range(upto + 1):
        out.append(lam)
        s, r = st[k], rw[k]
        if ch[k] > 0 and s > 0 and r in (0, 1):
            ev = np.log(9.0) * (1 if s == 1 else -1) * (1 if r == 1 else -1)
            b = 1 / (1 + np.exp(-(lam + ev)))
            b = b * (1 - HAZARD) + (1 - b) * HAZARD
            lam = float(np.log(b / (1 - b)))
    return out


def model_params(fit):
    lg = lambda p: float(np.log(p / (1 - p)))
    mb = fit["model_based_rl_two_values"]["params"]
    by = fit["bayesian"]["params"]
    return (torch.tensor([lg(mb["learning_rate"]), lg(mb["forgetting"]), mb["inverse_temperature"], mb["bias"],
                          mb["perseveration"]]),
            torch.tensor([by["evidence_rewarded"], by["evidence_unrewarded"], lg(by["hazard"] / 0.5),
                          by["inverse_temperature"], by["bias"], by["perseveration"]]))


def cos(a, b):
    keys = set(a) | set(b)
    va, vb = np.array([a.get(k, 0.0) for k in keys]), np.array([b.get(k, 0.0) for k in keys])
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(va @ vb / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--events", type=int, default=1200)
    ap.add_argument("--fits", default="rl_features/rl_circuits/bayes_vs_mb.json")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    rnn, tcs, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    ch, st, rw = rec["choice"], rec["state"], rec["reward"]
    p_mb, p_by = model_params(json.load(open(args.fits))["models"])
    cand = [(e, u) for e in range(ch.shape[0]) for u in range(4, 97)
            if (ch[e, u - 2:u + 3] > 0).all() and (st[e, u - 2:u + 3] > 0).all() and rw[e, u] in (0, 1) and rw[e, u + 1] in (0, 1)]
    picks = [cand[i] for i in rng.permutation(len(cand))[:args.events]]
    surv = defaultdict(lambda: defaultdict(list))
    flows = {g: defaultdict(lambda: defaultdict(float)) for g in ("S1 outcome", "S2 outcome")}
    dh = {g: defaultdict(lambda: torch.zeros(48)) for g in flows}
    dh_r = {g: defaultdict(lambda: torch.zeros(48)) for g in flows}
    # belief direction: linear decoder of the ideal observer's belief logit from h after each step, per phase
    from circuit.rl_task.reversal_flow import observer
    b_all = observer(rec)
    lam_all = np.log(np.clip(b_all, 1e-6, 1) / np.clip(1 - b_all, 1e-6, 1))
    h_after = torch.cat([Hp[:, 1:], Hp[:, -1:]], 1).numpy()              # h after each step (last step approximate)
    # action preference direction: the choice read-out minus its overlap with the belief subspace
    from circuit.rl_task.choice_lag import belief_subspace
    Bsub, _ = belief_subspace(Hp, lam_all)
    wch = tracer.W_choice.detach().numpy()
    ap_dir = wch - Bsub @ (Bsub.T @ wch)
    ap_dir = torch.tensor(ap_dir / np.linalg.norm(ap_dir), dtype=torch.float32)
    dec = {}
    for ph in range(3):
        Hs = h_after[:, ph::3].reshape(-1, 48)
        # the belief that h after this step should hold: after D1 the trial's prior is updated with the last reward
        tgt = (lam_all if ph >= 1 else np.concatenate([lam_all[:, :1], lam_all[:, :-1]], 1)).reshape(-1)
        A = np.column_stack([Hs, np.ones(len(Hs))])
        wdec = np.linalg.lstsq(A, tgt, rcond=None)[0][:48]
        dec[ph] = torch.tensor(wdec, dtype=torch.float32)
    n_g = defaultdict(int)
    checks = []
    w = tracer.W_choice
    for e, u in picks:
        lo, v, t = u - 2, u + 1, u + 2
        x = X[e, 3 * lo:3 * t + 2].clone()
        xc = x.clone()
        xc[9, REW] = 1 - xc[9, REW]                                      # trial u's reward, shown at v.D1
        sign = 1.0 if favours_s1(st[e, u], rw[e, u]) else -1.0
        acts = [tracer.run_forward_pass(gru_run(rnn, xx, Hp[e, 3 * lo])) for xx in (x, xc)]
        Lv = sign * (float(acts[0]["choice"][10]) - float(acts[1]["choice"][10]))
        Lt = sign * (float(acts[0]["choice"][13]) - float(acts[1]["choice"][13]))
        same = "same state" if st[e, v] == st[e, u] else "other state"
        trans = "common" if st[e, v] == ch[e, v] else "rare"
        conf = "confirming" if favours_s1(st[e, v], rw[e, v]) == (sign > 0) else "opposing"
        group = "S1 outcome" if st[e, u] == 1 else "S2 outcome"
        key = f"{same}, {conf}"
        rw_c = rw[e].copy(); rw_c[u] = 1 - rw_c[u]
        lam = [observer_two(st[e], r_, ch[e], t) for r_ in (rw[e], rw_c)]
        with torch.no_grad():
            S, C = torch.tensor(np.stack([st[e], st[e]])), torch.tensor(np.stack([ch[e], ch[e]]))
            R = torch.tensor(np.stack([rw[e], rw_c])).clamp(min=0).float()
            V = (C > 0) & (S > 0) & torch.tensor(np.stack([rw[e] >= 0, rw_c >= 0]))
            mbl, _ = mb_model(p_mb, S, R, C, V)
            byl, _ = bayes_model(p_by, S, R, C, V)
        for name, (a_v, a_t) in (("agent", (Lv, Lt)),
                                 ("ideal observer", (sign * (lam[0][v] - lam[1][v]), sign * (lam[0][t] - lam[1][t]))),
                                 ("fitted MB-RL", (sign * float(mbl[0, v] - mbl[1, v]), sign * float(mbl[0, t] - mbl[1, t]))),
                                 ("fitted Bayesian", (sign * float(byl[0, v] - byl[1, v]), sign * float(byl[0, t] - byl[1, t])))):
            if abs(a_v) > 1e-3:
                surv[name][key].append(a_t / a_v)
                surv[name][same].append(a_t / a_v)
                surv[name][f"{same}, {trans} transition"].append(a_t / a_v)
                surv[name][f"{same}, {trans} transition, {conf}"].append(a_t / a_v)
        f = full_flow(tracer, acts[0], acts[1], sign * w, 13)
        checks.append(max(abs(f["check"]), f["preact_check"]))
        n_g[group] += 1
        for (kd, ts, fi), fl in zip(f["nodes"], f["flow"].tolist()):
            if ts in STEP_NAMES and fl != 0:
                flows[group][ts][f"{kd}:{fi}"] += fl
        rsign = 1.0 if rw[e, u] == 1 else -1.0                           # orientation by the reward itself
        for ts in STEP_NAMES:
            d_ = acts[0]["h_ts"][ts] - acts[1]["h_ts"][ts]
            dh[group][ts] += sign * d_
            dh_r[group][ts] += rsign * d_
    result = {"episodes": args.episodes, "events": len(picks), "max_check": float(np.max(checks)),
              "n_by_group": dict(n_g),
              "survival_median": {name: {k: {"median": round(float(np.median(v)), 3), "n": len(v)} for k, v in sorted(d.items())}
                                  for name, d in surv.items()},
              "graph": {}}
    for ts, sn in STEP_NAMES.items():
        a = {k: v / n_g["S1 outcome"] for k, v in flows["S1 outcome"][ts].items()}
        b = {k: v / n_g["S2 outcome"] for k, v in flows["S2 outcome"][ts].items()}
        ha, hb = dh["S1 outcome"][ts] / n_g["S1 outcome"], dh["S2 outcome"][ts] / n_g["S2 outcome"]
        ra, rb = dh_r["S1 outcome"][ts] / n_g["S1 outcome"], dh_r["S2 outcome"][ts] / n_g["S2 outcome"]
        d_ = dec[ts % 3]
        u_ = d_ / d_.norm()
        perp = lambda v_: v_ - (v_ @ u_) * u_
        result["graph"][sn] = {
            "flow_vector_cosine_S1_vs_S2_evidence": round(cos(a, b), 3),
            "total_flow": {"S1 outcome": round(sum(a.values()), 3), "S2 outcome": round(sum(b.values()), 3)},
            "dh_cosine_S1_vs_S2_evidence": round(float(torch.nn.functional.cosine_similarity(ha, hb, 0)), 3),
            "dh_norm": {"S1 outcome": round(float(ha.norm()), 3), "S2 outcome": round(float(hb.norm()), 3)},
            "dh_along_belief_decoder_in_belief_logits": {"S1 outcome": round(float(ha @ d_), 3), "S2 outcome": round(float(hb @ d_), 3)},
            "dh_along_action_preference_direction": {"S1 outcome": round(float(ha @ ap_dir), 3), "S2 outcome": round(float(hb @ ap_dir), 3)},
            "dh_share_of_norm_along_belief_decoder": {"S1 outcome": round(float(abs(ha @ u_) / ha.norm()), 3),
                                                      "S2 outcome": round(float(abs(hb @ u_) / hb.norm()), 3)},
            "dh_rest_cosine_oriented_by_reward": round(float(torch.nn.functional.cosine_similarity(perp(ra), perp(rb), 0)), 3),
            "dh_rest_cosine_oriented_by_evidence": round(float(torch.nn.functional.cosine_similarity(perp(ha), perp(hb), 0)), 3),
            "top_features": {g: dict(sorted(((k, round(v_, 3)) for k, v_ in d.items()), key=lambda kv: -abs(kv[1]))[:args.top])
                             for g, d in (("S1 outcome", a), ("S2 outcome", b))}}
    print(json.dumps(result["survival_median"]), flush=True)
    for sn, g in result["graph"].items():
        print(sn, json.dumps({k: v for k, v in g.items() if k != "top_features"}), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
