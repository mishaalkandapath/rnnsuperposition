"""Is the two-step-task GRU a Bayesian observer or model-based RL, and what are the bands of its phase portrait?

Ji-An, Benna & Mattar (2025, Nature 644: 993, "Discovering cognitive strategies with tiny recurrent neural networks")
plot each trial's policy logit L(t) = log P(a1) / P(a2) at GO against the logit change dL = L(t+1) - L(t), coloured
by the trial's state and reward. A one-dimensional Bayesian agent gives one curve per input type; their meta-RL agent
(same task, rewards 0.8 / 0.2) gave Bayesian-like curves split into parallel bands (a "history effect"). Closed-loop
episodes of our agent (reversal_flow.rollout; rewards 0.9 / 0.1).

portrait  the agent's (L, dL) points by input type (binned means and spread, and 6000 sampled points), and the
          ideal observer's (hazard 0.025) for comparison; the Bayesian symmetry (a rewarded S1 and an unrewarded S2 are the same evidence).
bands     how much of dL's spread around a curve per input type (piecewise-linear in L) the history explains: the
          current transition, the previous trial's input type and transition, the one before, episode time.
models    cognitive models fitted to the agent's choices (half the episodes; log-likelihood per trial on the other
          half; ceiling = the agent's own choice probabilities): Bayesian (evidence per rewarded / unrewarded
          outcome, hazard, inverse temperature, bias, perseveration) and model-based RL with two state values
          (learning rate, forgetting of the unvisited value, inverse temperature, bias, perseveration).
geometry  one memory axis or two: the observer's belief logit and the fitted MB-RL values decoded linearly from h
          after D1 (reward seen) and before it; the MB-RL dimension a belief does not have, Q(S1) + Q(S2), beyond
          the belief (also offered to `bands` as a band variable).
go_split  the logit at GO split exactly into the carried memory, w . (1 - z) h, and the GO step's write,
          w . z n = sum_f w . (z d_f) a_f + bias / error writes; how each depends on the previous outcome beyond the
          belief, whether the write explains the bands, how much of each part survives into the next trial's logit
          (exact, h being linear in its writes given the gates), and the GO features whose writes depend most on
          the previous trial's input type.

persistence  one feature pushed (+1 activation) at trial t, in the real GRU: the logit change at GO of trial t and
          of trial t + 1. GO features (readout) vs next-D1 detectors (which write the memory).

    python -m circuit.rl_task.bayes_vs_mb --out rl_features/rl_circuits/bayes_vs_mb.json [--episodes 3000]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from circuit.rl_task.contrast_flow import load_model
from circuit.rl_task.reversal_flow import HAZARD, observer, rollout

TYPES = ("S1 R=1", "S1 R=0", "S2 R=1", "S2 R=0")


def trial_table(rec):
    L, ch, st, rw = rec["logit"], rec["choice"], rec["state"], rec["reward"]
    N = L.shape[0]
    typ = np.where(st > 0, (st - 1) * 2 + (1 - np.clip(rw, 0, 1)).astype(int), -1)
    rare = ((st != ch) & (ch > 0)).astype(int)
    valid = (ch > 0) & (st > 0) & (rw >= 0)
    sh = lambda a, f: np.concatenate([np.full((N, 1), f, a.dtype), a[:, :-1]], 1)
    return {"L": L, "dL": np.concatenate([L[:, 1:] - L[:, :-1], np.zeros((N, 1))], 1), "typ": typ, "rare": rare,
            "valid": valid, "prev_typ": sh(typ, -1), "prev_rare": sh(rare, 0), "prev2_typ": sh(sh(typ, -1), -1),
            "trial": np.broadcast_to(np.arange(L.shape[1]), L.shape)}


def mask_for(tt):
    m = tt["valid"].copy()
    m[:, 1:-1] &= tt["valid"][:, :-2] & tt["valid"][:, 2:]                # previous and next trial valid
    m[:, :3] = False; m[:, -1] = False
    return m


def basis(x, knots):
    return np.column_stack([np.ones_like(x), x] + [np.maximum(x - k, 0) for k in knots])


def band_r2(y, x, T, extras):
    knots = np.percentile(x, [10, 30, 50, 70, 90])
    B = basis(x, knots)
    cols = [B * (T == t)[:, None] for t in range(4)]
    for ex in extras:
        for t in range(4):
            for lv in np.unique(ex)[1:]:
                cols.append(((T == t) & (ex == lv)).astype(float)[:, None] * B)
    D = np.column_stack(cols)
    beta = np.linalg.lstsq(D, y, rcond=None)[0]
    return float(1 - ((y - D @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()), D @ beta


def portrait(tt, lam, dlam, m):
    bins = np.arange(-12, 13, 2)
    out = {"agent": {}, "ideal_observer_belief_logit": {}}
    for name, (x, y) in (("agent", (tt["L"], tt["dL"])), ("ideal_observer_belief_logit", (lam, dlam))):
        for t, tn in enumerate(TYPES):
            sel = m & (tt["typ"] == t)
            xs, ys = x[sel], y[sel]
            rows = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                k = (xs >= lo) & (xs < hi)
                if k.sum() >= 50:
                    rows.append({"L": float((lo + hi) / 2), "dL_mean": round(float(ys[k].mean()), 3),
                                 "dL_sd": round(float(ys[k].std()), 3), "n": int(k.sum())})
            out[name][tn] = rows
    sym = {}
    for a, b in ((0, 3), (1, 2)):
        rows = []
        for lo, hi in ((-9, -5), (-5, -1), (-1, 1), (1, 5), (5, 9)):
            ma = m & (tt["typ"] == a) & (tt["L"] > lo) & (tt["L"] < hi)
            mb = m & (tt["typ"] == b) & (tt["L"] > lo) & (tt["L"] < hi)
            if ma.sum() > 30 and mb.sum() > 30:
                rows.append({"L": [lo, hi], TYPES[a]: round(float(tt["dL"][ma].mean()), 2),
                             TYPES[b]: round(float(tt["dL"][mb].mean()), 2)})
        sym[f"{TYPES[a]} vs {TYPES[b]}"] = rows
    out["bayes_symmetry"] = sym
    idx = np.flatnonzero(m.ravel())
    pick = np.random.default_rng(0).choice(idx, size=min(6000, len(idx)), replace=False)
    out["points"] = {k: np.round(tt[k].ravel()[pick], 3).tolist() for k in ("L", "dL")}
    out["points"].update({k: tt[k].ravel()[pick].astype(int).tolist() for k in ("typ", "prev_typ")})
    out["points"]["observer_belief_logit"] = np.round(lam.ravel()[pick], 3).tolist()
    out["points"]["observer_dlam"] = np.round(dlam.ravel()[pick], 3).tolist()
    return out


def bands(tt, m, qsum_resid=None):
    y, x, T = tt["dL"][m], tt["L"][m], tt["typ"][m]
    base, _ = band_r2(y, x, T, [])
    out = {"r2_curve_per_input_type": round(base, 4), "sd_dL": round(float(y.std()), 3), "added": {}}
    cands = {"current transition": [tt["rare"][m]], "previous input type": [tt["prev_typ"][m]],
             "previous transition": [tt["prev_rare"][m]],
             "previous input type and transition": [tt["prev_typ"][m] * 2 + tt["prev_rare"][m]],
             "previous reward only": [np.where(tt["prev_typ"][m] < 0, -1, tt["prev_typ"][m] % 2)],
             "previous state only": [np.where(tt["prev_typ"][m] < 0, -1, tt["prev_typ"][m] // 2)],
             "input type two trials back": [tt["prev2_typ"][m]], "episode time (decile)": [tt["trial"][m] // 10],
             "all of these": [tt["rare"][m], tt["prev_typ"][m] * 2 + tt["prev_rare"][m], tt["prev2_typ"][m],
                              tt["trial"][m] // 10]}
    if qsum_resid is not None:
        q = qsum_resid[m]
        cands["MB-RL Q(S1) + Q(S2) beyond the belief (quintiles)"] = [np.digitize(q, np.percentile(q, [20, 40, 60, 80]))]
    for k, ex in cands.items():
        r, _ = band_r2(y, x, T, ex)
        out["added"][k] = {"r2": round(r, 4), "share_of_spread_explained": round((r - base) / (1 - base), 3)}
    return out


def mb_model(params, st, rw, ch, valid):
    """Model-based RL, two state values. Returns the policy logit (N, T) before each trial's choice."""
    alpha, phi = torch.sigmoid(params[0]), torch.sigmoid(params[1])
    beta, bias, pers = params[2], params[3], params[4]
    N, T = st.shape
    q1, q2 = torch.full((N,), 0.5), torch.full((N,), 0.5)
    prev = torch.zeros(N)
    out, qs = [], []
    for t in range(T):
        out.append(beta * 0.6 * (q1 - q2) + bias + pers * prev)
        qs.append(torch.stack([q1, q2], 1))
        v, s, r = valid[:, t], st[:, t], rw[:, t]
        n1 = torch.where(v & (s == 1), q1 + alpha * (r - q1), torch.where(v, q1 + phi * (0.5 - q1), q1))
        n2 = torch.where(v & (s == 2), q2 + alpha * (r - q2), torch.where(v, q2 + phi * (0.5 - q2), q2))
        q1, q2 = n1, n2
        prev = torch.where(v, torch.where(ch[:, t] == 1, 1.0, -1.0), prev)
    return torch.stack(out, 1), torch.stack(qs, 1)


def bayes_model(params, st, rw, ch, valid):
    """Bayesian observer: evidence per rewarded / unrewarded outcome (in logits), hazard. Policy logit (N, T)."""
    lr, lu, h = params[0], params[1], torch.sigmoid(params[2]) * 0.5
    beta, bias, pers = params[3], params[4], params[5]
    N, T = st.shape
    lam = torch.zeros(N)
    prev = torch.zeros(N)
    out, lams = [], []
    for t in range(T):
        b = torch.sigmoid(lam)
        out.append(beta * (2 * b - 1) + bias + pers * prev)
        lams.append(lam)
        v, s, r = valid[:, t], st[:, t], rw[:, t]
        sign = torch.where(s == 1, 1.0, -1.0) * torch.where(r == 1, 1.0, -1.0)   # + = evidence that S1 is good
        ev = sign * torch.where(r == 1, lr, lu)
        b2 = torch.sigmoid(lam + torch.where(v, ev, torch.zeros_like(ev)))
        b2 = b2 * (1 - h) + (1 - b2) * h
        lam = torch.logit(b2.clamp(1e-6, 1 - 1e-6))
        prev = torch.where(v, torch.where(ch[:, t] == 1, 1.0, -1.0), prev)
    return torch.stack(out, 1), torch.stack(lams, 1)


def fit(model, init, st, rw, ch, valid, steps=400):
    p = torch.tensor(init, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([p], lr=0.05)
    y = (ch == 1).float()
    for _ in range(steps):
        logit, _ = model(p, st, rw, ch, valid)
        nll = torch.nn.functional.binary_cross_entropy_with_logits(logit[valid], y[valid])
        opt.zero_grad(); nll.backward(); opt.step()
    return p.detach()


def models(rec, rng):
    st, rw, ch = (torch.tensor(rec[k]) for k in ("state", "reward", "choice"))
    rw = rw.clamp(min=0)
    valid = (ch > 0) & (st > 0) & torch.tensor(rec["reward"] >= 0)
    N = st.shape[0]
    perm = rng.permutation(N)
    tr, te = torch.tensor(perm[:N // 2]), torch.tensor(perm[N // 2:])
    y = (ch == 1).float()
    L_agent = torch.tensor(rec["logit"])
    out = {}
    ceiling = -torch.nn.functional.binary_cross_entropy_with_logits(L_agent[te][valid[te]], y[te][valid[te]]).item()
    out["ceiling_agent_own_policy_loglik_per_trial"] = round(ceiling, 4)
    specs = {"bayesian": (bayes_model, [2.0, 2.0, -3.0, 8.0, 0.0, 0.0],
                          lambda p: {"evidence_rewarded": p[0], "evidence_unrewarded": p[1],
                                     "hazard": float(torch.sigmoid(torch.tensor(p[2])) * 0.5),
                                     "inverse_temperature": p[3], "bias": p[4], "perseveration": p[5]}),
             "model_based_rl_two_values": (mb_model, [0.0, -2.0, 10.0, 0.0, 0.0],
                                           lambda p: {"learning_rate": float(torch.sigmoid(torch.tensor(p[0]))),
                                                      "forgetting": float(torch.sigmoid(torch.tensor(p[1]))),
                                                      "inverse_temperature": p[2], "bias": p[3], "perseveration": p[4]})}
    latent = {}
    for name, (fn, init, show) in specs.items():
        p = fit(fn, init, st[tr], rw[tr], ch[tr], valid[tr])
        with torch.no_grad():
            logit, lat = fn(p, st, rw, ch, valid)
            ll = -torch.nn.functional.binary_cross_entropy_with_logits(logit[te][valid[te]], y[te][valid[te]]).item()
            agree = ((logit[te] > 0) == (L_agent[te] > 0))[valid[te]].float().mean().item()
            corr = float(np.corrcoef(logit[te][valid[te]].numpy(), L_agent[te][valid[te]].numpy())[0, 1])
        out[name] = {"params": {k: round(float(v), 4) for k, v in show(p.tolist()).items()},
                     "test_loglik_per_trial": round(ll, 4), "choice_sign_agreement_with_agent": round(agree, 4),
                     "logit_correlation_with_agent": round(corr, 4)}
        latent[name] = lat.numpy()
    return out, latent


def decode(H, targets, rng):
    """Held-out R^2 of linear decoders from H (n, 48)."""
    n = H.shape[0]
    perm = rng.permutation(n)
    a, b = perm[:n // 2], perm[n // 2:]
    A = np.column_stack([H, np.ones(n)])
    out = {}
    for k, y in targets.items():
        w = np.linalg.lstsq(A[a], y[a], rcond=None)[0]
        p = A[b] @ w
        out[k] = round(float(1 - ((y[b] - p) ** 2).sum() / ((y[b] - y[b].mean()) ** 2).sum()), 4)
    return out


def geometry(Hp, rec, latent, rng, m, step=1):
    h_go = Hp[:, step::3].numpy()                       # state entering GO (step 1: reward seen) or D1 (step 0: not yet)
    lam = observer(rec)
    lam = np.log(np.clip(lam, 1e-6, 1 - 1e-6) / np.clip(1 - lam, 1e-6, 1))
    q = latent["model_based_rl_two_values"]
    mm = m.copy()
    H = h_go[mm]
    qsum, qdiff = (q[..., 0] + q[..., 1])[mm], (q[..., 0] - q[..., 1])[mm]
    lam_m = lam[mm]
    resid = qsum - np.polyval(np.polyfit(lam_m, qsum, 3), lam_m)     # Q(S1) + Q(S2) beyond the belief
    out = decode(H, {"observer belief logit": lam_m, "MB-RL Q(S1) - Q(S2)": qdiff, "MB-RL Q(S1) + Q(S2)": qsum,
                     "MB-RL Q(S1) + Q(S2) beyond the belief": resid}, rng)
    out["sd_Qsum_beyond_belief_over_sd_Qsum"] = round(float(resid.std() / qsum.std()), 3)
    full = np.zeros(m.shape); full[mm] = resid
    return out, full


def go_split(rnn, tcs, tracer, X, Hp, tt, m, rng):
    gru, tc = rnn.layers[0], tcs["candidate"]
    w = tracer.W_choice
    with torch.no_grad():
        x, h = X[:, 1::3], Hp[:, 1::3]                                  # GO input and the state entering GO
        r = torch.sigmoid(gru.input_to_reset(x) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(x) + gru.hidden_to_update(h))
        n = torch.tanh(gru.input_to_new(x) + gru.hidden_to_new(r * h))
        a = tc.act(tc.input_to_features(torch.cat([r * h, x], -1)))
        carried = ((1 - z) * h) @ w
        written = (z * n) @ w
        per_feat = a * torch.einsum("ntu,fu->ntf", z, tc.features_to_outputs.weight.T * w[None, :])   # w . (z d_f) a_f
    carried, written, per_feat = carried.numpy(), written.numpy(), per_feat.numpy()
    L = tt["L"]
    recon = carried + written + float(tracer.b_choice)
    # how much of each part survives into the next trial's logit (h is linear in its writes given the gates)
    with torch.no_grad():
        xa, ha = X, Hp
        za = torch.sigmoid(gru.input_to_update(xa) + gru.hidden_to_update(ha))
        go = np.arange(1, X.shape[1], 3)
        C = (1 - za[:, go[:-1] + 1]) * (1 - za[:, go[:-1] + 2]) * (1 - za[:, go[:-1] + 3])
        w_next = ((C * (z * n)[:, :-1]) @ w).numpy()
        c_next = ((C * ((1 - z) * h)[:, :-1]) @ w).numpy()
    mn = m[:, :-1]
    retention = {"GO write": round(float(np.median(w_next[mn] / written[:, :-1][mn])), 4),
                 "carried part": round(float(np.median(c_next[mn] / carried[:, :-1][mn])), 4),
                 "mean_abs_surviving_into_next_logit": {"GO write": round(float(np.abs(w_next[mn]).mean()), 3),
                                                        "carried part": round(float(np.abs(c_next[mn]).mean()), 3)}}
    out = {"split_check_max_abs": round(float(np.abs(recon - L).max()), 5), "retention_into_next_logit": retention,
           "mean_abs": {"carried": round(float(np.abs(carried[m]).mean()), 3), "written_at_GO": round(float(np.abs(written[m]).mean()), 3)}}
    # how each part depends on the previous outcome beyond the belief (here: beyond L itself)
    T, x = tt["typ"][m], L[m]
    pt = tt["prev_typ"][m]
    for name, part in (("carried", carried[m]), ("written_at_GO", written[m])):
        knots = np.percentile(x, [10, 30, 50, 70, 90])
        B = basis(x, knots)
        beta = np.linalg.lstsq(B, part, rcond=None)[0]
        res = part - B @ beta
        out[f"{name}_beyond_L_by_previous_input_type"] = {TYPES[k]: round(float(res[pt == k].mean()), 3) for k in range(4)}
    # does the GO write (beyond L) explain the bands?
    y = tt["dL"][m]
    base, _ = band_r2(y, x, T, [])
    wq = np.digitize(written[m] - np.polyval(np.polyfit(x, written[m], 3), x),
                     np.percentile(written[m] - np.polyval(np.polyfit(x, written[m], 3), x), [20, 40, 60, 80]))
    r_w, _ = band_r2(y, x, T, [wq])
    r_p, _ = band_r2(y, x, T, [pt])
    r_both, _ = band_r2(y, x, T, [pt, wq])
    _, fit0 = band_r2(y, x, T, [])
    wb = written[m] - np.polyval(np.polyfit(x, written[m], 3), x)
    out["corr_dL_residual_with_GO_write_beyond_L"] = round(float(np.corrcoef(y - fit0, wb)[0, 1]), 3)
    out["bands_explained"] = {"previous input type": round((r_p - base) / (1 - base), 3),
                              "GO write beyond L (quintiles)": round((r_w - base) / (1 - base), 3),
                              "both": round((r_both - base) / (1 - base), 3)}
    # GO features whose write into L depends most on the previous trial's input type (at the same L)
    pf = per_feat[m]
    knots = np.percentile(x, [10, 30, 50, 70, 90])
    B = basis(x, knots)
    rows = []
    for f in range(pf.shape[1]):
        col = pf[:, f]
        if np.abs(col).mean() < 1e-3:
            continue
        res = col - B @ np.linalg.lstsq(B, col, rcond=None)[0]
        means = [float(res[pt == k].mean()) for k in range(4)]
        rows.append({"feature": int(f), "spread_across_previous_types": round(max(means) - min(means), 3),
                     "write_beyond_L_by_previous_type": {TYPES[k]: round(v, 3) for k, v in enumerate(means)},
                     "mean_abs_write": round(float(np.abs(col).mean()), 3)})
    out["go_features_by_previous_type"] = sorted(rows, key=lambda r: -r["spread_across_previous_types"])[:12]
    return out


PUSHES = (("candidate 121 at GO", "n", 121, 1), ("candidate 146 at GO", "n", 146, 1), ("update 55 at GO", "z", 55, 1),
          ("candidate 77 at D1", "n", 77, 0), ("candidate 208 at D1", "n", 208, 0), ("candidate 18 at D1", "n", 18, 0))


def persistence(rnn, tcs, X, Hp, rec, rng, n=600, push=1.0):
    """Push one feature (activation + push) at trial t and measure the logit change at GO of trial t and of t + 1:
    readout-only features move L(t) but not L(t + 1); memory features move both. Real GRU, one feature edited
    along its decoder direction (circuit_checks.run_patched without held errors)."""
    from circuit.rl_task.circuit_checks import run_patched
    ch = rec["choice"]
    picks = []
    while len(picks) < n:
        e, t = int(rng.integers(X.shape[0])), int(rng.integers(20, 80))
        if (ch[e, t:t + 2] > 0).all():
            picks.append((e, t))
    x = torch.stack([X[e, 3 * t:3 * t + 5] for e, t in picks])
    h0 = torch.stack([Hp[e, 3 * t] for e, t in picks])
    tcm = {"r": tcs["reset"], "z": tcs["update"], "n": tcs["candidate"]}
    gru = rnn.layers[0]
    none = {g: torch.zeros(len(picks), 5, tcm[g].features_to_outputs.weight.shape[1], dtype=torch.bool) for g in "rzn"}
    base = run_patched(rnn, tcs, x, h0, none, {g: torch.zeros(len(picks), 5, v.shape[2]) for g, v in none.items()})
    out = {}
    for name, g, f, step in PUSHES:
        # live activation at that step in the unedited run, plus the push
        with torch.no_grad():
            h = h0
            for t in range(step + 1):
                xt = x[:, t]
                r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
                z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
                inp = torch.cat([r * h, xt], -1) if g == "n" else torch.cat([h, xt], -1)
                a = tcm[g].act(tcm[g].input_to_features(inp))[:, f]
                nn_ = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
                h = (1 - z) * h + z * nn_
        mask = {k: v.clone() for k, v in none.items()}
        mask[g][:, step, f] = True
        value = {k: torch.zeros(len(picks), 5, v.shape[2]) for k, v in none.items()}
        value[g][:, step, f] = a + push
        y = run_patched(rnn, tcs, x, h0, mask, value)
        d_t, d_next = (y[:, 1] - base[:, 1]).numpy(), (y[:, 4] - base[:, 4]).numpy()
        out[name] = {"logit_change_trial_t": round(float(np.mean(d_t)), 3), "logit_change_trial_t_plus_1": round(float(np.mean(d_next)), 3),
                     "kept_share_median": round(float(np.median(d_next / np.where(np.abs(d_t) > 1e-3, d_t, np.nan))), 3)
                     if step == 1 else None}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    rnn, tcs, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    tt = trial_table(rec)
    m = mask_for(tt)
    b = observer(rec)
    lam = np.log(np.clip(b, 1e-6, 1) / np.clip(1 - b, 1e-6, 1))
    dlam = np.concatenate([lam[:, 1:] - lam[:, :-1], np.zeros((lam.shape[0], 1))], 1)
    result = {"episodes": args.episodes, "hazard": HAZARD, "n_points": int(m.sum())}
    result["portrait"] = portrait(tt, lam, dlam, m)
    result["models"], latent = models(rec, rng)
    print("models", json.dumps(result["models"]), flush=True)
    result["geometry"] = {}
    result["geometry"]["after D1 (reward seen)"], qres = geometry(Hp, rec, latent, rng, m, 1)
    result["geometry"]["before D1 (reward not yet seen)"], _ = geometry(Hp, rec, latent, rng, m, 0)
    print("geometry", json.dumps(result["geometry"]), flush=True)
    result["bands"] = bands(tt, m, qres)
    print("bands", json.dumps(result["bands"]), flush=True)
    result["go_split"] = go_split(rnn, tcs, tracer, X, Hp, tt, m, rng)
    result["persistence"] = persistence(rnn, tcs, X, Hp, rec, rng)
    print("persistence", json.dumps(result["persistence"]), flush=True)
    print("go_split", json.dumps({k: v for k, v in result["go_split"].items() if k != "go_features_by_previous_type"}), flush=True)
    for r in result["go_split"]["go_features_by_previous_type"][:8]:
        print("  go feature", json.dumps(r), flush=True)
    print("symmetry", json.dumps(result["portrait"]["bayes_symmetry"]), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
