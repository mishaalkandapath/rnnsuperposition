"""Which step-to-step edges turn a stored token's code?

A token's trace in h (original minus a run with the token in slot i swapped,
feedback changed to match) turns by about 180 degrees between its write step
and its output step. At each step t it splits exactly into

    dh_t = (1 - zb) * dh_{t-1}          carried (stays along the old code)
         + zb * dn_t                    written by the candidate
         + dz * (nb - hb_{t-1})         gate term

and the written part splits over candidate features g at step t (Da_g zb d_g,
plus transcoder error). Each Da_g splits exactly (secant / DeepLIFT rescale,
circuit.copy_task.feature_circuits.preact_system) into Da_f of upstream features f times
the mean edge, plus gate, input and bias / error terms. So every feature ->
feature edge f(s) -> g(t) has a written vector  kappa_g Mbar_gf Da_f zb_t d_g.

Every piece is projected on two axes, per sequence:
  e1  the trace as stored at t-1 (dh_{t-1} / |dh_{t-1}|)
  e2  the token's final read direction (W_c[orig] - W_c[swapped], read at
      output step L+i) with its e1 component removed
Moving from +e1 towards e2 and then to -e1 is the turn. "Turning" pieces push
along e2 (turn forward) or against e1 (undo the old code).

    python -m circuit.copy_task.rotation_edges --rnn RNN.ckpt --traces T4.pt \\
        --transcoders reset=R.ckpt update=U.ckpt candidate=C.ckpt \\
        --feature_results features/<run>_autointerp --magnitude_results features/<run>_magnitude --out rotation.json
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import StackDataset

from circuit.copy_task.copy_circuits import gru_trace, step_label, swap_token
from circuit.edge_att import CircuitTracer
from circuit.copy_task.explained_fidelity import load_transcoder
from circuit.copy_task.feature_circuits import preact_system
from circuit.copy_task.token_circuits import label_of, load_label_sets
from training.splice_eval import build_task_rnn

torch.serialization.add_safe_globals([StackDataset])


@torch.no_grad()
def turn_pieces(tracer, run_o, run_s, L, i, orig, new):
    """Per step t in (i, L+i]: e1 / e2 projections of the trace pieces, and of every feature edge's written vector."""
    tau = L + i
    nodes = sorted({(t, f) for run in (run_o, run_s) for t, f in torch.nonzero(run["f_n"][:tau + 1] > 0).tolist()})
    so, ss = (preact_system(tracer, run, nodes, orig[i], tau) for run in (run_o, run_s))
    Mb, dM = (so["M"] + ss["M"]) / 2, so["M"] - ss["M"]
    ab, da, dp = (so["a"] + ss["a"]) / 2, so["a"] - ss["a"], so["p"] - ss["p"]
    ok = dp.abs() > 1e-6
    kappa = torch.where(ok, da / torch.where(ok, dp, torch.ones_like(dp)), torch.zeros_like(dp))
    src = {"gates": dM @ ab, "input": so["u_in"] - ss["u_in"], "bias/error writes": so["u_write"] - ss["u_write"]}
    dec = tracer.hidden_transcoder.features_to_outputs.weight
    steps = torch.tensor([t for t, _ in nodes])
    fidx = torch.tensor([f for _, f in nodes])
    W_c = tracer.W_o - tracer.W_o.mean(0, keepdim=True)
    read = W_c[orig[i]] - W_c[new]
    out = []
    for t in range(i + 1, tau + 1):
        dprev = run_o["h_ts"][t - 1] - run_s["h_ts"][t - 1]
        dcur = run_o["h_ts"][t] - run_s["h_ts"][t]
        zb = (run_o["z_ts"][t] + run_s["z_ts"][t]) / 2
        nb = (run_o["h_new_ts"][t] + run_s["h_new_ts"][t]) / 2
        hb = (run_o["h_ts"][t - 1] + run_s["h_ts"][t - 1]) / 2
        dz, dn = run_o["z_ts"][t] - run_s["z_ts"][t], run_o["h_new_ts"][t] - run_s["h_new_ts"][t]
        e1 = dprev / dprev.norm().clamp_min(1e-8)
        e2 = read - (read @ e1) * e1
        e2 = e2 / e2.norm().clamp_min(1e-8)
        proj = lambda v: (float(v @ e1), float(v @ e2))
        gi = torch.nonzero(steps == t).flatten()                         # destination features at step t
        cols = zb[None, :] * dec[:, fidx[gi]].T                          # (n_t, H): z_b * d_g
        e1g, e2g = cols @ e1, cols @ e2                                  # per unit of activation change
        feat_written = da[gi] @ cols
        rec = {"t": t, "step": step_label(t, L),
               "carried": proj((1 - zb) * dprev), "written": proj(zb * dn), "gate": proj(dz * (nb - hb)),
               "written_by_features": proj(feat_written), "transcoder_error": proj(zb * dn - feat_written),
               "trace": proj(dcur), "cos_prev": float(dcur @ dprev / (dcur.norm() * dprev.norm()).clamp_min(1e-8)),
               "readout": float(read @ dcur)}
        # destination features: their written vector, split by where their activation change comes from
        rec["dest"] = {int(fidx[g_]): (float(da[g_] * e1g[k]), float(da[g_] * e2g[k])) for k, g_ in enumerate(gi.tolist())}
        by_source = defaultdict(lambda: np.zeros(2))
        edges = {}
        for k, g_ in enumerate(gi.tolist()):
            coef = kappa[g_]
            for key, v in src.items():
                by_source[key] += float(coef * v[g_]) * np.array([float(e1g[k]), float(e2g[k])])
            contrib = coef * Mb[g_] * da                                 # (n,): activation change of g from each upstream node
            for f_ in torch.nonzero(contrib.abs() > 1e-5).flatten().tolist():
                c = float(contrib[f_])
                s = nodes[f_][0]
                by_source[f"features at {step_label(s, L)}"] += c * np.array([float(e1g[k]), float(e2g[k])])
                edges[(nodes[f_], (t, int(fidx[g_])))] = (c * float(e1g[k]), c * float(e2g[k]))
        rec["by_source"] = {k: tuple(v) for k, v in by_source.items()}
        rec["edges"] = edges
        out.append(rec)
    return out


@torch.no_grad()
def feature_angles(tracer, run_o, run_s, L, i, orig, new):
    """Angle at which each carrier writes the token, in a fixed per-token frame:
    f1 = the token's trace at its write step, f2 = its final read direction made orthogonal
    to f1. Carrier g at step t writes v = zb_t d_g Da_g; returns [(t, g, angle, |v in plane|, |v|)],
    and the trace's own angle per step."""
    dec = tracer.hidden_transcoder.features_to_outputs.weight
    W_c = tracer.W_o - tracer.W_o.mean(0, keepdim=True)
    f1 = run_o["h_ts"][i] - run_s["h_ts"][i]
    f1 = f1 / f1.norm().clamp_min(1e-8)
    f2 = (W_c[orig[i]] - W_c[new])
    f2 = f2 - (f2 @ f1) * f1
    f2 = f2 / f2.norm().clamp_min(1e-8)
    out, trace = [], []
    for t in range(i, L + i + 1):
        da = run_o["f_n"][t] - run_s["f_n"][t]
        zb = (run_o["z_ts"][t] + run_s["z_ts"][t]) / 2
        g = torch.nonzero(da.abs() > 1e-4).flatten()
        v = (zb[None, :] * dec[:, g].T) * da[g][:, None]                 # (n, H)
        a, b = v @ f1, v @ f2
        for k, gg in enumerate(g.tolist()):
            out.append((t, gg, float(torch.rad2deg(torch.atan2(b[k], a[k]))), float(torch.hypot(a[k], b[k])), float(v[k].norm())))
        d = run_o["h_ts"][t] - run_s["h_ts"][t]
        trace.append((t, float(torch.rad2deg(torch.atan2(d @ f2, d @ f1))), float(torch.hypot(d @ f1, d @ f2) / d.norm())))
    return out, trace


def circ_mean(angles, weights):
    """Weighted circular mean (degrees) and resultant length R in [0, 1] (1 = all at one angle)."""
    a = np.deg2rad(np.asarray(angles))
    w = np.asarray(weights)
    c, s = (w * np.cos(a)).sum(), (w * np.sin(a)).sum()
    return float(np.rad2deg(np.arctan2(s, c))), float(np.hypot(c, s) / max(w.sum(), 1e-12))


def phases(args, tracer, rnn, data, rows, L, tokens, rng, labels):
    per_step = defaultdict(lambda: ([], []))                             # (slot, steps since write) -> angles, weights
    per_feat = defaultdict(lambda: ([], [], []))                         # feature -> angles, weights, steps since write
    in_plane = defaultdict(lambda: [0.0, 0.0])
    trace = defaultdict(list)
    for row in rows:
        x = data["inputs"][row].float()
        orig = tokens[row].tolist()
        run_o = tracer.run_forward_pass(gru_trace(rnn, x))
        for i in range(L):
            new = int(rng.choice([v for v in range(30) if v not in orig]))
            run_s = tracer.run_forward_pass(gru_trace(rnn, swap_token(x, L, i, new)))
            fa, tr = feature_angles(tracer, run_o, run_s, L, i, orig, new)
            for t, g, ang, w, norm in fa:
                per_step[(i, t - i)][0].append(ang); per_step[(i, t - i)][1].append(w)
                per_feat[g][0].append(ang); per_feat[g][1].append(w); per_feat[g][2].append(t - i)
                in_plane[g][0] += w; in_plane[g][1] += norm
            for t, ang, frac in tr:
                trace[(i, t - i)].append((ang, frac))
    steps_out = []
    for (i, k) in sorted(per_step):
        angs, ws = per_step[(i, k)]
        hist, edges_ = np.histogram(angs, bins=12, range=(-180, 180), weights=ws)
        m, R = circ_mean(angs, ws)
        ta = [a for a, _ in trace[(i, k)]]
        tm, tR = circ_mean(ta, np.ones(len(ta)))
        steps_out.append({"slot": i, "steps_since_write": k, "step": step_label(i + k, L), "carriers_mean_angle": m,
                          "carriers_R": R, "trace_angle": tm, "trace_R": tR,
                          "trace_share_in_plane": float(np.mean([f for _, f in trace[(i, k)]])),
                          "angle_histogram": hist.tolist(), "bins": edges_.tolist()})
    feats_out = []
    for g, (angs, ws, ks) in per_feat.items():
        if sum(ws) < 0.05 * len(rows):                                   # skip features that barely carry anything
            continue
        m, R = circ_mean(angs, ws)
        by_k = {}
        for k in sorted(set(ks)):
            sel = [j for j, kk in enumerate(ks) if kk == k]
            if sum(ws[j] for j in sel) > 0.02 * len(rows):
                by_k[k] = (*circ_mean([angs[j] for j in sel], [ws[j] for j in sel]),
                           sum(ws[j] for j in sel) / (len(rows) * L))      # mean weight per (sequence, slot)
        feats_out.append({"feature": g, "mean_angle": m, "R": R, "weight": float(sum(ws) / len(rows)),
                          "group": "keeper" if abs(m) < 90 else "flipper",
                          "share_of_write_in_plane": in_plane[g][0] / max(in_plane[g][1], 1e-12),
                          "angle_by_steps_since_write": {int(k): {"angle": v[0], "R": v[1], "weight": v[2]}
                                                         for k, v in by_k.items()},
                          "label": label_of(labels, "hidden", g)["label"]})
    feats_out.sort(key=lambda d: -d["weight"])
    return {"steps": steps_out, "features": feats_out}


@torch.no_grad()
def sign_test(tracer, rnn, data, rows, L, tokens, rng, groups):
    """Input sign vs memory sign, per candidate feature.

    input part:  how token v moves the feature as the current input, from the encoder:
                 s_in = W_x[f, orig] - W_x[f, swapped]
    memory part: how it moves the feature once stored, at steps after the write (up to the output
                 step, before the token is fed back): dm = W_h[f] . (r_o h_o - r_s h_s)
    Correlation across (sequence, slot) at each number of steps since the write. Prediction: < 0 for
    flippers (stored code opposite to the input code), > 0 for keepers.
    """
    W_x, W_h = tracer.W_n_x, tracer.W_n_h                                # (F, X), (F, H)
    F = W_x.shape[0]
    acc = defaultdict(lambda: np.zeros((4, F)))                          # k -> sums of s*m, s*s, m*m, n
    for row in rows:
        x = data["inputs"][row].float()
        orig = tokens[row].tolist()
        tr_o = gru_trace(rnn, x)
        for i in range(L):
            new = int(rng.choice([v for v in range(30) if v not in orig]))
            tr_s = gru_trace(rnn, swap_token(x, L, i, new))
            s_in = (W_x[:, orig[i]] - W_x[:, new]).numpy()
            for t in range(i + 1, L + i + 1):
                dm = (W_h @ (tr_o["r_ts"][t] * tr_o["h_prevs"][t] - tr_s["r_ts"][t] * tr_s["h_prevs"][t])).numpy()
                a = acc[t - i]
                a[0] += s_in * dm; a[1] += s_in ** 2; a[2] += dm ** 2; a[3] += 1
    out = {}
    for k, a in sorted(acc.items()):
        corr = a[0] / np.sqrt(a[1] * a[2] + 1e-12)                       # uncentred: signs relative to zero change
        out[k] = corr
    res = {"per_feature": {}, "by_group": {}}
    for f in groups:
        res["per_feature"][f] = {int(k): float(v[f]) for k, v in out.items()}
    for grp in ("keeper", "flipper"):
        fs = [f for f, g_ in groups.items() if g_ == grp]
        res["by_group"][grp] = {int(k): {"median": float(np.median(v[fs])), "share_negative": float(np.mean(v[fs] < 0))}
                                for k, v in out.items()}
    return res


@torch.no_grad()
def reversal_test(tracer, rnn, data, rows, L, tokens, rng, groups, ks=(1, 2, 3)):
    """What makes the flippers' memory code reverse?

    A flipper f reads memory as m_t(f) = W_h[f] . (r_t * h_{t-1}). Between the original and the
    token-swapped run, d(r_t * h_{t-1}) splits exactly into
        rb_t * (1 - zb_{t-1}) * dh_{t-2}        carried (older writes)
        rb_t * zb_{t-1} * Da_g d_g              written at t-1 by candidate feature g
        rb_t * zb_{t-1} * De_{t-1}              transcoder error at t-1
        rb_t * dz_{t-1} * (nb - hb_{t-2})       update-gate change at t-1
        dr_t * hb_{t-1}                         reset-gate change at t
    The flippers' input-vs-memory correlation (sum s*m / sqrt(sum s^2 sum m^2), s = W_x[f, orig] -
    W_x[f, swapped]) splits the same way over these pieces; returned per steps since write k
    (median over flippers), with the written piece also split per source feature and by group.
    """
    W_x, W_h = tracer.W_n_x, tracer.W_n_h
    dec = tracer.hidden_transcoder.features_to_outputs.weight
    fl = torch.tensor(sorted(f for f, g_ in groups.items() if g_ == "flipper"))
    nF = tracer.hidden_transcoder.n_feats
    keys = ["carried (older writes)", "written at t-1 (features)", "transcoder error at t-1", "update-gate change at t-1",
            "reset-gate change at t"]
    num = {k: {key: torch.zeros(len(fl)) for key in keys} for k in ks}
    src = {k: torch.zeros(nF, len(fl)) for k in ks}                     # per source feature g: sum s * c
    ss = {k: torch.zeros(len(fl)) for k in ks}
    mm = {k: torch.zeros(len(fl)) for k in ks}
    for row in rows:
        x = data["inputs"][row].float()
        orig = tokens[row].tolist()
        ro = tracer.run_forward_pass(gru_trace(rnn, x))
        for i in range(L):
            new = int(rng.choice([v for v in range(30) if v not in orig]))
            rs = tracer.run_forward_pass(gru_trace(rnn, swap_token(x, L, i, new)))
            s = W_x[fl, orig[i]] - W_x[fl, new]
            for k in ks:
                t = i + k
                if t > L + i:
                    continue
                u = t - 1                                                # the step whose write is read
                rb = (ro["r_ts"][t] + rs["r_ts"][t]) / 2
                zb, dz = (ro["z_ts"][u] + rs["z_ts"][u]) / 2, ro["z_ts"][u] - rs["z_ts"][u]
                hprev_o, hprev_s = ro["h_prevs"][u], rs["h_prevs"][u]     # h_{t-2}
                nb = (ro["h_new_ts"][u] + rs["h_new_ts"][u]) / 2
                pieces = {"carried (older writes)": rb * (1 - zb) * (hprev_o - hprev_s),
                          "transcoder error at t-1": rb * zb * (ro["e_n"][u] - rs["e_n"][u]),
                          "update-gate change at t-1": rb * dz * (nb - (hprev_o + hprev_s) / 2),
                          "reset-gate change at t": (ro["r_ts"][t] - rs["r_ts"][t]) * (ro["h_prevs"][t] + rs["h_prevs"][t]) / 2}
                da = ro["f_n"][u] - rs["f_n"][u]
                g = torch.nonzero(da.abs() > 1e-5).flatten()
                per_g = (W_h[fl] @ ((rb * zb)[:, None] * dec[:, g])) * da[g][None, :]   # (|fl|, n_g)
                pieces_c = {key: W_h[fl] @ v for key, v in pieces.items()}
                pieces_c["written at t-1 (features)"] = per_g.sum(1)
                m = sum(pieces_c.values())
                for key, c in pieces_c.items():
                    num[k][key] += s * c
                src[k][g] += (s[:, None] * per_g).T
                ss[k] += s ** 2
                mm[k] += m ** 2
    out = {}
    for k in ks:
        D = torch.sqrt(ss[k] * mm[k]).clamp_min(1e-12)
        contrib = {key: (v / D) for key, v in num[k].items()}
        per_src = (src[k] / D[None, :]).mean(1)                          # mean over flippers
        top = torch.argsort(per_src)[:10].tolist()                       # most negative = drive the reversal
        out[k] = {"total_median": float(sum(contrib.values()).median()),
                  "pieces_median": {key: float(v.median()) for key, v in contrib.items()},
                  "written_by_group": {grp: float(per_src[[f for f, g_ in groups.items() if g_ == grp]].sum())
                                       for grp in ("keeper", "flipper")},
                  "written_by_unlisted": float(per_src[[f for f in range(nF) if f not in groups]].sum()),
                  "top_reversing_sources": [(f, float(per_src[f]), groups.get(f, "-")) for f in top]}
    return out


@torch.no_grad()
def reversal_interventions(rnn, tcs, data, L, groups, weights, n, seed, n_keepers=10):
    """Causal tests of the reversal (real GRU, candidate features mean-ablated at one step):
      A  keepers removed one step after the write (step i+1): the flippers should not reverse at +2
      B  flippers removed at the write step (no input drive): they should reverse already at +1
    plus random same-size controls. For each slot i, the original and a token-swapped run (feedback
    changed to match) get the same edit; reported: the flippers' input-vs-memory correlation at
    +1..+3 (median over flippers) and the accuracy of output i and of the other outputs."""
    from circuit.copy_task.circuit_tests import clean_acts, feature_hook, run
    tc = tcs["candidate"]
    W_x, W_h = tc.input_to_features.weight[:, 128:], tc.input_to_features.weight[:, :128]
    tokens = data["inputs"][:, :L, :-1].argmax(-1)
    distinct = np.flatnonzero([len(set(r)) == L for r in tokens.tolist()])
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    rows = torch.from_numpy(np.sort(rng.choice(distinct, size=min(n, len(distinct)), replace=False)))
    x, orig = data["inputs"][rows].float(), tokens[rows]
    N = len(x)
    fl = sorted(f for f, g_ in groups.items() if g_ == "flipper")
    keepers = sorted((f for f, g_ in groups.items() if g_ == "keeper"), key=lambda f: -weights[f])[:n_keepers]
    clean = clean_acts(rnn, x, L, tcs)
    step_means = clean["candidate"].mean(0)
    active = torch.nonzero((clean["candidate"] > 0).float().mean((0, 1)) > 0.02).flatten().tolist()
    pool = [f for f in active if f not in groups]
    present = torch.zeros(N, 30, dtype=torch.bool).scatter_(1, orig, True)
    flt = torch.tensor(fl)

    def recorded(xx, hook):
        rec = {"r": [], "h": []}

        def hk(t, h, xt, r, z, nn):
            r, z, nn = hook(t, h, xt, r, z, nn) if hook else (r, z, nn)
            rec["r"].append(r); rec["h"].append(h)
            return r, z, nn
        logits = run(rnn, xx, L, hk)
        return logits, torch.stack(rec["r"], 1), torch.stack(rec["h"], 1)

    out = {}
    for i in range(L):
        new = torch.multinomial((~present).float(), 1, generator=g)[:, 0]
        xs = x.clone()
        for tt in [i] + ([L + 1 + i] if i + 1 < L else []):
            xs[:, tt] = 0
            xs[torch.arange(N), tt, new] = 1
        s = (W_x[flt][:, orig[:, i]] - W_x[flt][:, new]).T                 # (N, |fl|)
        rand = [int(v) for v in rng.choice(pool, n_keepers, replace=False)]
        conds = {"none": None,
                 f"A: keepers {keepers} removed at +1": feature_hook(tcs, "candidate", keepers, lambda t: step_means[t].expand(N, -1), {i + 1}),
                 "A control: 10 random others at +1": feature_hook(tcs, "candidate", rand, lambda t: step_means[t].expand(N, -1), {i + 1}),
                 "B: flippers removed at the write step": feature_hook(tcs, "candidate", fl, lambda t: step_means[t].expand(N, -1), {i}),
                 "B control: 34 random others at the write step": feature_hook(
                     tcs, "candidate", [int(v) for v in rng.choice(pool, len(fl), replace=False)], lambda t: step_means[t].expand(N, -1), {i})}
        for name, hook in conds.items():
            lo, ro_, ho = recorded(x, hook)
            ls, rs_, hs_ = recorded(xs, hook)
            corr = {}
            for k in (1, 2, 3):
                t = i + k
                if t > L + i:
                    continue
                m = (ro_[:, t] * ho[:, t] - rs_[:, t] * hs_[:, t]) @ W_h[flt].T      # (N, |fl|)
                c = (s * m).sum(0) / torch.sqrt((s ** 2).sum(0) * (m ** 2).sum(0)).clamp_min(1e-12)
                corr[k] = float(c.median())
            pred = lo.argmax(-1)
            others = [j for j in range(L) if j != i]
            out.setdefault(name.split(":")[0], {})[i] = {
                "condition": name, "flipper_corr": corr,
                "acc_own": float((pred[:, i] == orig[:, i]).float().mean()),
                "acc_others": float((pred[:, others] == orig[:, others]).float().mean())}
            r_ = out[name.split(":")[0]][i]
            print(f"slot {i} | {name[:52]:>52}: flipper corr " + " ".join(f"+{k}:{v:+.2f}" for k, v in corr.items())
                  + f" | output {i} acc {r_['acc_own']:.2f}, others {r_['acc_others']:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--transcoders", nargs=3, required=True, metavar="GATE=PATH")
    ap.add_argument("--feature_results", default=None)
    ap.add_argument("--magnitude_results", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--phases", action="store_true", help="angle at which each carrier writes the token, instead of edges")
    ap.add_argument("--reversal_tests", default=None, metavar="PHASES_JSON",
                    help="run the causal reversal tests, with keeper / flipper groups from a --phases output")
    ap.add_argument("--n_tests", type=int, default=3000, help="sequences for the reversal tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    paths = dict(item.split("=", 1) for item in args.transcoders)
    rnn = build_task_rnn("copy", 31, 128)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    tc, upd, rst = (load_transcoder(paths[k]) for k in ("candidate", "update", "reset"))
    tracer = CircuitTracer(rnn, upd, tc, device="cpu", reset_transcoder=rst)
    labels = load_label_sets(args.feature_results, args.magnitude_results)
    data = torch.load(args.traces, map_location="cpu", mmap=True).datasets
    L = data["outputs"].shape[1]
    tokens = data["inputs"][:, :L, :-1].argmax(-1)
    rng = np.random.default_rng(args.seed)
    distinct = np.flatnonzero([len(set(r)) == L for r in tokens.tolist()])
    rows = np.sort(rng.choice(distinct, size=min(args.n, len(distinct)), replace=False))
    n = len(rows)
    if args.reversal_tests:
        ph = json.load(open(args.reversal_tests))
        groups = {f["feature"]: f["group"] for f in ph["features"]}
        weights = {f["feature"]: f["weight"] for f in ph["features"]}
        tcs = {"candidate": tc, "update": upd, "reset": rst}
        res = reversal_interventions(rnn, tcs, data, L, groups, weights, args.n_tests, args.seed)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"args": vars(args), "L": L, "tests": res}, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
        return
    if args.phases:
        res = phases(args, tracer, rnn, data, rows, L, tokens, rng, labels)
        for s in res["steps"]:
            print(f"slot {s['slot']} {s['step']:>4} (+{s['steps_since_write']}): trace at {s['trace_angle']:+5.0f} deg "
                  f"(in plane {s['trace_share_in_plane']:.2f}) | carriers mean {s['carriers_mean_angle']:+5.0f} deg, R {s['carriers_R']:.2f}")
        print("features (largest weight first): mean angle, consistency R, share of write in the plane, angle by steps since write")
        for f in res["features"][:30]:
            by = " ".join(f"+{k}:{v['angle']:+.0f}" for k, v in f["angle_by_steps_since_write"].items())
            print(f"  {f['feature']:>3} {f['mean_angle']:+5.0f} deg R {f['R']:.2f} plane {f['share_of_write_in_plane']:.2f} | {by} | {f['label'][:45]}")
        groups = {f["feature"]: f["group"] for f in res["features"]}
        res["sign_test"] = sign_test(tracer, rnn, data, rows, L, tokens, rng, groups)
        print("sign test: correlation of input part (encoder weights) and memory part (stored token), by steps since write")
        for grp, by in res["sign_test"]["by_group"].items():
            print(f"  {grp:>8}s ({sum(g_ == grp for g_ in groups.values())}): " + "  ".join(
                f"+{k}: median {v['median']:+.2f} (negative {v['share_negative']:.0%})" for k, v in by.items()))
        for f in res["features"][:16]:
            pf = res["sign_test"]["per_feature"][f["feature"]]
            print(f"  {f['feature']:>3} ({f['group']}): " + " ".join(f"+{k}:{v:+.2f}" for k, v in pf.items()))
        res["reversal"] = reversal_test(tracer, rnn, data, rows, L, tokens, rng, groups)
        print("reversal: flippers' input-vs-memory correlation split by what in h_{t-1} they read (median over flippers)")
        for k, r_ in res["reversal"].items():
            print(f"  +{k}: total {r_['total_median']:+.2f} | " + "; ".join(f"{key} {v:+.2f}" for key, v in r_["pieces_median"].items()))
            print(f"       written at t-1 by group (mean over flippers, summed over sources): keepers {r_['written_by_group']['keeper']:+.2f}, "
                  f"flippers {r_['written_by_group']['flipper']:+.2f}, other features {r_['written_by_unlisted']:+.2f}")
            print("       top reversing sources: " + ", ".join(f"{f} ({grp}) {v:+.3f}" for f, v, grp in r_["top_reversing_sources"][:8]))
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"args": vars(args), "L": L, **res}, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
        return

    agg = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))          # (slot, t) -> key -> mean (e1, e2)
    dest = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))
    edges = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))
    scal = defaultdict(lambda: defaultdict(float))
    start = time.time()
    for row in rows:
        x = data["inputs"][row].float()
        orig = tokens[row].tolist()
        run_o = tracer.run_forward_pass(gru_trace(rnn, x))
        for i in range(L):
            new = int(rng.choice([v for v in range(30) if v not in orig]))
            run_s = tracer.run_forward_pass(gru_trace(rnn, swap_token(x, L, i, new)))
            for rec in turn_pieces(tracer, run_o, run_s, L, i, orig, new):
                key = (i, rec["t"])
                for k in ("carried", "written", "gate", "written_by_features", "transcoder_error", "trace"):
                    agg[key][k] += np.array(rec[k]) / n
                for k, v in rec["by_source"].items():
                    agg[key]["written via " + k] += np.array(v) / n
                for f, v in rec["dest"].items():
                    dest[key][f] += np.array(v) / n
                for e, v in rec["edges"].items():
                    edges[key][e] += np.array(v) / n
                scal[key]["cos_prev"] += rec["cos_prev"] / n
                scal[key]["readout"] += rec["readout"] / n
    print(f"{n} sequences x {L} slots, {time.time() - start:.0f}s")

    def lab(f):
        return label_of(labels, "hidden", f)["label"]

    out = []
    for (i, t) in sorted(agg):
        a = agg[(i, t)]
        top_turn = sorted(dest[(i, t)].items(), key=lambda kv: -kv[1][1])[:args.top]
        top_undo = sorted(dest[(i, t)].items(), key=lambda kv: kv[1][0])[:args.top]
        top_edges = sorted(edges[(i, t)].items(), key=lambda kv: -kv[1][1])[:args.top]
        e2s = np.array(sorted((v[1] for v in edges[(i, t)].values()), reverse=True))
        pos = e2s[e2s > 0]
        k50 = int(np.searchsorted(np.cumsum(pos), 0.5 * pos.sum()) + 1) if len(pos) else 0
        self_e2 = sum(v[1] for (src, dst), v in edges[(i, t)].items() if src[1] == dst[1] and src[0] == t - 1)
        rec_conc = {"edges_total_e2": float(e2s.sum()), "edges_for_half_of_positive_e2": k50, "n_edges": len(e2s),
                    "self_edge_e2": float(self_e2), "self_edge_share": float(self_e2 / e2s.sum()) if e2s.sum() else None}
        rec = {"slot": i, "step": step_label(t, L), "cos_prev": scal[(i, t)]["cos_prev"], "readout": scal[(i, t)]["readout"],
               "concentration": rec_conc,
               "pieces": {k: {"e1": float(v[0]), "e2": float(v[1])} for k, v in a.items()},
               "turning_features": [{"feature": f, "e1": float(v[0]), "e2": float(v[1]), "label": lab(f)} for f, v in top_turn],
               "undoing_features": [{"feature": f, "e1": float(v[0]), "e2": float(v[1]), "label": lab(f)} for f, v in top_undo],
               "turning_edges": [{"src": src[1], "src_step": step_label(src[0], L), "dst": dst[1], "dst_step": step_label(dst[0], L),
                                  "e1": float(v[0]), "e2": float(v[1]), "src_label": lab(src[1]), "dst_label": lab(dst[1])}
                                 for (src, dst), v in top_edges]}
        out.append(rec)
        p = rec["pieces"]
        print(f"slot {i} {step_label(t, L):>4}: cos(prev) {rec['cos_prev']:+.2f} readout {rec['readout']:+.1f} | (e1, e2): "
              f"carried ({p['carried']['e1']:+.2f}, {p['carried']['e2']:+.2f}) written ({p['written']['e1']:+.2f}, {p['written']['e2']:+.2f}) "
              f"gate ({p['gate']['e1']:+.2f}, {p['gate']['e2']:+.2f}) error ({p['transcoder_error']['e1']:+.2f}, {p['transcoder_error']['e2']:+.2f})")
        via = sorted(((k, v) for k, v in p.items() if k.startswith("written via")), key=lambda kv: -abs(kv[1]["e2"]))
        print("      written via: " + "; ".join(f"{k[12:]} ({v['e1']:+.2f}, {v['e2']:+.2f})" for k, v in via[:5]))
        print("      turning features (e2): " + "; ".join(f"{d['feature']} {d['e2']:+.2f} [{d['label'][:30]}]" for d in rec["turning_features"][:4]))
        print("      turning edges (e2): " + "; ".join(f"{e['src']}@{e['src_step']}->{e['dst']} {e['e2']:+.2f}" for e in rec["turning_edges"][:5]))
        c = rec["concentration"]
        print(f"      edges: total e2 {c['edges_total_e2']:+.2f} over {c['n_edges']} edges; {c['edges_for_half_of_positive_e2']} edges "
              f"carry half; self-edges (same feature, t-1 -> t) {c['self_edge_share'] if c['self_edge_share'] is None else round(c['self_edge_share'], 2)}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "L": L, "steps": out}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
