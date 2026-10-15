"""Conditional usefulness and per-component clamp tests (real GRU), for transcoder features and GRU units.

RL version of circuit/copy_task/conditional_analyses.py and clamp_tests.py (docs/autointerp_guide.md section 5).
Windows are held-out (odd-episode) windows of the step table, so every step has its task variables and every
component's activation; edits are made inside the REAL GRU:
  feature f of a gate:  gate value += (target - a_f) d_f   (along its decoder direction; the transcoder error is
                        kept, and a reset edit is carried into the candidate)
  GRU unit u of a gate: gate value[u] = target

importance  every firing feature and every unit ablated alone (features: per-(phase, episode-decile) mean and zero;
            units: per-(phase, episode-decile) mean). A component used in one situation is diluted when its effect
            is averaged over every window, so the drop in GO-choice agreement, switch agreement and the value error
            is reported
              overall     over all held-out windows
              active      over the windows where it takes part (features: active at some step; units: its value at
                          some step is in the top 10% of |value - timing|)
              worst       for the previous-trial pattern (choice x state reached x reward) where it hurts most
                          (GO choices after that pattern; value at those trials' steps)
clamp       does each explained component do what its explanation implies? The explanation's prediction p (short
            form or predicate, as in explanation_substitution.py) marks where the component should be high (H: top
            15% of p within its phases, or the predicate true) and low (L: bottom 15%, or false). The model's
            natural behaviour differs between H and L (choice logit at the next GO, value at the step):
            delta_nat. Clamp ON: at one L step per window, set the component to its mean activation in H ->
            predicted: behaviour moves the way delta_nat says. Clamp OFF: at one H step, set it to its mean in L
            -> predicted: the opposite. Control: random other explained components of the same source and gate,
            clamped the same way at the same steps. Pass = both directions agree with delta_nat and the effect
            projected on that direction beats every control.

    python -m circuit.rl_task.conditional_tests --rnn RNN.ckpt --traces rl_features/rl_trace_sequences \\
        --transcoders reset=R.ckpt update=U.ckpt candidate=C.ckpt --sections importance clamp \\
        --out rl_features/rl_fidelity/conditional_tests.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from circuit.copy_task.explained_fidelity import load_transcoder
from circuit.rl_task.autointerp_magnitude import PATTERNS, _pattern
from circuit.rl_task.explanation_substitution import FEATURE_GATE, UNIT_RANGE, eval_windows, predictions
from circuit.rl_task.ranked_fidelity import MAG_UNIT_GATE, verdict_tiers
from training.splice_eval import build_task_rnn

UNIT_KEY = {"reset": "reset", "update": "update", "candidate": "candidate"}


@torch.no_grad()
def run_edit(rnn, x, h0, edit=None):
    """Readouts (n, T, 4) of the real GRU; edit(t, h, xt, r, z, n) -> (r, z, n)."""
    gru, readout = rnn.layers[0], rnn.layers[-1]
    h, out = h0, []
    for t in range(x.shape[1]):
        xt = x[:, t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        if edit is not None:
            r0, n0 = r, n
            r, z, n = edit(t, h, xt, r, z, n)
            if r is not r0:                              # a reset edit must reach the candidate
                n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h)) + (n - n0)
        h = (1 - z) * h + z * n
        out.append(readout(h))
    return torch.stack(out, 1)


def make_edit(gate, source, comp, target, mask, tc=None):
    """Edit setting component comp to target[:, t] where mask[:, t] (tensors (n, T))."""
    if source == "features":
        d = tc.features_to_outputs.weight[:, comp]
        w_in, b_in = tc.input_to_features.weight[comp], tc.input_to_features.bias[comp]

    def edit(t, h, xt, r, z, n):
        m = mask[:, t]
        if not m.any():
            return r, z, n
        if source == "features":
            inp = torch.cat([r * h if gate == "candidate" else h, xt], -1)
            a = tc.act(tc.input_to_features(inp))[:, comp]
            delta = ((target[:, t] - a) * m)[:, None] * d
            if gate == "candidate":
                return r, z, n + delta
            if gate == "update":
                return r, (z + delta).clamp(0, 1), n
            return (r + delta).clamp(0, 1), z, n
        val = {"reset": r, "update": z, "candidate": n}[gate].clone()
        val[:, comp] = torch.where(m, target[:, t], val[:, comp])
        return (val, z, n) if gate == "reset" else (r, val, n) if gate == "update" else (r, z, val)
    return edit


def run_groups(rnn, groups, edit_for_group, chunk=4096):
    outs = []
    for gi, g in enumerate(groups):
        parts = []
        for s in range(0, len(g["inputs"]), chunk):
            parts.append(run_edit(rnn, g["inputs"][s:s + chunk], g["h0"][s:s + chunk], edit_for_group(gi, s, chunk)))
        outs.append(torch.cat(parts))
    return outs


# ----------------------------------------------------------------------------- per-step context

def context(groups, ip, refs):
    """Per group: phase, previous-trial pattern, GO / switch masks (n, T), and the reference readouts."""
    c = ip.cols
    last_all = _pattern(np.asarray(c["prev_action"]), np.asarray(c["prev_state"]), np.asarray(c["prev_reward"]))
    ctx = []
    for g, ref in zip(groups, refs):
        rows = g["rows"]
        phase = np.asarray(c["phase"])[rows]
        go = phase == 1
        choice = ref[..., 1:3].argmax(-1).numpy()                   # 0 a1, 1 a2 (meaningful at GO)
        sw = np.zeros_like(go)
        go_idx = np.nonzero(go[0])[0]
        prev_choice = np.full(choice.shape, -1)
        prev_choice[:, go_idx[1:]] = choice[:, go_idx[:-1]]
        sw = go & (prev_choice >= 0) & (choice != prev_choice)
        ctx.append({"phase": phase, "go": go, "switch": sw, "last": last_all[rows], "choice": choice,
                    "logit": (ref[..., 1] - ref[..., 2]).numpy(), "value": ref[..., 3].numpy()})
    return ctx


def behaviour(outs, refs, ctx, value_std):
    """GO agreement, switch agreement, value nMAE, and per-pattern GO agreement / value error."""
    agree, n_go, sw_a, n_sw, v_err, n_v = 0, 0, 0, 0, 0.0, 0
    per = {k: [0, 0, 0.0, 0] for k in range(len(PATTERNS))}         # go agree, go n, value err, value n
    for out, ref, cx in zip(outs, refs, ctx):
        ok = np.isfinite(out.numpy()).all(-1) & (np.abs(out.numpy()) < 1e3).all(-1)
        same = (out[..., 1:3].argmax(-1).numpy() == cx["choice"]) & ok
        verr = np.where(ok, np.abs(out[..., 3].numpy() - cx["value"]), 0.0)
        agree += int(same[cx["go"]].sum()); n_go += int(cx["go"].sum())
        sw_a += int(same[cx["switch"]].sum()); n_sw += int(cx["switch"].sum())
        v_err += float(verr.sum()); n_v += int(ok.sum())
        for k in per:
            m = cx["last"] == k
            per[k][0] += int(same[m & cx["go"]].sum()); per[k][1] += int((m & cx["go"]).sum())
            per[k][2] += float(verr[m].sum()); per[k][3] += int(m.sum())
    return {"go": agree / max(n_go, 1), "switch": sw_a / max(n_sw, 1), "value": v_err / max(n_v, 1) / value_std,
            "per_pattern_go": {PATTERNS[k]: (v[0] / v[1] if v[1] >= 200 else None) for k, v in per.items()},
            "per_pattern_value": {PATTERNS[k]: (v[2] / v[3] / value_std if v[3] >= 600 else None) for k, v in per.items()}}


def restrict(outs, refs, ctx, keep):
    """Subset each group to the windows in keep (list of bool arrays per group)."""
    return ([o[k] for o, k in zip(outs, keep)], [r[k] for r, k in zip(refs, keep)],
            [{n: v[k] for n, v in c.items()} for c, k in zip(ctx, keep)])


# ----------------------------------------------------------------------------- component activations

def activations(ip, source, gate, comp, rows_all):
    if source == "features":
        return ip.activations(FEATURE_GATE[gate], comp).astype(np.float64)
    return np.asarray(np.load(os.path.join("rl_features/rl_unit_values", f"{UNIT_KEY[gate]}.npy"),
                              mmap_mode="r")[:, comp], dtype=np.float64)


def timing_mean(ip, act):
    c = ip.cols
    code = np.asarray(c["phase"]).astype(np.int64) * 10 + np.clip(np.asarray(c["ep_trial"]) // 10, 0, 9)
    dev = ~ip.test
    m = np.bincount(code[dev], weights=act[dev], minlength=30) / np.maximum(np.bincount(code[dev], minlength=30), 1)
    return m[code]


# ----------------------------------------------------------------------------- importance

def importance(rnn, tcs, groups, refs, ctx, ip, gates, sources, value_std, log):
    out = []
    base = behaviour(refs, refs, ctx, value_std)
    counts = json.load(open("rl_features/rl_chosen/chosen_rl_v1_r128_store/store.json"))["counts"]
    for gate in gates:
        for source in sources:
            tc = tcs[gate] if source == "features" else None
            comps = sorted(int(f) for f in counts[FEATURE_GATE[gate]]) if source == "features" else list(range(48))
            modes = ("mean", "zero") if source == "features" else ("mean",)
            t0 = time.time()
            for comp in comps:
                act = activations(ip, source, gate, comp, None)
                tm = timing_mean(ip, act)
                resid = np.abs(act - tm)
                thr = np.quantile(resid[ip.test], 0.9) if source == "units" else 0.0
                rec = {"gate": gate, "source": source, "id": comp}
                takes = [((act[g["rows"]] > 0).any(1) if source == "features" else (resid[g["rows"]] > thr).any(1))
                         for g in groups]
                rec["share_windows_taking_part"] = float(np.mean(np.concatenate(takes)))
                for mode in modes:
                    targets = [torch.from_numpy((tm[g["rows"]] if mode == "mean" else np.zeros(g["rows"].shape))).float()
                               for g in groups]
                    edit_for = lambda gi, s, ch: make_edit(gate, source, comp, targets[gi][s:s + ch],
                                                           torch.ones_like(targets[gi][s:s + ch], dtype=torch.bool), tc)
                    outs = run_groups(rnn, groups, edit_for)
                    allb = behaviour(outs, refs, ctx, value_std)
                    o, r_, c_ = restrict(outs, refs, ctx, takes)
                    actb = behaviour(o, r_, c_, value_std) if sum(int(k.sum()) for k in takes) else None
                    drops = {p: (base["per_pattern_go"][p] - v) for p, v in allb["per_pattern_go"].items()
                             if v is not None and base["per_pattern_go"][p] is not None}
                    vinc = {p: (v - base["per_pattern_value"][p]) for p, v in allb["per_pattern_value"].items()
                            if v is not None and base["per_pattern_value"][p] is not None}
                    worst_p = max(drops, key=drops.get) if drops else None
                    worst_v = max(vinc, key=vinc.get) if vinc else None
                    rec[mode] = {"overall": {k: base[k] - allb[k] if k != "value" else allb[k] - base[k]
                                             for k in ("go", "switch", "value")},
                                 "active": None if actb is None else {
                                     "go": 1 - actb["go"], "switch": 1 - actb["switch"], "value": actb["value"]},
                                 "worst_pattern_go": [worst_p, drops.get(worst_p)],
                                 "worst_pattern_value": [worst_v, vinc.get(worst_v)]}
                out.append(rec)
            log(f"importance {gate} {source}: {len(comps)} components in {time.time() - t0:.0f}s")
    return {"baseline": {k: base[k] for k in ("go", "switch", "value")}, "components": out}


# ----------------------------------------------------------------------------- clamp tests

def next_go(phase_row):
    """For each step of a window (all windows share the layout), the step of the first GO at or after it (-1)."""
    T = len(phase_row)
    nxt, cur = np.full(T, -1), -1
    for t in range(T - 1, -1, -1):
        if phase_row[t] == 1:
            cur = t
        nxt[t] = cur
    return nxt


def clamp(rnn, tcs, groups, refs, ctx, ip, gates, sources, args, rng, n_controls, log):
    eval_rows = np.concatenate([g["rows"].ravel() for g in groups])
    pos = {}
    start = 0
    for gi, g in enumerate(groups):
        pos[gi] = (start + np.arange(g["rows"].size)).reshape(g["rows"].shape)
        start += g["rows"].size
    results = []
    for gate in gates:
        for source in sources:
            tc = tcs[gate] if source == "features" else None
            width = tc.n_feats if tc is not None else 48
            tiers = verdict_tiers(gate, source, width, args)
            comps = [int(i) for i in np.nonzero(tiers == 0)[0]]
            info = {}
            for comp in comps:
                p = predictions(args.store, ip, gate, source, comp, args)
                if p is None:
                    continue
                pred, act = np.asarray(p[0])[eval_rows], np.asarray(p[3])[eval_rows]
                ph = np.asarray(ip.cols["phase"])[eval_rows]
                phases = [k for k in range(3) if np.std(pred[ph == k]) > 1e-9]
                if not phases:
                    continue
                inph = np.isin(ph, phases)
                if p[-1] == "predicate":
                    H, L = inph & (pred > 0), inph & (pred <= 0)
                else:
                    lo, hi = np.quantile(pred[inph], [0.15, 0.85])
                    H, L = inph & (pred >= hi), inph & (pred <= lo)
                if H.sum() < 200 or L.sum() < 200:
                    continue
                info[comp] = {"H": H, "L": L, "on_value": float(act[H].mean()), "off_value": float(act[L].mean()),
                              "kind": p[-1]}
            t0 = time.time()
            for comp, inf in info.items():
                rec = {"gate": gate, "source": source, "id": comp, "kind": inf["kind"],
                       "on_value": inf["on_value"], "off_value": inf["off_value"]}
                for mode, cond, value in (("on", "L", inf["on_value"]), ("off", "H", inf["off_value"])):
                    picks, masks, targets = [], [], []
                    for gi, g in enumerate(groups):
                        cand = inf[cond][pos[gi]]                                  # (n, T)
                        ng = next_go(ctx[gi]["phase"][0])
                        cand = cand & (ng >= 0)[None, :]
                        steps = np.full(len(cand), -1)
                        for w in np.nonzero(cand.any(1))[0]:
                            steps[w] = rng.choice(np.nonzero(cand[w])[0])
                        m = np.zeros(cand.shape, dtype=bool)
                        ok = steps >= 0
                        m[np.nonzero(ok)[0], steps[ok]] = True
                        picks.append((steps, ng))
                        masks.append(torch.from_numpy(m))
                        targets.append(torch.full(cand.shape, value))
                    # natural contrast: H minus L, on the readouts the clamp is measured on
                    def readouts(gi, steps, ng, outs=None):
                        o = refs[gi] if outs is None else outs[gi]
                        w = np.nonzero(steps >= 0)[0]
                        s, gs = steps[w], ng[steps[w]]
                        return (o[w, gs, 1] - o[w, gs, 2]).numpy(), o[w, s, 3].numpy()
                    nat = {}
                    for name, cnd in (("H", "H"), ("L", "L")):
                        lo_c, va = [], []
                        for gi, g in enumerate(groups):
                            mm = inf[cnd][pos[gi]] & (next_go(ctx[gi]["phase"][0]) >= 0)[None, :]
                            w, t = np.nonzero(mm)
                            ng = next_go(ctx[gi]["phase"][0])
                            lo_c.append(ctx[gi]["logit"][w, ng[t]]); va.append(ctx[gi]["value"][w, t])
                        nat[name] = (np.concatenate(lo_c).mean(), np.concatenate(va).mean())
                    d_nat = (nat["H"][0] - nat["L"][0], nat["H"][1] - nat["L"][1])
                    sign = +1 if mode == "on" else -1                                  # off: predicted opposite

                    def effect(component, value_for):
                        tgt = [torch.full(m.shape, value_for) for m in masks]
                        edit_for = lambda gi, s, ch: make_edit(gate, source, component, tgt[gi][s:s + ch],
                                                               masks[gi][s:s + ch], tc)
                        outs = run_groups(rnn, groups, edit_for)
                        dl, dv = [], []
                        for gi in range(len(groups)):
                            steps, ng = picks[gi]
                            a, b = readouts(gi, steps, ng, outs)
                            a0, b0 = readouts(gi, steps, ng)
                            dl.append(a - a0); dv.append(b - b0)
                        return np.concatenate(dl), np.concatenate(dv)
                    dl, dv = effect(comp, value)
                    others = [c for c in info if c != comp]
                    ctrl = []
                    for c in rng.choice(others, size=min(n_controls, len(others)), replace=False) if others else []:
                        cv = info[c]["on_value"] if mode == "on" else info[c]["off_value"]
                        cl, cvv = effect(int(c), cv)
                        ctrl.append((float(cl.mean()), float(cvv.mean())))
                    res = {"n": int(len(dl)), "delta_nat_logit": float(d_nat[0]), "delta_nat_value": float(d_nat[1]),
                           "d_logit": float(dl.mean()), "d_value": float(dv.mean()),
                           "agree_logit": float(np.mean(np.sign(dl) == sign * np.sign(d_nat[0]))),
                           "agree_value": float(np.mean(np.sign(dv) == sign * np.sign(d_nat[1]))),
                           "controls": ctrl}
                    for k, d, dn in (("logit", dl, d_nat[0]), ("value", dv, d_nat[1])):
                        proj = sign * np.sign(dn) * d.mean()                           # along the predicted direction
                        cproj = [sign * np.sign(dn) * (c[0] if k == "logit" else c[1]) for c in ctrl]
                        res[f"{k}_projected"] = float(proj)
                        res[f"{k}_beats_controls"] = bool(ctrl) and bool(proj > max(cproj))
                    rec[mode] = res
                # the readout this component is about: the one its H / L contrast moves more (in readout std)
                key = "logit" if abs(rec["on"]["delta_nat_logit"]) / args.logit_std >= \
                    abs(rec["on"]["delta_nat_value"]) / args.value_std else "value"
                rec["readout"] = key
                rec["pass"] = all(rec[m][f"{key}_projected"] > 0 and rec[m][f"{key}_beats_controls"] for m in ("on", "off"))
                rec["sign_ok"] = all(rec[m][f"{key}_projected"] > 0 for m in ("on", "off"))
                results.append(rec)
            n_pass = sum(r["pass"] for r in results if r["gate"] == gate and r["source"] == source)
            n_tot = sum(1 for r in results if r["gate"] == gate and r["source"] == source)
            log(f"clamp {gate} {source}: {n_pass}/{n_tot} pass ({time.time() - t0:.0f}s)")
    return results


def clamp_setup(gate, source, comp, args, ip, groups, eval_rows):
    """H / L masks, clamp values and the explanation's prediction for one component (None if untestable)."""
    p = predictions(args.store, ip, gate, source, comp, args)
    if p is None:
        return None
    pred, act = np.asarray(p[0])[eval_rows], np.asarray(p[3])[eval_rows]
    ph = np.asarray(ip.cols["phase"])[eval_rows]
    phases = [k for k in range(3) if np.std(pred[ph == k]) > 1e-9]
    if not phases:
        return None
    inph = np.isin(ph, phases)
    if p[-1] == "predicate":
        H, L = inph & (pred > 0), inph & (pred <= 0)
    else:
        lo, hi = np.quantile(pred[inph], [0.15, 0.85])
        H, L = inph & (pred >= hi), inph & (pred <= lo)
    return {"H": H, "L": L, "act": act, "on_value": float(act[H].mean()), "off_value": float(act[L].mean()),
            "kind": p[-1]}


def diagnose(rnn, tcs, groups, refs, ctx, ip, comps, args, rng, log):
    """Is the clamp test faithful? For each (gate, source, id): an identity edit, the single-step clamp, a
    double-dose clamp, a clamp held at every eligible step, and the natural contrast within previous-trial patterns."""
    eval_rows = np.concatenate([g["rows"].ravel() for g in groups])
    pos, start = {}, 0
    for gi, g in enumerate(groups):
        pos[gi] = (start + np.arange(g["rows"].size)).reshape(g["rows"].shape)
        start += g["rows"].size
    last = np.concatenate([c["last"].ravel() for c in ctx])
    out = []
    for gate, source, comp in comps:
        tc = tcs[gate] if source == "features" else None
        s = clamp_setup(gate, source, comp, args, ip, groups, eval_rows)
        if s is None:
            continue
        rec = {"gate": gate, "source": source, "id": comp, "kind": s["kind"]}
        ng = [next_go(c["phase"][0]) for c in ctx]
        # identity: set the component to its own clean value at every step -> no change expected
        act_g = [torch.from_numpy(s["act"][pos[gi]]).float() for gi in range(len(groups))]
        outs = run_groups(rnn, groups, lambda gi, st, ch: make_edit(gate, source, comp, act_g[gi][st:st + ch],
                                                                    torch.ones_like(act_g[gi][st:st + ch], dtype=torch.bool), tc))
        rec["identity_max_abs_change"] = float(max((o - r).abs().max() for o, r in zip(outs, refs)))
        # natural contrast: overall, and within previous-trial patterns (weighted by L-step count)
        def nat(H, L):
            # the choice logit at the next GO, for every step
            lo = np.concatenate([c["logit"][:, np.maximum(n_, 0)].ravel() for c, n_ in zip(ctx, ng)])
            va = np.concatenate([c["value"].ravel() for c in ctx])
            ok = np.concatenate([np.broadcast_to((n_ >= 0)[None, :], c["logit"].shape).ravel() for c, n_ in zip(ctx, ng)])
            h, l = H & ok, L & ok
            res = {"logit": float(lo[h].mean() - lo[l].mean()), "value": float(va[h].mean() - va[l].mean())}
            w, dl, dv = 0, 0.0, 0.0
            for k in range(len(PATTERNS)):
                hk, lk = h & (last == k), l & (last == k)
                if hk.sum() >= 50 and lk.sum() >= 50:
                    n = lk.sum()
                    dl += n * (lo[hk].mean() - lo[lk].mean()); dv += n * (va[hk].mean() - va[lk].mean()); w += n
            res["logit_within_pattern"] = float(dl / w) if w else None
            res["value_within_pattern"] = float(dv / w) if w else None
            res["share_of_L_in_matched_patterns"] = float(w / max(int(l.sum()), 1))
            return res
        rec["natural"] = nat(s["H"], s["L"])
        for mode, cond, value in (("on", "L", s["on_value"]), ("off", "H", s["off_value"])):
            for variant in ("single", "double", "held"):
                masks, targets, picks = [], [], []
                for gi, g in enumerate(groups):
                    cand = s[cond][pos[gi]] & (ng[gi] >= 0)[None, :]
                    steps = np.full(len(cand), -1)
                    for w in np.nonzero(cand.any(1))[0]:
                        steps[w] = rng.choice(np.nonzero(cand[w])[0])
                    m = np.zeros(cand.shape, dtype=bool)
                    ok = steps >= 0
                    if variant == "held":            # every eligible step up to the measured GO
                        for w in np.nonzero(ok)[0]:
                            m[w] = cand[w] & (np.arange(cand.shape[1]) <= ng[gi][steps[w]])
                    else:
                        m[np.nonzero(ok)[0], steps[ok]] = True
                    tgt = torch.full(cand.shape, value)
                    if variant == "double":          # push twice as far from the clean value
                        clean = act_g[gi]
                        tgt = clean + 2 * (tgt - clean)
                    masks.append(torch.from_numpy(m)); targets.append(tgt); picks.append(steps)
                outs = run_groups(rnn, groups, lambda gi, st, ch: make_edit(gate, source, comp, targets[gi][st:st + ch],
                                                                            masks[gi][st:st + ch], tc))
                dl, dv = [], []
                for gi in range(len(groups)):
                    steps = picks[gi]
                    w = np.nonzero(steps >= 0)[0]
                    gs = ng[gi][steps[w]]
                    dl.append(((outs[gi][w, gs, 1] - outs[gi][w, gs, 2]) - (refs[gi][w, gs, 1] - refs[gi][w, gs, 2])).numpy())
                    dv.append((outs[gi][w, steps[w], 3] - refs[gi][w, steps[w], 3]).numpy())
                dl, dv = np.concatenate(dl), np.concatenate(dv)
                rec[f"{mode}_{variant}"] = {"d_logit": float(dl.mean()), "d_value": float(dv.mean()), "n": int(len(dl))}
        out.append(rec)
        log(f"  diagnose {gate} {source} {comp}: identity {rec['identity_max_abs_change']:.1e}; nat logit "
            f"{rec['natural']['logit']:+.2f} (within {rec['natural']['logit_within_pattern']}) | on d_logit single "
            f"{rec['on_single']['d_logit']:+.3f} double {rec['on_double']['d_logit']:+.3f} held {rec['on_held']['d_logit']:+.3f}"
            f" | off single {rec['off_single']['d_logit']:+.3f} held {rec['off_held']['d_logit']:+.3f}")
    return out


def substitute_one(rnn, tcs, groups, refs, ctx, ip, comps, importance_by_key, args, log, min_drop=0.02):
    """Per-component explanation test, where the component matters: replace it (at every step) by its
    explanation's prediction vs by its (phase, episode-decile) mean, and compare GO-choice agreement on the GO steps
    after its worst previous-trial pattern, and the value error at that pattern's steps.
    recovery = (substituted - ablated) / (clean - ablated); pass = recovery >= 0.5 on the GO choice (or on the value,
    when ablation barely moves the choice)."""
    eval_rows = np.concatenate([g["rows"].ravel() for g in groups])
    pos, start = {}, 0
    for gi, g in enumerate(groups):
        pos[gi] = (start + np.arange(g["rows"].size)).reshape(g["rows"].shape)
        start += g["rows"].size
    out = []
    for gate, source, comp in comps:
        tc = tcs[gate] if source == "features" else None
        p = predictions(args.store, ip, gate, source, comp, args)
        if p is None:
            continue
        pred = np.asarray(p[0], dtype=np.float64)
        if source == "units":
            lo, hi = UNIT_RANGE[gate]
            pred = np.clip(pred, lo, hi)
        act = activations(ip, source, gate, comp, None)
        tm = timing_mean(ip, act)
        imp = importance_by_key[(gate, source, comp)]["mean"]
        worst = imp["worst_pattern_go"][0]
        k = PATTERNS.index(worst) if worst in PATTERNS else None

        def run_with(values):
            tg = [torch.from_numpy(values[g["rows"]]).float() for g in groups]
            return run_groups(rnn, groups, lambda gi, st, ch: make_edit(
                gate, source, comp, tg[gi][st:st + ch], torch.ones_like(tg[gi][st:st + ch], dtype=torch.bool), tc))

        def scores(outs):
            go_ok, go_n, v_err, v_n = 0, 0, 0.0, 0
            for o, r, cx in zip(outs, refs, ctx):
                sel = cx["last"] == k
                same = o[..., 1:3].argmax(-1).numpy() == cx["choice"]
                go_ok += int(same[sel & cx["go"]].sum()); go_n += int((sel & cx["go"]).sum())
                v_err += float(np.abs(o[..., 3].numpy() - cx["value"])[sel].sum()); v_n += int(sel.sum())
            return go_ok / max(go_n, 1), v_err / max(v_n, 1) / args.value_std
        a_go, a_v = scores(run_with(tm))
        s_go, s_v = scores(run_with(pred))
        rec = {"gate": gate, "source": source, "id": comp, "kind": p[-1], "worst_pattern": worst,
               "ablated_go": a_go, "substituted_go": s_go, "ablated_value_err": a_v, "substituted_value_err": s_v}
        rec["recovery_go"] = (s_go - a_go) / (1 - a_go) if 1 - a_go >= min_drop else None
        rec["recovery_value"] = (a_v - s_v) / a_v if a_v > 1e-3 else None
        main = rec["recovery_go"] if rec["recovery_go"] is not None else rec["recovery_value"]
        rec["recovery"] = main
        rec["pass"] = bool(main is not None and main >= 0.5)
        out.append(rec)
        log(f"  substitute {gate} {source} {comp} ({worst}): GO agreement ablated {a_go:.3f} -> explanation {s_go:.3f} "
            f"(recovery {rec['recovery_go'] if rec['recovery_go'] is None else round(rec['recovery_go'], 2)}); value err "
            f"{a_v:.3f} -> {s_v:.3f} | {'PASS' if rec['pass'] else 'fail'}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--store", default="rl_features/rl_chosen/chosen_rl_v1_r128_store")
    ap.add_argument("--transcoders", nargs="+", required=True, metavar="GATE=CKPT")
    ap.add_argument("--feature_results", default="rl_features/rl_autointerp")
    ap.add_argument("--feature_magnitude", default="rl_features/rl_magnitude")
    ap.add_argument("--unit_results", nargs="*", default=["rl_features/rl_neuron_d01_autointerp",
                                                         "rl_features/rl_neuron_matched_autointerp",
                                                         "rl_features/rl_neuron_d05_autointerp"])
    ap.add_argument("--unit_magnitude", default="rl_features/rl_magnitude_units")
    ap.add_argument("--feature_magnitude_observed", default="")
    ap.add_argument("--unit_magnitude_observed", default="")
    ap.add_argument("--feature_refined", default="",
                    help="substitute: refined explanations (rl_features/rl_refined) used before the other records")
    ap.add_argument("--components", nargs="*", default=None, metavar="GATE/SOURCE/ID",
                    help="substitute: exactly these components (no usefulness or verdict filter)")
    ap.add_argument("--importance_from", default="",
                    help="substitute: read the importance section from this report when --out has none")
    ap.add_argument("--gates", nargs="*", default=["reset", "update", "candidate"])
    ap.add_argument("--sources", nargs="*", default=["features", "units"], choices=["features", "units"])
    ap.add_argument("--sections", nargs="*", default=["importance", "clamp"],
                    choices=["importance", "clamp", "diagnose", "substitute"])
    ap.add_argument("--useful_go", type=float, default=0.05,
                    help="diagnose: components whose worst-pattern GO drop (importance) is at least this")
    ap.add_argument("--n_windows", type=int, default=6000)
    ap.add_argument("--n_controls", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(args.seed)
    log = lambda s: print(s, flush=True)

    rnn = build_task_rnn("rl", input_size=8, hidden_size=48)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    rnn.eval()
    groups, ip, _ = eval_windows(args.store, args.traces, args.n_windows, seed=args.seed)
    refs = [run_edit(rnn, g["inputs"], g["h0"]) for g in groups]
    ctx = context(groups, ip, refs)
    vals = np.concatenate([c["value"].ravel() for c in ctx])
    lg = np.concatenate([c["logit"][c["go"]] for c in ctx])
    args.value_std, args.logit_std = float(vals.std()), float(lg.std())
    tcs = {k: load_transcoder(v) for k, v in (s.split("=", 1) for s in args.transcoders)}
    report = json.load(open(args.out)) if os.path.exists(args.out) else {}
    report["args"] = {k: v for k, v in vars(args).items()}
    if "importance" in args.sections:
        report["importance"] = importance(rnn, tcs, groups, refs, ctx, ip, args.gates, args.sources, args.value_std, log)
        json.dump(report, open(args.out, "w"), indent=1)
    if "clamp" in args.sections:
        report["clamp"] = clamp(rnn, tcs, groups, refs, ctx, ip, args.gates, args.sources, args, rng,
                                args.n_controls, log)
        json.dump(report, open(args.out, "w"), indent=1)
    if "diagnose" in args.sections:
        # the explained components that matter where they take part (needs importance and clamp results)
        useful = {(r["gate"], r["source"], r["id"]) for r in report["importance"]["components"]
                  if (r["mean"]["worst_pattern_go"][1] or 0) >= args.useful_go}
        comps = [(r["gate"], r["source"], r["id"]) for r in report["clamp"] if (r["gate"], r["source"], r["id"]) in useful]
        log(f"diagnose: {len(comps)} explained useful components")
        report["diagnose"] = diagnose(rnn, tcs, groups, refs, ctx, ip, comps, args, rng, log)
        json.dump(report, open(args.out, "w"), indent=1)
    if "substitute" in args.sections:
        if "importance" not in report and args.importance_from:
            report["importance"] = json.load(open(args.importance_from))["importance"]
        by_key = {(r["gate"], r["source"], r["id"]): r for r in report["importance"]["components"]}
        if args.components is not None:
            comps = [(g, s, int(i)) for g, s, i in (c.split("/") for c in args.components)]
        else:
            comps = [(g, s, i) for (g, s, i), r in by_key.items()
                     if (r["mean"]["worst_pattern_go"][1] or 0) >= args.useful_go]
            tiers = {(g, s): verdict_tiers(g, s, tcs[g].n_feats if s == "features" else 48, args)
                     for g in args.gates for s in args.sources}
            comps = [(g, s, i) for g, s, i in comps if tiers[(g, s)][i] == 0]
        log(f"substitute: {len(comps)} explained useful components")
        report["substitute"] = substitute_one(rnn, tcs, groups, refs, ctx, ip, comps, by_key, args, log)
        json.dump(report, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
