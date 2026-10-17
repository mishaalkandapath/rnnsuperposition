"""Causal checks of the two-step-task circuits (real GRU, edits along feature directions, transcoder error kept):
is the circuit read off the contrastive graphs enough to carry a change to the choice, and is it needed?

For each contrast (an original run o and an edited run s of the same window; contrast_flow.py, reversal_flow.py)
the circuit is that window's top-k features per step by |flow| (contrast_flow.full_flow), over the steps from the
first changed input to the target choice. Patching sets a feature of a gate to its value in the other run's clean
pass. At the patched steps every gate is its transcoder's reconstruction from the live features (patched ones set
to their target) plus that transcoder's error held at its value in the clean pass of the run being patched, as the
error nodes are frozen in the graphs. (Kept live, the error is not small on mixed inputs: with the gate from one run
and the input from the other, the candidate transcoder's error is about 1.4 instead of 0.05, and it carries the
change.) A reset edit reaches the candidate. Patched gate values are kept in the GRU's own ranges (the share
clipped is reported). Four modes,
as in the copy task's minimal circuit (docs/interp_circuits.md section 3):

  sufficiency free     run s; circuit set to its values in o; everything else recomputed
  sufficiency clamped  run s; circuit set to its values in o; every other feature held at its value in s
  necessity free       run o; circuit set to its values in s; everything else recomputed
  necessity clamped    run o; circuit set to its values in s; every other feature held at its value in o

Score: the share of the target change carried, (y - y_s) / (y_o - y_s) for sufficiency, (y_o - y) / (y_o - y_s) for
necessity (windows with |y_o - y_s| >= 1 logit), and how often the choice ends on o's side (sufficiency) or s's side
(necessity) where the two runs choose differently. Control: k random other features per step that also changed.

Contrasts: state (c.D2 state swapped), reward ((c+1).D1 reward swapped), target the next choice; reversal (the
switch trial's choice, recent outcomes replaced by the old world's, reversal_flow "old_world").

    python -m circuit.rl_task.circuit_checks --out rl_features/rl_circuits/circuit_checks.json [--n_windows 100]
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.contrast_flow import REW, edit, full_flow, gru_run, load_model, load_windows

KS = (1, 3, 5, 10, 20)
GATE_KEYS = {"r": "f_r", "z": "f_z", "n": "f_n"}


@torch.no_grad()
def run_patched(rnn, tcs, x, h0, mask, value, steps=None, err=None, head="choice"):
    """Real GRU on windows x (N, T, 8) from h0 (N, H); mask / value: {kind: (N, T, F)}: features set to value
    where mask. At `steps` (N, T) each gate is its transcoder's reconstruction plus the error err[kind] (N, T, H)
    instead (without steps: the real gate plus the patched features' change). Returns the choice logit a1 - a2
    (or the value, head="value") after every step (N, T)."""
    gru, readout = rnn.layers[0], rnn.layers[-1]
    tr, tz, tn = tcs["reset"], tcs["update"], tcs["candidate"]
    h, out = h0, []
    for t in range(x.shape[1]):
        xt = x[:, t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        rep = steps[:, t][:, None] if steps is not None else None
        r = gate(tr, torch.cat([h, xt], -1), r, mask["r"][:, t], value["r"][:, t], rep, err and err["r"][:, t], 0)
        z = gate(tz, torch.cat([h, xt], -1), z, mask["z"][:, t], value["z"][:, t], rep, err and err["z"][:, t], 0)
        n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        n = gate(tn, torch.cat([r * h, xt], -1), n, mask["n"][:, t], value["n"][:, t], rep, err and err["n"][:, t], -1)
        h = (1 - z) * h + z * n
        o = readout(h)
        out.append(o[:, 1] - o[:, 2] if head == "choice" else o[:, 3])
    return torch.stack(out, 1)


CLAMPED = [0, 0]                                         # gate values outside the GRU's range, values patched


def gate(tc, inp, real, m, value, rep, err, low):
    """One gate value: real + the patched features' change, or (where rep) reconstruction + held error; kept in the
    GRU's own range [low, 1] (a patched update gate outside [0, 1] makes h grow without bound)."""
    if not m.any() and (rep is None or not rep.any()):
        return real
    live = tc.act(tc.input_to_features(inp))
    a = torch.where(m, value, live)
    out = real + (a - live) @ tc.features_to_outputs.weight.T
    if rep is not None:
        out = torch.where(rep, tc.features_to_outputs(a) + err, out)
    CLAMPED[0] += int(((out < low) | (out > 1)).sum()); CLAMPED[1] += out.numel()
    return out.clamp(low, 1)


def prepare(rnn, tracer, pairs, w_sign):
    """pairs: [(x_o, x_s, h0, tau, first_step)]. Per item: both runs' clean activations, the target change and
    every changed node's flow."""
    items = []
    for x_o, x_s, h0, tau, first in pairs:
        acts = [tracer.run_forward_pass(gru_run(rnn, x[:tau + 1], h0)) for x in (x_o, x_s)]
        f = full_flow(tracer, acts[0], acts[1], w_sign(x_o) * tracer.W_choice, tau)
        y = [float(a["choice"][tau]) * w_sign(x_o) for a in acts]
        changed = [(nd, abs(fl)) for nd, fl, da in zip(f["nodes"], f["flow"].tolist(), f["da"].tolist())
                   if da != 0 and nd[1] >= first]
        items.append({"x": (x_o[:tau + 1], x_s[:tau + 1]), "h0": h0, "tau": tau, "y": y,
                      "a": [{k: a[v] for k, v in GATE_KEYS.items()} for a in acts], "changed": changed,
                      "e": [{k: a[f"e_{k}"] for k in "rzn"} for a in acts],
                      "sign": w_sign(x_o)})
    return items


def select(item, k, rng, random_control):
    """Top-k changed nodes per step by |flow| (or k random changed nodes per step outside the top-k)."""
    by_step = defaultdict(list)
    for nd, fl in item["changed"]:
        by_step[nd[1]].append((fl, nd))
    out = []
    for t, lst in by_step.items():
        lst.sort(key=lambda v: -v[0])
        if random_control:
            rest = [nd for _, nd in lst[k:]]
            out += [rest[j] for j in rng.choice(len(rest), size=min(k, len(rest)), replace=False)] if rest else []
        else:
            out += [nd for _, nd in lst[:k]]
    return out


def evaluate(rnn, tcs, items, k, rng, random_control):
    by_len = defaultdict(list)
    for it in items:
        by_len[it["tau"] + 1].append(it)
    res = defaultdict(list)
    for T, its in by_len.items():
        N = len(its)
        circ = [select(it, k, rng, random_control) for it in its]
        x = [torch.stack([it["x"][j] for it in its]) for j in (0, 1)]
        h0 = torch.stack([it["h0"] for it in its])
        vals = [{g: torch.stack([it["a"][j][g] for it in its]) for g in "rzn"} for j in (0, 1)]
        cmask = {g: torch.zeros_like(vals[0][g], dtype=torch.bool) for g in "rzn"}
        for i, nodes in enumerate(circ):
            for g, t, f in nodes:
                cmask[g][i, t, f] = True
        # every feature active in either run at the patched steps (for the clamped modes)
        first = [min((nd[1] for nd, _ in it["changed"]), default=T) for it in its]
        steps = torch.zeros(N, T, dtype=torch.bool)
        for i, f0 in enumerate(first):
            steps[i, f0:] = True
        allm = {g: ((vals[0][g] > 0) | (vals[1][g] > 0)) & steps[:, :, None] for g in "rzn"}
        y_o = torch.tensor([it["y"][0] for it in its]); y_s = torch.tensor([it["y"][1] for it in its])
        sign = torch.tensor([it["sign"] for it in its])
        for mode, run, src, other in (("sufficiency_free", 1, 0, None), ("sufficiency_clamped", 1, 0, 1),
                                      ("necessity_free", 0, 1, None), ("necessity_clamped", 0, 1, 0)):
            mask = {g: cmask[g] | (allm[g] if other is not None else False) for g in "rzn"}
            value = {g: torch.where(cmask[g], vals[src][g], vals[other][g] if other is not None else vals[src][g])
                     for g in "rzn"}
            err = {g: torch.stack([it["e"][run][g] for it in its]) for g in "rzn"}
            y = run_patched(rnn, tcs, x[run], h0, mask, value, steps, err)[:, -1] * sign
            d = y_o - y_s
            frac = (y - y_s) / d if mode.startswith("suff") else (y_o - y) / d
            differ = torch.sign(y_o) != torch.sign(y_s)
            target_side = torch.sign(y) == (torch.sign(y_o) if mode.startswith("suff") else torch.sign(y_s))
            keep = d.abs() >= 1
            res[mode + "_share"] += frac[keep].tolist()
            res[mode + "_choice"] += target_side[differ].tolist()
        res["n_features_per_step"] += [len(c) / max(1, len({nd[1] for nd in c})) for c in circ]
    out = {k_: round(float(np.median(v)), 3) if k_.endswith("share") else round(float(np.mean(v)), 3)
           for k_, v in res.items()}
    out["gate_values_clamped"] = round(CLAMPED[0] / max(CLAMPED[1], 1), 5)
    CLAMPED[0] = CLAMPED[1] = 0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", default="rl_features/rl_trace_sequences")
    ap.add_argument("--n_windows", type=int, default=100, help="windows per core-trial pattern")
    ap.add_argument("--contrasts", nargs="+", default=["state", "reward", "reversal"])
    ap.add_argument("--episodes", type=int, default=3000, help="reversal: rollout episodes")
    ap.add_argument("--max_events", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rnn, tcs, tracer = load_model()
    rng = np.random.default_rng(args.seed)
    result = {}
    for contrast in args.contrasts:
        if contrast == "reversal":
            from circuit.rl_task.reversal_flow import rollout, settled_reversals, switch_events
            X, Hp, rec = rollout(rnn, args.episodes, args.seed)
            K = 5
            pairs, signs = [], []
            for e, R, S, old_a, new_a in [v for v in switch_events(rec, settled_reversals(rec)) if v[2] >= K][:args.max_events]:
                lo = S - K
                x = X[e, 3 * lo:3 * S + 2].clone()
                x_cf = x.clone()
                for k in range(K):
                    if rec["choice"][e, lo + k] > 0 and rec["reward"][e, lo + k] >= 0:
                        x_cf[3 * (k + 1), REW] = float(rec["state"][e, lo + k] == old_a)
                if not torch.equal(x, x_cf):
                    pairs.append((x, x_cf, Hp[e, 3 * lo], 3 * K + 1, 3)); signs.append(1.0 if new_a == 1 else -1.0)
            sign_of = {id(p[0]): s for p, s in zip(pairs, signs)}
            items = prepare(rnn, tracer, pairs, lambda x: sign_of[id(x)])
        else:
            windows = load_windows(args.traces, args.n_windows, args.seed)
            pairs = [(w["x"], edit(w, contrast), w["h0"], 3 * w["c"] + 4,
                      3 * w["c"] + (2 if contrast == "state" else 3)) for w in windows]
            items = prepare(rnn, tracer, pairs, lambda x: 1.0)
        print(f"{contrast}: {len(items)} pairs prepared ({time.time() - t0:.0f}s)", flush=True)
        out = {"n": len(items), "median_abs_change": round(float(np.median([abs(i["y"][0] - i["y"][1]) for i in items])), 3)}
        for k in KS:
            out[f"top{k}"] = evaluate(rnn, tcs, items, k, rng, False)
            out[f"random{k}"] = evaluate(rnn, tcs, items, k, rng, True)
            print(f"{contrast} k={k}: {out[f'top{k}']}  | random: {out[f'random{k}']} ({time.time() - t0:.0f}s)", flush=True)
        result[contrast] = out
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
