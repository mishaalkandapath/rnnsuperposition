"""Interventions that test the copy-circuit account (real GRU, one length).

  del      the <DEL> candidate feature (default 123): removed at the delimiter
           (mean / resample), alone and with the reset-gate <DEL> feature
           (default 31); and its <DEL>-step write injected one step early.
           Prediction: reading fails; an early injection shifts the readout.
  pop      pop features: candidate features at copy step L+k that suppress the
           just-emitted token orig[k-1], found by exact contrastive flow
           (circuit.copy_task.feature_circuits.change_flow) between the true feedback and
           a mismatched one, target = logit of orig[k-1] at output k. Then the
           top ones are removed at the copy steps. Prediction: repeats of the
           previous token (as with mismatched feedback).
  timing   freeze the hidden state for one copy step (z = 0: nothing written or
           erased), teacher-forced and free-running (outputs fed back).
           Queue prediction: free-running outputs are delayed by one, not skipped.
  carry    at a mid-relay step of slot i: (a) steer the slot-i carriers by the
           slot-token tables as if slot i held another token; (b) z = 1 at that
           step (h = the candidate's rewrite; the carried copy is dropped);
           (c) both. If the carried copy is what makes single-step steering
           fail, (c) flips output i where (a) does not.
  resample the dense-feature drop tests and the gate-feature ablations with
           resample ablation (each feature takes its value from another real
           sequence at the same step) next to mean ablation, with random controls.

Ablations edit the real gate / candidate by the feature's change:
value - (a_f - replacement) * d_f, keeping the transcoder's error.

    python -m circuit.copy_task.circuit_tests --rnn RNN.ckpt --traces T4.pt \\
        --transcoders reset=R.ckpt update=U.ckpt candidate=C.ckpt --out circuit_tests.json
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import StackDataset

from circuit.copy_task.autointerp import TOKENS
from circuit.copy_task.copy_circuits import gru_trace, step_label, swap_token
from circuit.edge_att import CircuitTracer
from circuit.copy_task.explained_fidelity import load_transcoder
from circuit.copy_task.feature_circuits import candidate_acts, change_flow, selectivity
from training.splice_eval import build_task_rnn

torch.serialization.add_safe_globals([StackDataset])
FIVE = [162, 214, 284, 388, 487]
CIRCUIT_DENSE = [8, 32, 54, 77, 79, 91, 212, 231, 266, 274, 349, 368, 387, 440, 500]


@torch.no_grad()
def run(rnn, x, L, hook=None, free=False):
    """GRU over x (N, T, X); hook(t, h, xt, r, z, n) -> (r, z, n). free: copy-phase inputs are the
    model's own previous outputs. Returns output-step logits (N, L, V)."""
    gru, readout = rnn.layers[0], rnn.layers[-1]
    N, T, X = x.shape
    h = torch.zeros(N, gru.hidden_size)
    out = []
    for t in range(T):
        xt = x[:, t]
        if free and t > L:
            xt = torch.nn.functional.one_hot(out[-1].argmax(-1), X).float()
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        if hook is not None:
            r0, n0 = r, n
            r, z, n = hook(t, h, xt, r, z, n)
            if r is not r0:                                     # a reset edit must reach the candidate:
                n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h)) + (n - n0)   # keep candidate edits
        h = (1 - z) * h + z * n
        if t >= L:
            out.append(readout(h))
    return torch.stack(out, 1)


@torch.no_grad()
def clean_acts(rnn, x, L, tcs):
    """Feature activations of each transcoder on the clean run, per step: {gate: (N, T, F)}."""
    store = defaultdict(list)

    def hook(t, h, xt, r, z, n):
        hx = torch.cat([h, xt], -1)
        store["update"].append(tcs["update"].act(tcs["update"].input_to_features(hx)))
        store["reset"].append(tcs["reset"].act(tcs["reset"].input_to_features(hx)))
        store["candidate"].append(tcs["candidate"].act(tcs["candidate"].input_to_features(torch.cat([r * h, xt], -1))))
        return r, z, n
    run(rnn, x, L, hook)
    return {k: torch.stack(v, 1) for k, v in store.items()}


def feature_hook(tcs, gate, feats, replace, steps=None):
    """Hook removing `feats` of one gate: value - (a_f - replace(t)[:, feats]) d_f at the given steps.
    replace(t) -> (N, F) replacement activations (means broadcast, or another sequence's values)."""
    feats = list(feats)
    tc = tcs[gate]

    def hook(t, h, xt, r, z, n):
        if not feats or (steps is not None and t not in steps):
            return r, z, n
        inp = torch.cat([r * h, xt], -1) if gate == "candidate" else torch.cat([h, xt], -1)
        a = tc.act(tc.input_to_features(inp))[:, feats]
        delta = (a - replace(t)[:, feats]) @ tc.features_to_outputs.weight[:, feats].T
        if gate == "candidate":
            n = n - delta
        elif gate == "update":
            z = (z - delta).clamp(0, 1)
        else:
            r = (r - delta).clamp(0, 1)
        return r, z, n
    return hook


def chain(*hooks):
    def hook(t, h, xt, r, z, n):
        for hk in hooks:
            r, z, n = hk(t, h, xt, r, z, n)
        return r, z, n
    return hook


def metrics(logits, orig):
    """Accuracy per output, and the share of outputs equal to the previous / next original token."""
    pred = logits.argmax(-1)
    L = orig.shape[1]
    prev = torch.full_like(orig, -1)
    prev[:, 1:] = orig[:, :-1]
    nxt = torch.full_like(orig, -1)
    nxt[:, :-1] = orig[:, 1:]
    return {"accuracy": (pred == orig).float().mean(0).tolist(),
            "repeat_previous": (pred == prev).float().mean(0).tolist(),
            "skip_ahead": (pred == nxt).float().mean(0).tolist(),
            "accuracy_all": float((pred == orig).float().mean())}


def fmt(m):
    return (f"acc {' '.join(f'{v:.2f}' for v in m['accuracy'])} | repeat prev {' '.join(f'{v:.2f}' for v in m['repeat_previous'])}"
            f" | skip ahead {' '.join(f'{v:.2f}' for v in m['skip_ahead'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--transcoders", nargs=3, required=True, metavar="GATE=PATH")
    ap.add_argument("--sections", nargs="+", default=["del", "pop", "timing", "carry", "timeshift", "match", "resample"])
    ap.add_argument("--n", type=int, default=3000, help="sequences (distinct tokens)")
    ap.add_argument("--n_flow", type=int, default=300, help="sequences for finding pop features")
    ap.add_argument("--del_feature", type=int, default=123)
    ap.add_argument("--del_reset_feature", type=int, default=31)
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    paths = dict(item.split("=", 1) for item in args.transcoders)
    rnn = build_task_rnn("copy", 31, 128)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    tcs = {k: load_transcoder(paths[k]) for k in ("reset", "update", "candidate")}
    data = torch.load(args.traces, map_location="cpu", mmap=True).datasets
    L = data["outputs"].shape[1]
    T = 2 * L
    tokens_all = data["inputs"][:, :L, :-1].argmax(-1)
    distinct = np.flatnonzero([len(set(r)) == L for r in tokens_all.tolist()])
    rng = np.random.default_rng(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    rows = np.sort(rng.choice(distinct, size=min(args.n, len(distinct)), replace=False))
    x = data["inputs"][torch.from_numpy(rows)].float()
    orig = tokens_all[torch.from_numpy(rows)]
    N = len(x)
    clean = clean_acts(rnn, x, L, tcs)
    means = {k: v.mean((0, 1)) for k, v in clean.items()}                      # global means
    step_means = {k: v.mean(0) for k, v in clean.items()}                       # (T, F)
    perm = torch.randperm(N, generator=g)                                       # donor sequences for resampling
    mean_at = lambda gate: (lambda t: step_means[gate][t].expand(N, -1))
    resample_at = lambda gate: (lambda t: clean[gate][perm, t])
    base = metrics(run(rnn, x, L), orig)
    print(f"L={L}, {N} sequences; baseline: {fmt(base)}")
    results = {"L": L, "n": N, "baseline": base}

    if "del" in args.sections:
        f, fr = args.del_feature, args.del_reset_feature
        res = {}
        for name, hook in [
            (f"candidate {f} at <DEL>, mean", feature_hook(tcs, "candidate", [f], mean_at("candidate"), {L})),
            (f"candidate {f} at <DEL>, zero", feature_hook(tcs, "candidate", [f], lambda t: torch.zeros(N, 512), {L})),
            (f"reset {fr} at <DEL>, mean", feature_hook(tcs, "reset", [fr], mean_at("reset"), {L})),
            (f"reset {fr} at <DEL>, zero", feature_hook(tcs, "reset", [fr], lambda t: torch.zeros(N, 512), {L})),
            (f"both, zero", chain(feature_hook(tcs, "candidate", [f], lambda t: torch.zeros(N, 512), {L}),
                                  feature_hook(tcs, "reset", [fr], lambda t: torch.zeros(N, 512), {L})))]:
            res[name] = metrics(run(rnn, x, L, hook), orig)
            print(f"del: {name:>34}: {fmt(res[name])}")
        # inject 123's <DEL>-step write one step early (step L-1)
        a_del = clean["candidate"][:, L, f]
        write = a_del[:, None] * tcs["candidate"].features_to_outputs.weight[:, f][None]

        def early(t, h, xt, r, z, n, write=write):
            return (r, z, n + write) if t == L - 1 else (r, z, n)
        res[f"inject {f}'s <DEL> write at step L-1"] = metrics(run(rnn, x, L, early), orig)
        print(f"del: {'inject at L-1':>34}: {fmt(res[f'inject {f}' + chr(39) + 's <DEL> write at step L-1'])}")
        results["del"] = res

    if "pop" in args.sections:
        tracer = CircuitTracer(rnn, tcs["update"], tcs["candidate"], device="cpu", reset_transcoder=tcs["reset"])
        flow = defaultdict(float)
        start = time.time()
        frows = rows[:args.n_flow]
        for row in frows:
            xr = data["inputs"][row].float()
            o = tokens_all[row].tolist()
            run_o = tracer.run_forward_pass(gru_trace(rnn, xr))
            for k in range(1, L):
                new = int(rng.choice([v for v in range(30) if v not in o]))
                xm = xr.clone()
                xm[L + k] = 0
                xm[L + k, new] = 1                                              # mismatched feedback at L+k
                run_m = tracer.run_forward_pass(gru_trace(rnn, xm))
                target = list(o)
                target[k] = o[k - 1]                                            # logit of orig[k-1] at output k
                fl = change_flow(tracer, run_o, run_m, L, target, k)
                for (t, ff), v in fl["nodes"].items():
                    if t == L + k:
                        flow[ff] += v / (len(frows) * (L - 1))
        ranked = sorted(flow, key=lambda ff: flow[ff])                          # most negative = suppresses the popped token
        print(f"pop: flows from {len(frows)} sequences in {time.time() - start:.0f}s; top pop features: "
              + ", ".join(f"{ff} ({flow[ff]:+.2f})" for ff in ranked[:10]))
        copy_steps = set(range(L + 1, T))
        active = torch.nonzero((clean["candidate"][:, L + 1:] > 0).float().mean((0, 1)) > 0.05).flatten()
        res = {"top_features": [{"feature": ff, "mean_flow": flow[ff]} for ff in ranked[:20]]}
        for k in (1, 3, 5, 10):
            feats = ranked[:k]
            for mode, rep in (("mean", mean_at("candidate")), ("resample", resample_at("candidate"))):
                m = metrics(run(rnn, x, L, feature_hook(tcs, "candidate", feats, rep, copy_steps)), orig)
                res[f"top {k} pop features, {mode}"] = m
                print(f"pop: top {k:>2} ({mode:>8}): {fmt(m)}")
            rnd = [metrics(run(rnn, x, L, feature_hook(tcs, "candidate", active[torch.randperm(len(active), generator=g)[:k]].tolist(),
                                                     mean_at("candidate"), copy_steps)), orig) for _ in range(args.n_random)]
            res[f"random {k}, mean"] = {key: np.mean([r_[key] for r_ in rnd], 0).tolist() for key in rnd[0]}
            print(f"pop: random {k:>2} (    mean): {fmt(res[f'random {k}, mean'])}")
        results["pop"] = res

    if "timing" in args.sections:
        res = {}
        for k in range(L):
            t0 = L + k

            def freeze(t, h, xt, r, z, n, t0=t0):
                return (r, torch.zeros_like(z), n) if t == t0 else (r, z, n)
            for free in (False, True):
                name = f"freeze h at step {step_label(t0, L)} ({'free-running' if free else 'teacher-forced'})"
                res[name] = metrics(run(rnn, x, L, freeze, free=free), orig)
                print(f"timing: {name:>44}: {fmt(res[name])}")
        res["no intervention (free-running)"] = metrics(run(rnn, x, L, free=True), orig)
        results["timing"] = res

    if "carry" in args.sections:
        acts_all = candidate_acts(tcs["candidate"], {k: data[k] for k in ("inputs", "h_prevs", "r_ts")})
        eta, _, tables = selectivity(acts_all, tokens_all)
        present = torch.zeros(N, 30, dtype=torch.bool).scatter_(1, orig, True)
        new = torch.multinomial((~present).float(), 1, generator=g)[:, 0]
        dec = tcs["candidate"].features_to_outputs.weight
        res = {}
        for i in range(L):
            for t in range(i + 1, L + i):                                       # relay steps of slot i
                feats = torch.nonzero(eta[i, t] >= 0.05).flatten().tolist()
                delta = (tables[i, :, t][:, feats][new] - tables[i, :, t][:, feats][orig[:, i]]) @ dec[:, feats].T

                def steer(tt, h, xt, r, z, n, t=t, delta=delta):
                    return (r, z, n + delta) if tt == t else (r, z, n)

                def overwrite(tt, h, xt, r, z, n, t=t):
                    return (r, torch.ones_like(z), n) if tt == t else (r, z, n)
                for name, hook in (("steer", steer), ("z=1", overwrite), ("z=1 + steer", chain(steer, overwrite))):
                    pred = run(rnn, x, L, hook).argmax(-1)
                    others = [j for j in range(L) if j != i]
                    key = f"slot {i} at {step_label(t, L)}: {name}"
                    res[key] = {"flip_to_new": float((pred[:, i] == new).float().mean()),
                                "acc_own": float((pred[:, i] == orig[:, i]).float().mean()),
                                "acc_others": float((pred[:, others] == orig[:, others]).float().mean()),
                                "n_features": len(feats)}
                    print(f"carry: {key:>34}: flip->new {res[key]['flip_to_new']:.2f}  own acc {res[key]['acc_own']:.2f}  "
                          f"others {res[key]['acc_others']:.2f}")
        results["carry"] = res

    if "timeshift" in args.sections:
        # give a positional gate feature its typical value from another step (the clock is wrong)
        input_steps, copy_steps = list(range(1, L)), list(range(L + 1, T))
        shifts = [("update", 717, "off at in3/<DEL> (copy-phase value)", [L - 1, L], L + 1),
                  ("update", 717, "on in copy phase (<DEL> value)", copy_steps, L),
                  ("update", 650, "off at t=1-3 (copy-phase value)", [1, 2, 3], L + 1),
                  ("update", 650, "on in copy phase (t=2 value)", copy_steps, 2),
                  ("reset", 434, "on in input phase (copy-phase value)", input_steps, L + 1),
                  ("reset", 434, "off in copy phase (input-phase value)", copy_steps, 1),
                  ("reset", 312, "on in copy phase (in3 value)", copy_steps, L - 1),
                  ("reset", 312, "off at in3 (copy-phase value)", [L - 1], L + 1),
                  ("reset", 31, "on at in3 (<DEL> value)", [L - 1], L),
                  ("reset", 31, "off at <DEL> (input-phase value)", [L], 1)]
        res = {}
        for gate, ff, name, steps, src in shifts:
            value = step_means[gate][src].expand(N, -1)
            m = metrics(run(rnn, x, L, feature_hook(tcs, gate, [ff], lambda t, v=value: v, set(steps))), orig)
            moved = float((step_means[gate][src, ff] - step_means[gate][steps, ff].mean()))
            res[f"{gate} {ff}: {name}"] = {**m, "activation_change": moved}
            print(f"timeshift: {gate} {ff} {name:>38} (activation {moved:+.2f}): {fmt(m)}")
        results["timeshift"] = res

    if "match" in args.sections:
        # pop step t = L+k: same h_{t-1}, fed-back input = the stored head orig[k-1] (match) or another token
        gru, readout = rnn.layers[0], rnn.layers[-1]
        W_c = readout.weight - readout.weight.mean(0, keepdim=True)
        hs = []

        def keep_h(t, h, xt, r, z, n):
            hs.append(h.clone())
            return r, z, n
        run(rnn, x, L, keep_h)
        h_prev = torch.stack(hs, 1)                                             # (N, T, H): h entering each step
        present = torch.zeros(N, 30, dtype=torch.bool).scatter_(1, orig, True)
        u = torch.multinomial((~present).float(), 1, generator=g)[:, 0]
        cu, cr, cc = tcs["update"], tcs["reset"], tcs["candidate"]
        agg = defaultdict(float)
        feat_gate, feat_content = torch.zeros(cu.n_feats), torch.zeros(cc.n_feats)
        n_terms = 0
        for k in range(1, L):
            t = L + k
            h = h_prev[:, t]
            v = orig[:, k - 1]
            xs = {"match": torch.nn.functional.one_hot(v, 31).float(), "mismatch": torch.nn.functional.one_hot(u, 31).float()}
            st = {}
            for key, xt in xs.items():
                r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
                z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
                p = gru.input_to_new(xt) + gru.hidden_to_new(r * h)
                n = torch.tanh(p)
                st[key] = {"x": xt, "r": r, "z": z, "p": p, "n": n, "h": (1 - z) * h + z * n,
                           "fu": cu.act(cu.input_to_features(torch.cat([h, xt], -1))),
                           "fc": cc.act(cc.input_to_features(torch.cat([r * h, xt], -1)))}
            a, b = st["match"], st["mismatch"]
            w = W_c[v]                                                          # read direction of the stored head
            zb, nb = (a["z"] + b["z"]) / 2, (a["n"] + b["n"]) / 2
            dn, dz, dp = a["n"] - b["n"], a["z"] - b["z"], a["p"] - b["p"]
            content, gate = (w * zb * dn).sum(-1), (w * dz * (nb - h)).sum(-1)
            total = (w * (a["h"] - b["h"])).sum(-1)
            kappa = torch.where(dp.abs() > 1e-6, dn / torch.where(dp.abs() > 1e-6, dp, torch.ones_like(dp)), torch.zeros_like(dp))
            from_input = (w * zb * kappa * (gru.input_to_new(a["x"]) - gru.input_to_new(b["x"]))).sum(-1)
            from_reset = (w * zb * kappa * gru.hidden_to_new((a["r"] - b["r"]) * h)).sum(-1)
            # trace of the head token in h_{t-1}: h minus the same step of a run with slot k-1 swapped (feedback matched)
            hs2 = []
            xsw = x.clone()
            for tt in [k - 1, L + k]:
                xsw[:, tt] = 0
                xsw[torch.arange(N), tt, u] = 1

            def keep2(tt, hh, xt, r, z, n):
                hs2.append(hh.clone())
                return r, z, n
            run(rnn, xsw[:, :t], L, keep2)
            trace = h - hs2[-1] if len(hs2) == t else None
            for key, val in (("logit_change", total), ("content", content), ("gate", gate),
                             ("content_from_input", from_input), ("content_from_reset", from_reset),
                             ("head_logit_match", (w * a["h"]).sum(-1)), ("head_logit_mismatch", (w * b["h"]).sum(-1))):
                agg[key] += float(val.mean()) / (L - 1)
            if trace is not None:
                unit = trace / trace.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                for key in ("match", "mismatch"):
                    kept = ((st[key]["h"] * unit).sum(-1) / (h * unit).sum(-1)).clamp(-5, 5)
                    agg[f"trace_kept_{key}"] += float(kept.median()) / (L - 1)
            # feature split: gate term by update features, content term by candidate features
            dfu = a["fu"] - b["fu"]
            feat_gate += ((dfu[:, :, None] * cu.features_to_outputs.weight.T[None]) * (w * (nb - h))[:, None, :]).sum(-1).mean(0) / (L - 1)
            dfc = a["fc"] - b["fc"]
            feat_content += ((dfc[:, :, None] * cc.features_to_outputs.weight.T[None]) * (w * zb)[:, None, :]).sum(-1).mean(0) / (L - 1)
        top_g = torch.argsort(feat_gate.abs(), descending=True)[:8].tolist()
        top_c = torch.argsort(feat_content.abs(), descending=True)[:8].tolist()
        results["match"] = {**agg, "gate_by_update_feature": [(f, float(feat_gate[f])) for f in top_g],
                            "content_by_candidate_feature": [(f, float(feat_content[f])) for f in top_c]}
        print("match: head logit after the pop step: matched feedback {:+.1f}, mismatched {:+.1f}; difference {:+.1f} = "
              "content {:+.1f} (input {:+.1f}, reset {:+.1f}) + gate {:+.1f}".format(
                  agg["head_logit_match"], agg["head_logit_mismatch"], agg["logit_change"], agg["content"],
                  agg["content_from_input"], agg["content_from_reset"], agg["gate"]))
        print(f"match: share of the head's trace kept through the pop step: matched {agg.get('trace_kept_match', float('nan')):.2f}, "
              f"mismatched {agg.get('trace_kept_mismatch', float('nan')):.2f}")
        print("match: gate term by update feature: " + ", ".join(f"{f} {v:+.2f}" for f, v in results["match"]["gate_by_update_feature"]))
        print("match: content term by candidate feature: " + ", ".join(f"{f} {v:+.2f}" for f, v in results["match"]["content_by_candidate_feature"]))

    if "reset_combos" in args.sections:
        # reset features turned off (zero) singly, in pairs, in phase groups; failures broken down by
        # output position, by the token being output, and by token class
        R = [434, 312, 31, 322, 76, 271, 147, 477]
        zero = lambda t: torch.zeros(N, tcs["reset"].n_feats)
        res = {}

        def test(name, feats, detail=False):
            pred = run(rnn, x, L, feature_hook(tcs, "reset", feats, zero)).argmax(-1)
            ok = (pred == orig).float()
            rec = {"features": feats, "accuracy": ok.mean(0).tolist(), "accuracy_all": float(ok.mean())}
            if detail:
                tok_acc = {TOKENS[v]: float(ok[orig == v].mean()) for v in range(30) if (orig == v).sum() > 20}
                rec["accuracy_by_output_token"] = tok_acc
                rec["accuracy_digits"] = float(ok[orig >= 26].mean())
                rec["accuracy_letters"] = float(ok[orig < 26].mean())
                # does failure depend on what the previous output was (the fed-back token)?
                prev_tok = torch.full_like(orig, -1)
                prev_tok[:, 1:] = orig[:, :-1]
                rec["accuracy_by_previous_token"] = {TOKENS[v]: float(ok[prev_tok == v].mean()) for v in range(30)
                                                     if (prev_tok == v).sum() > 20}
            res[name] = rec
            return rec
        for f in R:
            test(f"{f} off", [f])
        pairs = []
        for a in range(len(R)):
            for b in range(a + 1, len(R)):
                r_ = test(f"{R[a]} + {R[b]} off", [R[a], R[b]])
                pairs.append((r_["accuracy_all"], R[a], R[b]))
        pairs.sort()
        print("reset_combos: single features off: " + ", ".join(f"{f} {res[f'{f} off']['accuracy_all']:.2f}" for f in R))
        print("reset_combos: worst pairs: " + ", ".join(f"{a}+{b} {acc:.2f}" for acc, a, b in pairs[:8]))
        groups = {"input phase (76, 147, 312, 271)": [76, 147, 312, 271], "<DEL> (31)": [31],
                  "copy phase (434, 322, 477)": [434, 322, 477], "input + <DEL>": [76, 147, 312, 271, 31],
                  "<DEL> + copy": [31, 434, 322, 477], "all 8": R}
        for name, feats in groups.items():
            r_ = test(name, feats, detail=True)
            worst = sorted(r_["accuracy_by_output_token"].items(), key=lambda kv: kv[1])[:5]
            print(f"reset_combos: {name:>32}: acc {r_['accuracy_all']:.3f} | per output " + " ".join(f"{v:.2f}" for v in r_["accuracy"])
                  + f" | digits {r_['accuracy_digits']:.2f} letters {r_['accuracy_letters']:.2f} | worst tokens "
                  + ", ".join(f"{t} {v:.2f}" for t, v in worst))
        # random controls: 8 random active reset features off
        active_r = torch.nonzero((clean["reset"] > 0).float().mean((0, 1)) > 0.02).flatten().tolist()
        pool = [f for f in active_r if f not in R]
        for k in range(args.n_random):
            feats = [int(v) for v in np.random.default_rng(10 + k).choice(pool, len(R), replace=False)]
            r_ = test(f"random 8 #{k}", feats)
            print(f"reset_combos: random 8 #{k} {feats}: acc {r_['accuracy_all']:.3f} | per output " + " ".join(f"{v:.2f}" for v in r_["accuracy"]))
        results["reset_combos"] = res

    if "resample" in args.sections:
        res = {}
        dense = torch.nonzero((clean["candidate"] > 0).float().mean((0, 1)) >= 0.2).flatten().tolist()
        pool = [ff for ff in dense if ff not in FIVE + CIRCUIT_DENSE]
        sets = [("candidate", "the 5", FIVE), ("candidate", "the 15 dense carriers", CIRCUIT_DENSE),
                ("update", "update 717, 650", [717, 650]), ("reset", "reset 434, 31, 76, 322", [434, 31, 76, 322])]
        sets += [("candidate", f"random 15 other dense #{k}", list(np.random.default_rng(k).choice(pool, 15, replace=False)))
                 for k in range(args.n_random)]
        for gate, name, feats in sets:
            for mode, rep in (("mean", mean_at(gate)), ("resample", resample_at(gate))):
                m = metrics(run(rnn, x, L, feature_hook(tcs, gate, [int(v) for v in feats], rep)), orig)
                res[f"{name} ({mode})"] = m
                print(f"resample: {name:>28} ({mode:>8}): acc {m['accuracy_all']:.3f} | per output "
                      + " ".join(f"{v:.2f}" for v in m["accuracy"]))
        results["resample"] = res

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), **results}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
