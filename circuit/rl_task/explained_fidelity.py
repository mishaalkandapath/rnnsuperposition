"""How much of the RL agent's computation runs through explained features?

RL version of circuit/copy_task/explained_fidelity.py (same selections, same
mean ablation). For one gate at a time, keep a subset of transcoder features
(the rest set to their mean) and measure

  fve            teacher-forced: fraction of variance of the true gate value
                 explained, on the stored trace inputs and hidden states;
  splice         the ablated gate runs inside the GRU, so errors compound over
                 the window (each window starts from its stored hidden state;
                 inputs are the recorded ones):
    go_agreement      same a1/a2 choice (argmax) as the original at GO steps
    switch_agreement  the same, only on GO steps where the original switches
                      from the previous trial's choice (the informative ones;
                      repeating is the default)
    kl_go             KL(original || ablated) of the 3-way policy at GO
    value_nmae        |value error| / std(original value), all steps
    nonfinite_frac    steps where the ablated agent diverged

Selections: full / none / f1>=tau / random (same size, from features that
fire) / explained (final verdict) / core (fewest features by single-ablation
importance reaching --core_frac of the full fve). Evaluation uses held-out
(odd) episodes.

--units runs the same selections on the GRU's own gate units (interpreted with
circuit/rl_task/neuron_baseline.py stores): the real gate with ablated units.
A candidate unit counts as explained if either its high or low end is.

    python -m circuit.rl_task.explained_fidelity --rnn RNN.ckpt --traces rl_features/rl_trace_sequences \\
        --transcoders reset=R.ckpt update=U.ckpt candidate=C.ckpt \\
        --feature_results rl_features/rl_autointerp --feature_store STORE --out fidelity.json
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import StackDataset

from circuit.copy_task.explained_fidelity import (GATES, Ablated, check_fingerprints, components, core_set,
                                                  f1_bands, fve, load_f1, load_transcoder)
from training.splice_eval import build_task_rnn

torch.serialization.add_safe_globals([StackDataset])
PAD = 5
KEYS = ("inputs", "h_prevs", "r_ts", "z_ts", "h_new_ts")
DIVERGED = 1e3          # |readout| above this counts as diverged (original readouts are O(1-10))


def load_eval(trace_dir, max_windows, seed=0):
    """Windows of held-out (odd) episodes, padding stripped, grouped by length."""
    rng = np.random.default_rng(seed)
    per_len = {}
    paths = sorted(glob.glob(os.path.join(trace_dir, "*_rl.pt")))
    cap = max_windows // len(paths) + 1
    for path in paths:
        cols = torch.load(path, map_location="cpu").datasets
        test = np.nonzero(cols["seq_id"].numpy() % 2 == 1)[0]
        rows = torch.from_numpy(np.sort(rng.choice(test, size=min(cap, len(test)), replace=False)))
        x = cols["inputs"][rows]
        pad = (x == PAD).flatten(2).all(-1).sum(-1)
        for n_pad in pad.unique().tolist():
            sel = rows[pad == n_pad]
            g = per_len.setdefault(n_pad, {k: [] for k in KEYS + ("ep_trial0",)})
            for k in KEYS:
                g[k].append(cols[k][sel][:, n_pad:].float())
            # episode trial of the window's first (unpadded) trial
            g["ep_trial0"].append(cols["trial_in_episode"][sel] - (cols["core_trial"][sel] - n_pad))
        del cols
    groups = []
    for n_pad, g in sorted(per_len.items()):
        ep_trial0 = torch.cat(g.pop("ep_trial0"))
        g = {k: torch.cat(v).flatten(1, 2) for k, v in g.items()}       # (n, steps, ·)
        g["h0"] = g["h_prevs"][:, 0]
        g["ep_trial0"] = ep_trial0
        groups.append(g)
    n = sum(g["inputs"].shape[0] for g in groups)
    print(f"eval: {n} windows of held-out episodes, {sum(g['inputs'].shape[0] * g['inputs'].shape[1] for g in groups)} steps")
    return groups


def teacher_forced(groups, gate):
    xs, ys = [], []
    for g in groups:
        h, x = g["h_prevs"], g["inputs"]
        state = g["r_ts"] * h if gate == "candidate" else h
        xs.append(torch.cat([state, x], -1).flatten(0, 1))
        ys.append(g[GATES[gate][0]].flatten(0, 1))
    return torch.cat(xs), torch.cat(ys)


@torch.no_grad()
def run(rnn, x, h0, gate=None, fn=None):
    """Readout [3 policy logits, value] at every step, gate replaced by fn (None: original)."""
    gru, readout = rnn.layers[0], rnn.layers[-1]
    h, out = h0, []
    for t in range(x.shape[1]):
        xt = x[:, t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        if gate == "reset":
            r = fn(torch.cat([h, xt], -1), r)
        if gate == "update":
            z = fn(torch.cat([h, xt], -1), z)
        n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        if gate == "candidate":
            n = fn(torch.cat([r * h, xt], -1), n)
        h = (1 - z) * h + z * n
        out.append(readout(h))
    return torch.stack(out, 1)


def run_all(rnn, g, gate=None, fn=None, chunk=4096):
    return torch.cat([run(rnn, g["inputs"][i:i + chunk], g["h0"][i:i + chunk], gate, fn)
                      for i in range(0, len(g["inputs"]), chunk)])


@torch.no_grad()
def splice(rnn, groups, refs, gate, fn):
    return behaviour_metrics([run_all(rnn, g, gate, fn) for g in groups], refs)


def behaviour_metrics(outs, refs):
    """Splice metrics of readouts `outs` against the original GRU's `refs` (lists over length groups)."""
    agree = n_go = sw_agree = n_sw = kl = 0.0
    v_abs, n_v, nonfinite, n_steps = 0.0, 0, 0, 0
    ref_values = []
    for out, ref in zip(outs, refs):
        # diverged: non-finite, or blown up (the update / candidate transcoders are unbounded)
        finite = torch.isfinite(out).all(-1) & (out.abs() < DIVERGED).all(-1)
        go = torch.arange(out.shape[1]) % 3 == 1
        choice = lambda o: o[:, go, 1:3].argmax(-1)                     # 0 a1, 1 a2
        rc, oc = choice(ref), choice(out)
        ok = finite[:, go]
        agree += ((rc == oc) & ok).sum().item(); n_go += rc.numel()
        switch = torch.zeros_like(rc, dtype=torch.bool)
        switch[:, 1:] = rc[:, 1:] != rc[:, :-1]
        sw_agree += ((rc == oc) & ok & switch).sum().item(); n_sw += switch.sum().item()
        lp, lq = F.log_softmax(ref[..., :3], -1), F.log_softmax(out[..., :3], -1)
        step_kl = (lp.exp() * (lp - lq)).sum(-1)[:, go]
        kl += step_kl[ok].sum().item()
        v_abs += (ref[..., 3] - out[..., 3]).abs()[finite].sum().item(); n_v += finite.sum().item()
        ref_values.append(ref[..., 3].flatten())
        nonfinite += (~finite).sum().item(); n_steps += finite.numel()
    std = torch.cat(ref_values).std().item()
    return {"go_agreement": agree / n_go, "switch_agreement": sw_agree / max(n_sw, 1), "n_switch": int(n_sw),
            "kl_go": kl / max(n_go, 1), "value_nmae": v_abs / max(n_v, 1) / max(std, 1e-12),
            "nonfinite_frac": nonfinite / max(n_steps, 1)}


def evaluate(rnn, groups, refs, X, Y, gate, f1, alive, importance, make_fn, taus, n_random, core_frac, seed,
             extra_sets=(), source="features"):
    rows, rng, width = [], np.random.default_rng(seed), len(f1)

    def add(selection, keep, tau=None, rep=None):
        fn = make_fn(keep)
        row = {"gate": gate, "source": source, "selection": selection, "tau": tau, "rep": rep,
               "n_kept": int(keep.sum()), "n_total": width, "fve": fve(X, Y, fn)}
        row.update(splice(rnn, groups, refs, gate, fn))
        rows.append(row)
        print(f"  {source:>14} {selection:>10}{'' if tau is None else f' {tau}':>5}{'' if rep is None else f' #{rep}':>4}  kept "
              f"{row['n_kept']:4d}/{width}  fve {row['fve']:.3f}  go {row['go_agreement']:.3f}  switch "
              f"{row['switch_agreement']:.3f}  kl_go {row['kl_go']:.4f}  value nmae {row['value_nmae']:.3f}"
              f"  diverged {row['nonfinite_frac']:.3f}", flush=True)
        return row

    full = add("full", torch.ones(width, dtype=torch.bool))
    add("none", torch.zeros(width, dtype=torch.bool))
    pool = torch.nonzero(alive).flatten().numpy()
    for tau in taus:
        keep = f1 >= tau
        add("f1", keep, tau)
        for rep in range(n_random):
            rand = torch.zeros(width, dtype=torch.bool)
            rand[rng.choice(pool, size=min(int(keep.sum()), len(pool)), replace=False)] = True
            add("random", rand, tau, rep)
    for name, keep in extra_sets:
        add(name, keep)
    core = core_set(X, Y, importance, make_fn, core_frac * full["fve"])
    add("core", core)
    summary = {"gate": gate, "source": source, "core_frac": core_frac, "full_fve": full["fve"],
               "core_ids": torch.nonzero(core).flatten().tolist(),
               "core": f1_bands(f1[core].numpy()), "alive": f1_bands(f1[alive].numpy()),
               "core_importance_share": float(importance[core].sum() / importance.sum())}
    print(f"  core set: {summary['core']}")
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--traces", required=True, help="directory of *_rl.pt trace-window files")
    ap.add_argument("--transcoders", nargs="*", default=[], metavar="GATE=CKPT", help="reset / update / candidate")
    ap.add_argument("--feature_results")
    ap.add_argument("--feature_store", help="feature store, to check transcoder fingerprints")
    ap.add_argument("--units", nargs="*", default=[], metavar="NAME=RESULTS",
                    help="autointerp results for GRU units (circuit/rl_task/neuron_baseline.py stores)")
    ap.add_argument("--gates", nargs="*", default=list(GATES), choices=list(GATES))
    ap.add_argument("--max_windows", type=int, default=20000)
    ap.add_argument("--taus", nargs="*", type=float, default=[0.8, 0.5, 0.2])
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--core_frac", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    transcoders = dict(s.split("=", 1) for s in args.transcoders)
    if args.feature_store:
        check_fingerprints(args.feature_store, {"rnn": args.rnn, **transcoders})
    rnn = build_task_rnn("rl", input_size=8, hidden_size=48)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    rnn.eval()
    groups = load_eval(args.traces, args.max_windows, args.seed)
    refs = [run_all(rnn, g) for g in groups]

    units = dict(s.split("=", 1) for s in args.units)
    rows, summaries = [], []
    for gate in args.gates:
        X, Y = teacher_forced(groups, gate)
        if gate in transcoders:
            tc = load_transcoder(transcoders[gate])
            means, alive, importance = components(X, Y, tc)
            f1, _ = load_f1(args.feature_results, [GATES[gate][1]], tc.n_feats)
            explained = explained_mask(args.feature_results, [GATES[gate][1]], tc.n_feats)
            print(f"{gate}: {len(X)} steps, {int(alive.sum())} features fire, {int(explained.sum())} explained", flush=True)
            make_fn = lambda keep, tc=tc, means=means: Ablated(keep, means, tc)
            r, s = evaluate(rnn, groups, refs, X, Y, gate, f1, alive, importance, make_fn, args.taus, args.n_random,
                            args.core_frac, args.seed, extra_sets=[("explained", explained)])
            rows += r
            summaries.append(s)
        for name, results in units.items():
            unit_gates = GATES[gate][2]
            if not any(os.path.exists(os.path.join(results, f"{g}_0.json")) for g in unit_gates):
                print(f"{gate}: no unit results in {results}, skipped", flush=True)
                continue
            # the real gate with ablated units; a candidate unit is explained if either end is (favours units)
            means, alive, importance = components(X, Y)
            f1, _ = load_f1(results, unit_gates, Y.shape[1])
            explained = explained_mask(results, unit_gates, Y.shape[1])
            make_fn = lambda keep, means=means: Ablated(keep, means)
            r, s = evaluate(rnn, groups, refs, X, Y, gate, f1, alive, importance, make_fn, args.taus, args.n_random,
                            args.core_frac, args.seed, extra_sets=[("explained", explained)], source=f"units {name}")
            rows += r
            summaries.append(s)
        json.dump({"args": vars(args), "rows": rows, "core": summaries}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


def explained_mask(results, result_gates, width):
    """Components with final verdict "explained" (any of the result gates, e.g. either candidate-unit end)."""
    out = torch.zeros(width, dtype=torch.bool)
    for rg in result_gates:
        for i in range(width):
            p = os.path.join(results, f"{rg}_{i}.json")
            if os.path.exists(p):
                r = json.load(open(p))
                out[i] |= (r.get("final_verdict") or r.get("verdict")) == "explained"
    return out


if __name__ == "__main__":
    main()
