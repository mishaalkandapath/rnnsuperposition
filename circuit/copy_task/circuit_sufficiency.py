"""Is the feature circuit enough, and needed, to carry token i to output L + i? (activation patching, real GRU)

For each sequence and slot i: an original run (token v in slot i) and a run with v
swapped for another token (feedback changed to match). The circuit = this sequence's
top-k candidate features per step, at steps i .. L+i, ranked by |exact contrastive
flow| (circuit.copy_task.feature_circuits.change_flow), so features driving either
token are in it. Patching sets a feature to a donor run's activation at the same step
(n += sum_f (a_f,donor - a_f) d_f). Four modes:

  sufficiency, free     swapped sequence; circuit set to its original values, everything else
                        recomputed. Does output i become v again?
  sufficiency, clamped  original sequence; circuit kept at original values, every other feature
                        clamped to its swapped-run value. Does output i stay v?
  necessity, free       original sequence; circuit set to its swapped-run values, everything else
                        recomputed. Does output i stop being v?
  necessity, clamped    original sequence; circuit set to swapped-run values, every other feature
                        clamped to its original value. Does output i stop being v?
Controls: k random features per step among those active in either run but outside the
circuit. "all" = every candidate feature (gates and the transcoder error are not patched).

    python -m circuit.copy_task.circuit_sufficiency --rnn RNN.ckpt --traces T4.pt \\
        --transcoders reset=R.ckpt update=U.ckpt candidate=C.ckpt --out sufficiency.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import StackDataset

from circuit.copy_task.circuit_tests import clean_acts, run
from circuit.copy_task.copy_circuits import gru_trace, swap_token
from circuit.copy_task.explained_fidelity import load_transcoder
from circuit.copy_task.feature_circuits import change_flow
from circuit.edge_att import CircuitTracer
from training.splice_eval import build_task_rnn

torch.serialization.add_safe_globals([StackDataset])


def patch_hook(tc, donor, mask):
    """Candidate hook: at each step, features in mask (N, T, F) take the donor's activation."""
    D = tc.features_to_outputs.weight

    def hook(t, h, xt, r, z, n):
        m = mask[:, t]
        if not m.any():
            return r, z, n
        a = tc.act(tc.input_to_features(torch.cat([r * h, xt], -1)))
        return r, z, n + ((donor[:, t] - a) * m) @ D.T
    return hook


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--transcoders", nargs=3, required=True, metavar="GATE=PATH")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 40])
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    paths = dict(item.split("=", 1) for item in args.transcoders)
    rnn = build_task_rnn("copy", 31, 128)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    tcs = {k: load_transcoder(paths[k]) for k in ("reset", "update", "candidate")}
    tc = tcs["candidate"]
    tracer = CircuitTracer(rnn, tcs["update"], tc, device="cpu", reset_transcoder=tcs["reset"])
    data = torch.load(args.traces, map_location="cpu", mmap=True).datasets
    L = data["outputs"].shape[1]
    T, F = 2 * L, tc.n_feats
    tokens = data["inputs"][:, :L, :-1].argmax(-1)
    rng = np.random.default_rng(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    distinct = np.flatnonzero([len(set(r)) == L for r in tokens.tolist()])
    rows = torch.from_numpy(np.sort(rng.choice(distinct, size=min(args.n, len(distinct)), replace=False)))
    x, orig = data["inputs"][rows].float(), tokens[rows]
    N, ar = len(x), torch.arange(len(rows))
    present = torch.zeros(N, 30, dtype=torch.bool).scatter_(1, orig, True)

    results = {}
    start = time.time()
    for i in range(L):
        new = torch.multinomial((~present).float(), 1, generator=g)[:, 0]
        xs = torch.stack([swap_token(x[s], L, i, int(new[s])) for s in range(N)])
        act_o, act_s = clean_acts(rnn, x, L, tcs)["candidate"], clean_acts(rnn, xs, L, tcs)["candidate"]
        steps = list(range(i, L + i + 1))
        # each sequence's features at each step, ranked by contrastive flow
        ranked = torch.full((N, T, F), -1, dtype=torch.long)
        n_ranked = torch.zeros(N, T, dtype=torch.long)
        flow_cover = {k: 0.0 for k in args.ks}                          # share of |flow| in the top k
        for s in range(N):
            ro = tracer.run_forward_pass(gru_trace(rnn, x[s]))
            rs = tracer.run_forward_pass(gru_trace(rnn, xs[s]))
            fl = change_flow(tracer, ro, rs, L, orig[s].tolist(), i)
            for t in steps:
                # by |flow|: features driving either token (the competitor's features count as circuit too)
                items = sorted(((abs(v), f) for (tt, f), v in fl["nodes"].items() if tt == t and abs(v) > 1e-6), reverse=True)
                n_ranked[s, t] = len(items)
                if items:
                    ranked[s, t, :len(items)] = torch.tensor([f for _, f in items], dtype=torch.long)
                    tot = sum(v for v, _ in items)
                    for k in args.ks:
                        flow_cover[k] += sum(v for v, _ in items[:k]) / tot / (N * len(steps))
        active = (act_o > 0) | (act_s > 0)                              # (N, T, F)

        def circuit_mask(k):
            m = torch.zeros(N, T, F, dtype=torch.bool)
            for t in steps:
                idx = ranked[:, t, :k]
                ok = idx >= 0
                m[ar[:, None].expand_as(idx)[ok], t, idx[ok]] = True
            return m

        def random_mask(k, exclude):
            """k random features per step among those active in either run, outside `exclude` (the circuit)."""
            m = torch.zeros(N, T, F, dtype=torch.bool)
            for t in steps:
                ok = active[:, t] & ~exclude[:, t]
                score = torch.rand(N, F, generator=g) + ok.float()
                top = torch.topk(score, k, dim=1).indices
                m[ar[:, None].expand_as(top), t, top] = ok.gather(1, top)
            return m

        all_mask = torch.zeros(N, T, F, dtype=torch.bool)
        all_mask[:, steps] = True
        others = [j for j in range(L) if j != i]

        def evaluate(name, mask):
            rest = all_mask & ~mask
            modes = {   # name: (input sequence, donor activations, which features are set to the donor)
                "sufficiency_free": (xs, act_o, mask),
                "sufficiency_clamped": (x, torch.where(mask, act_o, act_s), all_mask),
                "necessity_free": (x, act_s, mask),
                "necessity_clamped": (x, torch.where(mask, act_s, act_o), all_mask)}
            out = {"features_per_step": float(mask[:, steps].sum(-1).float().mean())}
            for mode, (xx, donor, m) in modes.items():
                lg = run(rnn, xx, L, patch_hook(tc, donor, m))
                pred = lg[:, i].argmax(-1)
                out[f"{mode}_output_is_v"] = float((pred == orig[:, i]).float().mean())
                out[f"{mode}_output_is_swapped"] = float((pred == new).float().mean())
                out[f"{mode}_p_v"] = float(lg.softmax(-1)[ar, i, orig[:, i]].mean())
                if mode.endswith("clamped") or mode == "necessity_free":
                    out[f"{mode}_other_outputs_acc"] = float((lg[:, others].argmax(-1) == orig[:, others]).float().mean())
            return out

        res = {"empty circuit": evaluate("none", torch.zeros(N, T, F, dtype=torch.bool))}
        for k in args.ks:
            cm = circuit_mask(k)
            res[f"circuit top {k}"] = {**evaluate(f"circuit {k}", cm), "share_of_abs_flow": flow_cover[k]}
            rnd = [evaluate(f"random {k}", random_mask(k, cm)) for _ in range(args.n_random)]
            res[f"random {k} (outside the circuit)"] = {key: float(np.mean([r_[key] for r_ in rnd])) for key in rnd[0]}
        res["all candidate features"] = evaluate("all", all_mask)
        res["n_ranked_per_step"] = float(n_ranked[:, steps].float().mean())
        results[i] = res
        print(f"slot {i} ({time.time() - start:.0f}s): ranked features per step {res['n_ranked_per_step']:.1f}")
        for key, v in res.items():
            if isinstance(v, dict):
                print(f"   {key:>34}: {v['features_per_step']:5.1f}/step | output = v: suff free {v['sufficiency_free_output_is_v']:.2f}, "
                      f"suff clamped {v['sufficiency_clamped_output_is_v']:.2f}, nec free {v['necessity_free_output_is_v']:.2f}, "
                      f"nec clamped {v['necessity_clamped_output_is_v']:.2f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "L": L, "slots": results}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
