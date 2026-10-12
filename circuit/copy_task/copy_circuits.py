"""Copy circuits: where does each output's token come from, and what keeps it?

For every sequence of length L (distinct tokens) and every output position j
(output step t = L + j), build the frozen-gate attribution graph of
circuit.edge_att.CircuitTracer (all three transcoders) and decompose three
targets, each a mean-centred logit at step t, exactly:

  correct    orig[j], the token being copied;
  popped     orig[j-1], the token just emitted and fed back (j >= 1);
  next       orig[j+1], the token still to come (j < L - 1).

Each target is split into:

  direct     writes into h_t that the readout reads directly, by the step
             s that wrote them (candidate features, errors, decoder bias).
             Says which step's features write the logit.
  sources    the target as a sum over sources through all paths, by step:
             inputs split into identity (x_s minus the mean token) and
             presence (the mean token; the delimiter at t = L), candidate
             errors, encoder and decoder biases. Identity at step j is the
             copied token. Gates are frozen constants here, as attention is
             in the paper.
  features   total effect of each active candidate feature: value * D, with D
             the derivative of the target through all paths (= the drop in
             the target if it were removed, JumpReLU masks fixed).
  gates      exact effect of each update/reset gate piece (feature, bias,
             error) on the target (CircuitTracer.gate_feature_ranking).

The checks: sources sum to the target value; error share is the analogue of
the paper's replacement score. Results are aggregated over sequences into
step x output tables and per-(feature, step) mean effects.

Linear attributions credit AND-like features additively, so per-step credit
is not dependence; swap_tests measures dependence in the real GRU.

    python -m circuit.copy_task.copy_circuits --rnn RNN.ckpt \\
        --traces features/copy_transcoder_v2_1of5/1M_128_high_seq{3,4}.pt \\
        --transcoders reset=R.ckpt update=U.ckpt candidate=C.ckpt \\
        --feature_results features/<run>_autointerp --n 500 --out circuits.json
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
from circuit.edge_att import CircuitTracer
from circuit.copy_task.explained_fidelity import load_transcoder, run
from training.splice_eval import build_task_rnn

torch.serialization.add_safe_globals([StackDataset])
GATE_NAMES = {"f_n": "hidden", "f_z": "update", "f_r": "reset"}
SECTIONS = ["graphs", "swaps", "contrast"]


def step_label(s, L):
    """in<i> while reading the sequence, DEL at t = L, out<k> where output k is emitted after DEL."""
    return f"in{s}" if s < L else "DEL" if s == L else f"out{s - L}"


def input_reference(T, L, X):
    """Reference input per step: the mean token while tokens are read or fed
    back, the delimiter itself at t = L. Input = reference (presence) + identity."""
    ref = torch.zeros(T, X)
    ref[:, :X - 1] = 1.0 / (X - 1)
    ref[L] = 0.0
    ref[L, X - 1] = 1.0
    return ref


@torch.no_grad()
def decompose(tracer, acts, L, orig=None):
    """Exact decomposition of every output's target logit (see module docstring).

    Everything target-independent (graph, local derivatives, carries, reads)
    is computed once; the sensitivities D of all targets are one solve.
    orig: the token sequence the targets refer to (default: this run's input),
    so a swapped run can be decomposed for the original run's targets.
    """
    nodes = tracer._build_nodes(acts)
    edges = tracer._compute_edges(nodes, acts)
    T, H = acts["h_ts"].shape
    z = acts["z_ts"]
    W_c = tracer.W_o - tracer.W_o.mean(0, keepdim=True)
    tc = tracer.hidden_transcoder
    b_enc, b_dec = tc.input_to_features.bias, tc.features_to_outputs.bias

    scalar = [n for n in nodes if n.node_type in ("feature", "output")]
    pos = {n.name: i for i, n in enumerate(scalar)}
    feats = [i for i, n in enumerate(scalar) if n.node_type == "feature"]
    feat_step = torch.tensor([scalar[i].timestep for i in feats])
    feat_idx = torch.tensor([scalar[i].feature_idx for i in feats])
    feat_val = acts["f_n"][feat_step, feat_idx]

    # Local derivatives between scalar nodes, and edges from errors into them.
    J = torch.zeros(len(scalar), len(scalar))
    err_edges = []                                               # (error step, dst pos, weight)
    for (s_name, d_name), w in edges.items():
        if d_name not in pos:
            continue
        if s_name in pos:
            J[pos[d_name], pos[s_name]] = w / float(acts["f_n"][scalar[pos[s_name]].timestep, scalar[pos[s_name]].feature_idx])
        elif s_name.startswith("e_n_"):
            err_edges.append((int(s_name.split("_")[2]), pos[d_name], w))

    # Target columns (kind, output j, token): the correct token, the token just
    # emitted (popped, j >= 1) and the one still to come (next, j < L - 1).
    if orig is None:
        orig = [int(acts["inputs"][j].argmax()) for j in range(L)]
    cols = [(kind, j, orig[j + d]) for j in range(L)
            for kind, d in (("correct", 0), ("popped", -1), ("next", 1)) if 0 <= j + d < L]
    C = len(cols)
    targets = [pos[f"o_{L + j}_{k}"] for _, j, k in cols]
    W = torch.zeros(len(scalar), C)
    W[targets, torch.arange(C)] = 1.0
    D = torch.linalg.solve(torch.eye(len(scalar)) - J.T, W)     # (n_scalar, C)
    D_f = D[feats]                                               # (n_feats, C)

    # Input contributions, split into identity (x - ref) and presence (ref).
    ref = input_reference(T, L, acts["inputs"].shape[1])
    W_x = tracer.W_n_x[feat_idx]                                 # (n_feats, X)
    identity = (W_x * (acts["inputs"][feat_step] - ref[feat_step])).sum(1)
    presence = (W_x * ref[feat_step]).sum(1)

    def by_step(per_feat):                                       # (n_feats, C) -> (T, C)
        return torch.zeros(T, C).index_add_(0, feat_step, per_feat)

    sources = {"identity": by_step(identity[:, None] * D_f), "presence": by_step(presence[:, None] * D_f),
               "enc_bias": by_step(b_enc[feat_idx][:, None] * D_f), "error": torch.zeros(T, C)}
    for s, d, w in err_edges:
        sources["error"][s] += w * D[d]

    # Decoder-bias writes z_s * b_dec, read by every scalar node through the carry.
    tau_of, read_vec = [], []
    for n in scalar:
        tau, read, gate = tracer._read_terms(n, acts)
        tau_of.append(tau + 1)
        read_vec.append(read if gate is None else gate * read)
    reads = torch.zeros(T + 1, H, C).index_add_(0, torch.tensor(tau_of),
                                                torch.stack(read_vec)[:, :, None] * D[:, None, :])
    carry = torch.ones(T, T, H)                                  # carry[s, tau] = C(s, tau)
    for s in range(T):
        for tau in range(s + 1, T):
            carry[s, tau] = carry[s, tau - 1] * (1 - z[tau])
    sources["dec_bias"] = torch.zeros(T, C)
    direct = torch.zeros(T, C)
    for s in range(T):
        writes = carry[s, s:] * (z[s] * b_dec)                   # (T - s, H): bias write as seen at tau >= s
        sources["dec_bias"][s] = torch.einsum("th,thl->l", writes, reads[s + 1:])
        for c, (_, j, k) in enumerate(cols):
            if L + j >= s:
                direct[s, c] += float(W_c[k] @ writes[L + j - s])
    column = {t: c for c, t in enumerate(targets)}
    for (s_name, d_name), w in edges.items():
        if d_name in pos and pos[d_name] in column and s_name.startswith(("f_n_", "e_n_")):
            direct[int(s_name.split("_")[2]), column[pos[d_name]]] += w

    effects = feat_val[:, None] * D_f                            # (n_feats, C)
    out = {}
    for c, (kind, j, k) in enumerate(cols):
        value = float(acts["h_ts"][L + j] @ W_c[k])
        src = {key: v[:, c].numpy() for key, v in sources.items()}
        out[(kind, j)] = {
            "value": value, "check": float(sum(v.sum() for v in src.values()) - value),
            "direct": direct[:, c].numpy(), "sources": src,
            "features": {(int(f), int(s)): float(e) for f, s, e in zip(feat_idx, feat_step, effects[:, c])},
            "gates": tracer.gate_scores(acts, scalar, D[:, c]),
            "correct": int(acts["logits"][j].argmax()) == orig[j]}
    return out


@torch.no_grad()
def swap_tests(rnn, x, L, seed=0):
    """Real-GRU counterfactuals: replace the token at position i by another.

    full        the fed-back outputs are changed to match (a consistent sequence);
    input_only  the fed-back outputs keep the original token, so memory and
                feedback disagree after i.
    Returns, per mode and swapped position i (rows) and output j (columns):
    the mean change in output j's logit of orig[j] (dlogit), the same as a
    fraction of the original logit (dlogit_frac), the probability of orig[j]
    (p_orig) and accuracy against the new sequence; for input_only also what
    output i + 1 emits instead.
    """
    g = torch.Generator().manual_seed(seed)
    N, ar = len(x), torch.arange(len(x))
    orig = x[:, :L, :-1].argmax(-1)
    centred = lambda out: out - out.mean(-1, keepdim=True)
    base = centred(run(rnn, x)).gather(2, orig[:, :, None])[..., 0]
    out = {}
    for mode in ("full", "input_only"):
        dlogit, frac, p_orig, acc, emitted = [], [], [], [], []
        for i in range(L):
            new = (orig[:, i] + torch.randint(1, x.shape[-1] - 1, (N,), generator=g)) % (x.shape[-1] - 1)
            x2 = x.clone()
            x2[:, i] = 0
            x2[ar, i, new] = 1
            if mode == "full" and i + 1 < L:
                x2[:, L + 1 + i] = 0
                x2[ar, L + 1 + i, new] = 1
            logits = centred(run(rnn, x2))
            want = orig.clone()
            want[:, i] = new
            change = logits.gather(2, orig[:, :, None])[..., 0] - base
            dlogit.append(change.mean(0).tolist())
            frac.append((change / base).mean(0).tolist())
            p_orig.append(logits.softmax(-1).gather(2, orig[:, :, None])[..., 0].mean(0).tolist())
            acc.append((logits.argmax(-1) == want).float().mean(0).tolist())
            if mode == "input_only" and i + 1 < L:
                p = logits[:, i + 1].argmax(-1)
                in_seq = (p[:, None] == want).any(1)
                emitted.append({"orig[i+1] (correct)": float((p == orig[:, i + 1]).float().mean()),
                                "new token at i": float((p == new).float().mean()),
                                "other token in sequence": float((in_seq & (p != orig[:, i + 1]) & (p != new)).float().mean()),
                                "not in sequence": float((~in_seq).float().mean())})
        out[mode] = {"dlogit": dlogit, "dlogit_frac": frac, "p_orig": p_orig, "accuracy": acc,
                     **({"emitted_after": emitted} if emitted else {})}
    return out


@torch.no_grad()
def gru_trace(rnn, x):
    """Trace of the real GRU on one sequence x (T, X), in the trace-file layout."""
    _, _, r, z, n, h_prev = rnn.layers[0](x[None], record_gates=True)
    return {"inputs": x, "h_prevs": h_prev[0], "r_ts": r[0], "z_ts": z[0], "h_new_ts": n[0]}


def swap_token(x, L, i, new):
    """x with the token at position i replaced by `new`, feedback changed to match."""
    x2 = x.clone()
    for t in [i] + ([L + 1 + i] if i + 1 < L else []):
        x2[t] = 0
        x2[t, new] = 1
    return x2


@torch.no_grad()
def contrast(tracer, rnn, cols, rows, L, labels, top, seed=0):
    """Contrastive graphs: decompose the original run and a run with token i
    swapped (feedback changed to match) for the same targets, the logits of
    the original tokens, and take swapped minus original.

    Both decompositions are exact, so the per-step differences sum exactly to
    the logit change of the real GRU. Credit that does not depend on which
    token is at i (e.g. "slot i is filled") is in both runs and cancels.
    Returns per (swapped position i, output j): the mean change in the target
    and its split by step (all sources, identity only, direct writes), and the
    features whose effect changed most.
    """
    rng = np.random.default_rng(seed)
    T, V = 2 * L, cols["inputs"].shape[-1] - 1
    sums = defaultdict(lambda: {"dvalue": 0.0, "total": np.zeros(T), "identity": np.zeros(T),
                                "direct": np.zeros(T), "features": defaultdict(float)})
    for row in rows:
        x = cols["inputs"][row].float()
        orig = [int(x[j].argmax()) for j in range(L)]
        base = decompose(tracer, tracer.run_forward_pass(gru_trace(rnn, x)), L)
        for i in range(L):
            new = int(rng.choice([v for v in range(V) if v not in orig]))
            swapped = decompose(tracer, tracer.run_forward_pass(gru_trace(rnn, swap_token(x, L, i, new))), L, orig)
            for j in range(L):
                a, b, s = base[("correct", j)], swapped[("correct", j)], sums[(i, j)]
                s["dvalue"] += b["value"] - a["value"]
                s["total"] += sum(b["sources"].values()) - sum(a["sources"].values())
                s["identity"] += b["sources"]["identity"] - a["sources"]["identity"]
                s["direct"] += b["direct"] - a["direct"]
                for key in set(a["features"]) | set(b["features"]):
                    s["features"][key] += b["features"].get(key, 0.0) - a["features"].get(key, 0.0)
    n = len(rows)
    cells = []
    for (i, j), s in sorted(sums.items()):
        ranked = sorted(s["features"], key=lambda key: -abs(s["features"][key]))[:top]
        cells.append({"i": i, "j": j, "dvalue": s["dvalue"] / n,
                      "total_by_step": (s["total"] / n).tolist(), "identity_by_step": (s["identity"] / n).tolist(),
                      "direct_by_step": (s["direct"] / n).tolist(),
                      "top_features": [{"feature": f, "step": step_label(t, L), "mean_change": s["features"][(f, t)] / n,
                                        **({"label": labels[("hidden", f)][0]} if ("hidden", f) in labels else {})}
                                       for f, t in ranked]})
    return {"n_sequences": n, "cells": cells}


def gate_key(name, where):
    """('update', u, feature index or 'bias'/'error') for a gate piece."""
    gate, u = where.split("@")
    if name.startswith(("f_z_", "f_r_")):
        return gate, int(u), int(name.split("_")[-1])
    return gate, int(u), "bias" if "bias" in name else "error"


def load_labels(results_dir):
    labels = {}
    if not results_dir:
        return labels
    for fname in os.listdir(results_dir):
        if fname.endswith(".json") and fname.split("_")[0] in ("hidden", "update", "reset"):
            d = json.load(open(os.path.join(results_dir, fname)))
            labels[(d.get("gate"), d.get("feature"))] = (d.get("label"), d.get("final_verdict"))
    return labels


def summarise(L, records, labels, top):
    """Mean tables over sequences for one length."""
    T, n = 2 * L, len(records)
    out = {"L": L, "n_sequences": n, "steps": [step_label(s, L) for s in range(T)], "outputs": []}
    for kind, j in records[0]:
        rs = [r[(kind, j)] for r in records]
        mean = lambda key: (sum(r[key] for r in rs) / n).tolist()
        sources = {key: (sum(r["sources"][key] for r in rs) / n).tolist() for key in rs[0]["sources"]}
        abs_total = sum(sum(np.abs(v).sum() for v in r["sources"].values()) for r in rs)
        abs_error = sum(np.abs(r["sources"]["error"]).sum() for r in rs)

        feats = defaultdict(float)
        fired = defaultdict(int)
        for r in rs:
            for key, v in r["features"].items():
                feats[key] += v / n
                fired[key] += 1
        ranked = sorted(feats, key=lambda key: -abs(feats[key]))[:top]
        gates = defaultdict(float)
        gate_fired = defaultdict(int)
        for r in rs:
            for name, score, where in r["gates"]:
                gates[gate_key(name, where)] += score / n
                gate_fired[gate_key(name, where)] += 1
        gate_steps = defaultdict(lambda: defaultdict(float))     # gate@step -> features / bias / error
        for (gate, u, piece), v in gates.items():
            gate_steps[f"{gate}@{step_label(u, L)}"][piece if isinstance(piece, str) else "features"] += v
        ranked_gates = sorted((key for key in gates if isinstance(key[2], int)),
                              key=lambda key: -abs(gates[key]))[:top]

        def label(gate, f):
            lab = labels.get((gate, f))
            return {"label": lab[0], "verdict": lab[1]} if lab else {}

        out["outputs"].append({
            "kind": kind, "j": j, "step": step_label(L + j, L),
            "target_value": float(np.mean([r["value"] for r in rs])),
            "max_abs_check": float(max(abs(r["check"]) for r in rs)),
            "accuracy": float(np.mean([r["correct"] for r in rs])),
            "error_share_abs": float(abs_error / abs_total),
            "direct_by_step": mean("direct"),
            "sources_by_step": sources,
            "top_features": [{"feature": f, "step": step_label(s, L), "mean_effect": feats[(f, s)],
                              "fired_frac": fired[(f, s)] / n, "effect_when_fired": feats[(f, s)] * n / fired[(f, s)],
                              **label("hidden", f)}
                             for f, s in ranked],
            "gate_by_step": {key: dict(v) for key, v in gate_steps.items()},
            "top_gate_features": [{"gate": g, "step": step_label(u, L), "feature": p, "mean_effect": gates[(g, u, p)],
                                   "fired_frac": gate_fired[(g, u, p)] / n,
                                   "effect_when_fired": gates[(g, u, p)] * n / gate_fired[(g, u, p)], **label(g, p)}
                                  for g, u, p in ranked_gates],
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rnn", required=True)
    parser.add_argument("--traces", nargs="+", required=True, help="one trace file per length")
    parser.add_argument("--transcoders", nargs=3, required=True, metavar="GATE=PATH")
    parser.add_argument("--feature_results", default=None, help="autointerp results, for labels")
    parser.add_argument("--n", type=int, default=500, help="sequences per length")
    parser.add_argument("--n_swap", type=int, default=5000, help="sequences per length for the swap tests")
    parser.add_argument("--n_contrast", type=int, default=300, help="sequences per length for contrastive graphs")
    parser.add_argument("--sections", nargs="+", default=SECTIONS, choices=SECTIONS,
                        help="sections to (re)compute; others are kept from an existing --out JSON")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = dict(item.split("=", 1) for item in args.transcoders)
    rnn = build_task_rnn("copy", 31, 128)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    tracer = CircuitTracer(rnn, load_transcoder(paths["update"]), load_transcoder(paths["candidate"]),
                           device="cpu", reset_transcoder=load_transcoder(paths["reset"]))
    labels = load_labels(args.feature_results)
    rng = np.random.default_rng(args.seed)

    keep = set(args.sections) != set(SECTIONS) and os.path.exists(args.out)
    saved = json.load(open(args.out)) if keep else {"tokens": TOKENS, "lengths": []}
    results = []
    for path in args.traces:
        cols = torch.load(path, map_location="cpu", mmap=True).datasets
        L = cols["outputs"].shape[1]
        orig = cols["inputs"][:, :L].argmax(-1)
        # Distinct tokens only, so the popped / next targets are unambiguous.
        distinct = np.flatnonzero((orig[:, :, None] == orig[:, None, :]).sum((1, 2)).numpy() == L)
        rows = np.sort(rng.choice(distinct, size=min(args.n, len(distinct)), replace=False))
        summary = next((s for s in saved["lengths"] if s["L"] == L), {"L": L})
        if "graphs" in args.sections:
            records, start = [], time.time()
            for i, row in enumerate(rows):
                records.append(decompose(tracer, tracer.run_forward_pass({k: cols[k][row].float() for k in cols}), L))
                if (i + 1) % 100 == 0:
                    print(f"L={L}: {i + 1}/{len(rows)} sequences, {time.time() - start:.0f}s", flush=True)
            summary.update(summarise(L, records, labels, args.top))
            summary["rows"] = rows.tolist()
            for o in summary["outputs"]:
                print(f"L={L} output {o['j']} {o['kind']}: target {o['target_value']:.1f}, check {o['max_abs_check']:.1e}, "
                      f"acc {o['accuracy']:.3f}, error share {o['error_share_abs']:.3f}")
        if "swaps" in args.sections:
            summary["swaps"] = swap_tests(rnn, cols["inputs"][:args.n_swap].float(), L, args.seed)
            summary["swaps"]["n_sequences"] = min(args.n_swap, cols["inputs"].shape[0])
        if "contrast" in args.sections:
            start = time.time()
            summary["contrast"] = contrast(tracer, rnn, cols, rows[:args.n_contrast], L, labels, args.top, args.seed)
            print(f"L={L}: contrastive graphs on {summary['contrast']['n_sequences']} sequences, "
                  f"{time.time() - start:.0f}s", flush=True)
        results.append(summary)

    saved["lengths"] = results
    saved.setdefault("runs", []).append(vars(args))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(saved, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
