"""Feature-level copy circuits: which candidate features carry slot i's token
from its input step to its output step, and how they hand it on.

In the style of Ameisen et al. (2025): nodes are candidate features at time
steps, edges are their linear effects with the gates frozen (as attention is
frozen there), and features with one role are grouped into supernodes.

  selectivity  for every candidate feature f, step t and slot i, the fraction
               of f's activation variance at t explained by the token in slot i
               (eta^2, over all sequences of the trace file). Supernode
               "S{i}@t" = features with eta^2 >= --eta at t; the remaining
               active features at t are "context@t".
  flow         contrastive flow graph per slot i. For each sequence, the
               original and a run with token i swapped (feedback changed to
               match) are decomposed for the same target, the logit of the
               original token at output i. Every edge is weighted by its
               influence on the target, contribution x D(destination); these
               flows add up through each node (in = out = the node's effect).
               Flows are summed into supernodes, inputs ("input@t") and errors
               ("error@t"), and swapped-run flows are subtracted: what remains
               is the flow that carries which token is in slot i.
  ablate       real GRU, candidate output minus the mean-ablated contribution
               of one supernode at one step (n - sum_f (a_f - mean_f) d_f).
               Accuracy per output: carrying slot i should break only output i.
               Controls: the context features of that step, and random active
               features of the same count.

    python -m circuit.copy_task.feature_circuits --rnn RNN.ckpt --traces T4.pt \\
        --transcoders reset=R.ckpt update=U.ckpt candidate=C.ckpt \\
        --feature_results features/<run>_autointerp --out feature_circuits.json
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
from circuit.copy_task.copy_circuits import gru_trace, load_labels, step_label, swap_token
from circuit.edge_att import CircuitTracer
from circuit.copy_task.explained_fidelity import load_transcoder
from training.splice_eval import build_task_rnn

torch.serialization.add_safe_globals([StackDataset])


@torch.no_grad()
def candidate_acts(tc, cols):
    """Candidate feature activations (N, T, F) on the stored traces."""
    h, r, x = cols["h_prevs"].float(), cols["r_ts"].float(), cols["inputs"].float()
    return tc.act(tc.input_to_features(torch.cat([r * h, x], -1)))


@torch.no_grad()
def selectivity(acts, tokens, min_rate=0.01):
    """eta^2 (L, T, F): share of each feature's variance at step t explained by
    the token in slot i (features firing on < min_rate of sequences at t get 0),
    the firing rate (T, F), and the slot-token tables (L, V, T, F): the mean
    activation of each feature at step t given the token in slot i."""
    N, T, F = acts.shape
    L = tokens.shape[1]
    var = acts.var(0, unbiased=False)                                    # (T, F)
    rate = (acts > 0).float().mean(0)
    eta, tables = torch.zeros(L, T, F), torch.zeros(L, 30, T, F)
    for i in range(L):
        onehot = torch.nn.functional.one_hot(tokens[:, i], 30).float()  # (N, V)
        counts = onehot.sum(0).clamp_min(1)
        tables[i] = torch.einsum("nv,ntf->vtf", onehot, acts) / counts[:, None, None]
        between = torch.einsum("v,vtf->tf", counts, (tables[i] - acts.mean(0)) ** 2) / N
        eta[i] = torch.where((var > 1e-12) & (rate >= min_rate), between / var.clamp_min(1e-12), torch.zeros_like(var))
    return eta, rate, tables


@torch.no_grad()
def edited_run(rnn, x, deltas):
    """Real GRU with candidate n += deltas[t] (N, H) at the given steps. Output-step logits."""
    gru, readout = rnn.layers[0], rnn.layers[-1]
    N, T, _ = x.shape
    h = torch.zeros(N, gru.hidden_size)
    out = []
    for t in range(T):
        xt = x[:, t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        if t in deltas:
            n = n + deltas[t]
        h = (1 - z) * h + z * n
        if t >= T - T // 2:
            out.append(readout(h))
    return torch.stack(out, 1)


@torch.no_grad()
def preact_system(tracer, acts, nodes, k, tau):
    """Candidate preactivations of `nodes` [(t, f)] as an exact linear system, gates frozen:

      p = M a + u_in + u_write,   target = m_T . a + u_T

    M[g, f]: effect of feature f's value (step s) on g's preactivation (step t > s),
    through the frozen carry, update gate and g's reset gate; u_in: input and
    encoder bias; u_write: decoder-bias and error writes (z_s * (b_dec + e_s))
    read the same way. Target: mean-centred logit of token k at step tau.
    """
    T, H = acts["h_ts"].shape
    z, r, x = acts["z_ts"], acts["r_ts"], acts["inputs"]
    tc = tracer.hidden_transcoder
    dec, b_dec, b_enc = tc.features_to_outputs.weight, tc.features_to_outputs.bias, tc.input_to_features.bias
    steps = torch.tensor([t for t, _ in nodes])
    fidx = torch.tensor([f for _, f in nodes])
    carry = torch.ones(T, T, H)                                          # carry[s, t] = prod_{u=s+1}^{t} (1 - z_u)
    for s in range(T):
        for t in range(s + 1, T):
            carry[s, t] = carry[s, t - 1] * (1 - z[t])
    read = tracer.W_n_h[fidx] * r[steps]                                 # (n, H): g reads r_t * h_{t-1}
    write = z[steps] * dec[:, fidx].T                                    # (n, H): f writes z_s * d_f per unit value
    bias_write = z * (b_dec + acts["e_n"])                               # (T, H)
    n = len(nodes)
    M, u_write = torch.zeros(n, n), torch.zeros(n)
    by_step = {t: torch.nonzero(steps == t).flatten() for t in steps.unique().tolist()}
    for t, gi in by_step.items():
        for s in range(t):
            C = carry[s, t - 1]
            u_write[gi] += read[gi] @ (C * bias_write[s])
            if s in by_step:
                fi = by_step[s]
                M[gi[:, None], fi[None, :]] = read[gi] @ (C[:, None] * write[fi].T)
    u_in = (tracer.W_n_x[fidx] * x[steps]).sum(-1) + b_enc[fidx]
    w_c = tracer.W_o[k] - tracer.W_o.mean(0)
    m_T = torch.stack([w_c @ (carry[s, tau] * write[g]) if s <= tau else torch.tensor(0.0)
                       for g, s in enumerate(steps.tolist())])
    u_T = sum(w_c @ (carry[s, tau] * bias_write[s]) for s in range(tau + 1))
    a = acts["f_n"][steps, fidx]
    p = acts["pf_n"][steps, fidx]
    return {"M": M, "m_T": m_T, "u_in": u_in, "u_write": u_write, "u_T": float(u_T), "a": a, "p": p}


@torch.no_grad()
def change_flow(tracer, run_o, run_s, L, orig, i):
    """Exact contrastive flow of the change in the target (logit of orig[i] at output i)
    from the swapped run to the original, through candidate features.

    Preactivation changes split exactly (bars = mean of the two runs):
      dp = Mbar da + dM abar + du_in + du_write
    Each feature's JumpReLU is replaced by its secant slope kappa = da / dp
    (DeepLIFT's rescale rule), which makes the change system linear and exact:
    with Dt = sensitivity of the target change to a node's preactivation change,
      d target = sum_g Dt_g (dM abar + du_in + du_write)_g + dm_T . abar + du_T.
    Unchanged features and inputs carry no flow (no partner credit).
    Returns node flows Dt_g dp_g, edge flows Dt_g Mbar_gf da_f, flows by source, and checks.
    """
    tau = L + i
    nodes = sorted({(t, f) for run in (run_o, run_s) for t, f in torch.nonzero(run["f_n"][:tau + 1] > 0).tolist()})
    so, ss = (preact_system(tracer, run, nodes, orig[i], tau) for run in (run_o, run_s))
    Mb, dM = (so["M"] + ss["M"]) / 2, so["M"] - ss["M"]
    ab, da, dp = (so["a"] + ss["a"]) / 2, so["a"] - ss["a"], so["p"] - ss["p"]
    kappa = torch.where(dp.abs() > 1e-6, da / torch.where(dp.abs() > 1e-6, dp, torch.ones_like(dp)), torch.zeros_like(dp))
    src = {"gates": dM @ ab, "input": so["u_in"] - ss["u_in"], "writes": so["u_write"] - ss["u_write"]}
    n = len(nodes)
    mTb, dmT = (so["m_T"] + ss["m_T"]) / 2, so["m_T"] - ss["m_T"]
    Dt = torch.linalg.solve((torch.eye(n) - Mb * kappa[None, :]).T, mTb * kappa)
    edges = Dt[:, None] * Mb * da[None, :]                               # (g, f): carried change f -> g
    flow = {
        "nodes": {nd: float(Dt[g] * dp[g]) for g, nd in enumerate(nodes)},
        "da": {nd: float(da[g]) for g, nd in enumerate(nodes)},          # activation change (orig - swapped)
        "edges": edges, "node_list": nodes,
        "into_nodes": {key: Dt * v for key, v in src.items()},           # per node, by source
        "target_direct": mTb * da,                                       # carried change written straight into the logit
        "target_gates": float(dmT @ ab), "target_writes": so["u_T"] - ss["u_T"],
    }
    total = float(sum((Dt * v).sum() for v in src.values()) + flow["target_gates"] + flow["target_writes"])
    flow["dtarget"] = float(so["m_T"] @ so["a"] + so["u_T"] - ss["m_T"] @ ss["a"] - ss["u_T"])
    flow["check"] = total - flow["dtarget"]
    flow["preact_check"] = float(max((so["M"] @ so["a"] + so["u_in"] + so["u_write"] - so["p"]).abs().max(),
                                     (dp - Mb @ da - sum(src.values())).abs().max()))
    return flow


@torch.no_grad()
def ablated_candidate_run(rnn, tc, x, step, feats, means):
    """Real GRU; at `step`, candidate n minus sum_f (a_f - mean_f) d_f for feats. Output-step logits.

    feats: feature ids (same for every sequence) or an (N, F) bool mask (per sequence).
    """
    gru, readout = rnn.layers[0], rnn.layers[-1]
    N, T, _ = x.shape
    h = torch.zeros(N, gru.hidden_size)
    out = []
    for t in range(T):
        xt = x[:, t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        if t == step and torch.is_tensor(feats):
            a = tc.act(tc.input_to_features(torch.cat([r * h, xt], -1)))
            n = n - ((a - means) * feats) @ tc.features_to_outputs.weight.T
        elif t == step and len(feats):
            a = tc.act(tc.input_to_features(torch.cat([r * h, xt], -1)))[:, feats]
            n = n - (a - means[feats]) @ tc.features_to_outputs.weight[:, feats].T
        h = (1 - z) * h + z * n
        if t >= T - T // 2:
            out.append(readout(h))
    return torch.stack(out, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rnn", required=True)
    parser.add_argument("--traces", required=True, help="trace file of one length")
    parser.add_argument("--transcoders", nargs=3, required=True, metavar="GATE=PATH")
    parser.add_argument("--feature_results", default=None)
    parser.add_argument("--eta", type=float, default=0.2, help="eta^2 threshold for slot carriers")
    parser.add_argument("--n_flow", type=int, default=300, help="sequences for the flow graphs")
    parser.add_argument("--n_ablate", type=int, default=3000, help="sequences for the ablations")
    parser.add_argument("--top", type=int, default=8, help="members listed per supernode / step")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5], help="carriers ablated per slot and step")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = dict(item.split("=", 1) for item in args.transcoders)
    rnn = build_task_rnn("copy", 31, 128)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    tc = load_transcoder(paths["candidate"])
    tracer = CircuitTracer(rnn, load_transcoder(paths["update"]), tc, device="cpu",
                           reset_transcoder=load_transcoder(paths["reset"]))
    labels = load_labels(args.feature_results)
    data = torch.load(args.traces, map_location="cpu", mmap=True).datasets
    cols = {k: data[k] for k in ("inputs", "h_prevs", "r_ts")}
    L = data["outputs"].shape[1]
    T = 2 * L
    tokens = cols["inputs"][:, :L, :-1].argmax(-1)
    rng = np.random.default_rng(args.seed)

    # 1. selectivity and supernodes
    acts = candidate_acts(tc, cols)
    eta, rate, tables = selectivity(acts, tokens)
    members = {}                                                         # (t, f) -> slot it carries most
    best_eta, best_slot = eta.max(0)
    for t, f in torch.nonzero(best_eta >= args.eta).tolist():
        members[(t, f)] = int(best_slot[t, f])
    supernodes = {}
    for t in range(T):
        for i in range(L):
            ids = sorted([f for (tt, f), s in members.items() if tt == t and s == i], key=lambda f: -float(eta[i, t, f]))
            if ids:
                supernodes[f"S{i}@{step_label(t, L)}"] = {
                    "slot": i, "step": step_label(t, L), "n_features": len(ids), "features": ids,
                    "top": [{"feature": f, "eta2": float(eta[i, t, f]), "fire_rate": float(rate[t, f]),
                             **({"label": labels[("hidden", f)][0]} if ("hidden", f) in labels else {})}
                            for f in ids[:args.top]]}
    print("supernodes (slot carriers, eta^2 >= %.2f):" % args.eta)
    for t in range(T):
        print(f"  {step_label(t, L):>4}: " + "  ".join(f"S{i}:{supernodes[f'S{i}@{step_label(t, L)}']['n_features']}"
                                                   for i in range(L) if f"S{i}@{step_label(t, L)}" in supernodes)
              + f"  | active {int((rate[t] > 0.01).sum())}")

    # 2. contrastive change flow
    orig_all = tokens.numpy()
    distinct = np.flatnonzero([len(set(r)) == L for r in orig_all])
    rows = np.sort(rng.choice(distinct, size=min(args.n_flow, len(distinct)), replace=False))
    n = len(rows)
    agg = [{"nodes": defaultdict(float), "present": defaultdict(int), "edges": defaultdict(float),
            "step_edges": defaultdict(float), "sources": defaultdict(float), "dtarget": 0.0,
            "check": 0.0, "preact_check": 0.0, "k80": defaultdict(list)} for _ in range(L)]
    own = defaultdict(dict)                                              # (slot, t) -> {row: its top carriers}
    start = time.time()
    for row in rows:
        x = cols["inputs"][row].float()
        orig = [int(v) for v in orig_all[row]]
        run_o = tracer.run_forward_pass(gru_trace(rnn, x))
        for i in range(L):
            new = int(rng.choice([v for v in range(30) if v not in orig]))
            run_s = tracer.run_forward_pass(gru_trace(rnn, swap_token(x, L, i, new)))
            fl, a = change_flow(tracer, run_o, run_s, L, orig, i), agg[i]
            nodes = fl["node_list"]
            by_t = defaultdict(list)
            for nd, v in fl["nodes"].items():
                a["nodes"][nd] += v / n
                a["present"][nd] += 1
                by_t[nd[0]].append((v, nd[1]))
            for t, items in by_t.items():                                # this sequence's own carriers
                items.sort(key=lambda vf: -vf[0])
                pos_flow = np.array([v for v, _ in items if v > 0])
                a["k80"][t].append(int(np.searchsorted(np.cumsum(pos_flow), 0.8 * pos_flow.sum()) + 1) if len(pos_flow) else 0)
                own[(i, t)][row] = [f for v, f in items if v > 0][:max(args.ks)]
            for gi, fi in torch.nonzero(fl["edges"].abs() > 1e-4).tolist():
                (s, f), (t, gg) = nodes[fi], nodes[gi]
                v = float(fl["edges"][gi, fi])
                a["edges"][((s, f), (t, gg))] += v / n
                a["step_edges"][(step_label(s, L), step_label(t, L))] += v / n
            for key, per_node in fl["into_nodes"].items():
                for g_, (t, _) in enumerate(nodes):
                    a["sources"][(key, step_label(t, L))] += float(per_node[g_]) / n
            for g_, (t, _) in enumerate(nodes):
                a["step_edges"][(step_label(t, L), "target")] += float(fl["target_direct"][g_]) / n
            a["sources"][("gates", "target")] += fl["target_gates"] / n
            a["sources"][("writes", "target")] += fl["target_writes"] / n
            a["dtarget"] += fl["dtarget"] / n
            a["check"] = max(a["check"], abs(fl["check"]))
            a["preact_check"] = max(a["preact_check"], fl["preact_check"])
    print(f"change flow: {n} sequences x {L} slots, {time.time() - start:.0f}s")

    def describe(t, f, value, present, i=None):
        """Feature summary; with slot i, its magnitude code: tokens in slot i that raise /
        lower its mean activation at step t (slot-token table minus its token average)."""
        lab = labels.get(("hidden", f))
        d = {"feature": f, "step": step_label(t, L), "flow": value, "present_frac": present / n,
             "fire_rate": float(rate[t, f]), **({"label": lab[0]} if lab else {})}
        if i is not None:
            tab = tables[i, :, t, f] - tables[i, :, t, f].mean()
            order = torch.argsort(tab, descending=True).tolist()
            d["eta2"] = float(eta[i, t, f])
            d["raised_by"] = [(TOKENS[k], round(float(tab[k]), 2)) for k in order[:4]]
            d["lowered_by"] = [(TOKENS[k], round(float(tab[k]), 2)) for k in order[::-1][:3]]
        return d

    flow_out, carriers = [], {}
    for i, a in enumerate(agg):
        by_step = defaultdict(list)
        for (t, f), v in a["nodes"].items():
            by_step[t].append((v, f))
        step_nodes = {}
        for t, items in sorted(by_step.items()):
            items.sort(key=lambda vf: -vf[0])                            # most supportive first
            carriers[(i, t)] = [f for v, f in items if v > 0]
            step_nodes[step_label(t, L)] = {"total": sum(v for v, _ in items),
                                            "k80_per_sequence": float(np.median(a["k80"][t])) if a["k80"][t] else 0,
                                            "top": [describe(t, f, v, a["present"][(t, f)], i) for v, f in items[:args.top]]}
        top_edges = sorted(a["edges"].items(), key=lambda kv: -abs(kv[1]))[:40]
        flow_out.append({
            "slot": i, "dtarget": a["dtarget"], "max_abs_check": a["check"], "max_preact_check": a["preact_check"],
            "nodes_by_step": step_nodes,
            "step_edges": [{"src": s, "dst": d, "flow": v} for (s, d), v in sorted(a["step_edges"].items(), key=lambda kv: -abs(kv[1]))],
            "sources": [{"source": k_, "step": st, "flow": v} for (k_, st), v in sorted(a["sources"].items(), key=lambda kv: -abs(kv[1]))],
            "top_edges": [{"src": describe(s, f, 0, 0)["feature"], "src_step": step_label(s, L),
                           "dst": gg, "dst_step": step_label(t, L), "flow": v} for ((s, f), (t, gg)), v in top_edges]})
        print(f"slot {i}: target change {a['dtarget']:.1f} (check {a['check']:.1e}, preact check {a['preact_check']:.1e})")
        print("   flow through each step's features (features carrying 80%, median per sequence): " + "  ".join(
            f"{st} {d['total']:+.1f} ({d['k80_per_sequence']:.0f})" for st, d in step_nodes.items()))
        print("   strongest step-to-step flows: " + "  ".join(
            f"{e['src']}->{e['dst']} {e['flow']:+.1f}" for e in flow_out[-1]["step_edges"][:8]))
        print("   sources: " + "  ".join(f"{e['source']}@{e['step']} {e['flow']:+.1f}" for e in flow_out[-1]["sources"][:6]))

    # 3. selectivity ablations: top-k carriers of slot i at step t, vs random active features
    x = cols["inputs"][:args.n_ablate].float()
    orig_t = tokens[:args.n_ablate]
    means = acts.mean(0)                                                 # (T, F) per-step means
    g = torch.Generator().manual_seed(args.seed)
    ablations = []
    for i in range(L):
        for t in range(i, L + i + 1):
            active = torch.nonzero(rate[t] > 0.01).flatten()
            for k in args.ks:
                feats = carriers.get((i, t), [])[:k]
                if not feats:
                    continue
                acc = (ablated_candidate_run(rnn, tc, x, t, feats, means[t]).argmax(-1) == orig_t).float().mean(0)
                rand = torch.stack([(ablated_candidate_run(rnn, tc, x, t, active[torch.randperm(len(active), generator=g)[:len(feats)]].tolist(),
                                                           means[t]).argmax(-1) == orig_t).float().mean(0) for _ in range(3)]).mean(0)
                others = [j for j in range(L) if j != i]
                ablations.append({"slot": i, "step": step_label(t, L), "k": len(feats), "features": feats,
                                  "accuracy": acc.tolist(), "random_accuracy": rand.tolist(),
                                  "drop_own": float(1 - acc[i]), "drop_others": float((1 - acc[others]).mean()),
                                  "random_drop_own": float(1 - rand[i]), "random_drop_others": float((1 - rand[others]).mean())})
                print(f"ablate slot {i} carriers at {step_label(t, L):>4} k={len(feats)} {feats}: acc " + " ".join(f"{v:.2f}" for v in acc)
                      + " | random " + " ".join(f"{v:.2f}" for v in rand))

    # 3b. per-sequence ablations: each flow sequence's own top-k carriers of slot i at step t
    x_rows = cols["inputs"][torch.from_numpy(rows)].float()
    tok_rows = tokens[torch.from_numpy(rows)]
    act_rows = acts[torch.from_numpy(rows)]                              # (n, T, F)
    own_ablations = []
    for i in range(L):
        for t in range(i, L + i + 1):
            for k in args.ks:
                mask = torch.zeros(n, acts.shape[-1], dtype=torch.bool)
                rand = torch.zeros_like(mask)
                for r_, row in enumerate(rows):
                    feats = own[(i, t)].get(row, [])[:k]
                    mask[r_, feats] = True
                    on = torch.nonzero(act_rows[r_, t] > 0).flatten()
                    rand[r_, on[torch.randperm(len(on), generator=g)[:len(feats)]]] = True
                acc = (ablated_candidate_run(rnn, tc, x_rows, t, mask.float(), means[t]).argmax(-1) == tok_rows).float().mean(0)
                racc = (ablated_candidate_run(rnn, tc, x_rows, t, rand.float(), means[t]).argmax(-1) == tok_rows).float().mean(0)
                others = [j for j in range(L) if j != i]
                own_ablations.append({"slot": i, "step": step_label(t, L), "k": k, "accuracy": acc.tolist(),
                                      "random_accuracy": racc.tolist(), "drop_own": float(1 - acc[i]),
                                      "drop_others": float((1 - acc[others]).mean()),
                                      "random_drop_own": float(1 - racc[i]), "random_drop_others": float((1 - racc[others]).mean())})
                print(f"own carriers: slot {i} at {step_label(t, L):>4} k={k}: acc " + " ".join(f"{v:.2f}" for v in acc)
                      + " | random " + " ".join(f"{v:.2f}" for v in racc))

    # 4. steering by token tables: move carriers' magnitudes as if slot i held another token
    rows_s = distinct[:args.n_ablate]
    x_s = cols["inputs"][torch.from_numpy(rows_s)].float()
    orig_s = tokens[torch.from_numpy(rows_s)]
    present = torch.zeros(len(rows_s), 30, dtype=torch.bool).scatter_(1, orig_s, True)
    new_s = torch.multinomial((~present).float(), 1, generator=g)[:, 0]
    dec = tc.features_to_outputs.weight
    perm = torch.stack([torch.randperm(30, generator=g) for _ in range(acts.shape[-1])], 1)   # (V, F) shuffled rows
    base_logits = edited_run(rnn, x_s, {})
    steer = []
    for i in range(L):
        def feature_sets(t):
            sel = torch.nonzero(eta[i, t] >= 0.05).flatten().tolist()
            return [("top 5 carriers", carriers.get((i, t), [])[:5]), ("top 20 carriers", carriers.get((i, t), [])[:20]),
                    (f"all slot-{i} selective", sel)]
        step_sets = [(step_label(t, L), [t]) for t in range(i, L + i + 1)]
        step_sets += [("relay (all steps between)", list(range(i + 1, L + i))), ("write to read-out (all)", list(range(i, L + i + 1)))]
        for where, steps_ in step_sets:
            for fs_name in ("top 5 carriers", "top 20 carriers", f"all slot-{i} selective"):
                for control in (False, True):
                    deltas = {}
                    for t in steps_:
                        feats = dict(feature_sets(t))[fs_name]
                        if not feats:
                            continue
                        tab = tables[i, :, t][:, feats]                      # (V, k)
                        if control:
                            tab = tab[perm[:, feats], torch.arange(len(feats))]
                        deltas[t] = (tab[new_s] - tab[orig_s[:, i]]) @ dec[:, feats].T
                    if not deltas:
                        continue
                    logits = edited_run(rnn, x_s, deltas)
                    pred = logits.argmax(-1)
                    gap = lambda lg: (lg[:, i].gather(1, new_s[:, None]) - lg[:, i].gather(1, orig_s[:, i:i + 1]))[:, 0]
                    others = [j for j in range(L) if j != i]
                    steer.append({"slot": i, "where": where, "features": fs_name, "control": "shuffled tables" if control else None,
                                  "n_features_per_step": {step_label(t, L): len(dict(feature_sets(t))[fs_name]) for t in steps_},
                                  "flip_to_new": float((pred[:, i] == new_s).float().mean()),
                                  "acc_own": float((pred[:, i] == orig_s[:, i]).float().mean()),
                                  "acc_others": float((pred[:, others] == orig_s[:, others]).float().mean()),
                                  "dgap_new_minus_orig": float((gap(logits) - gap(base_logits)).mean())})
                    s = steer[-1]
                    print(f"steer slot {i} at {where:>26} [{fs_name:>18}{' SHUFFLED' if control else ''}]: "
                          f"flip->new {s['flip_to_new']:.2f}  own acc {s['acc_own']:.2f}  others {s['acc_others']:.2f}  "
                          f"d(new-orig) {s['dgap_new_minus_orig']:+.1f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "L": L, "steps": [step_label(t, L) for t in range(T)],
               "supernodes": supernodes, "flows": flow_out, "ablations": ablations,
               "own_ablations": own_ablations, "steer": steer}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
