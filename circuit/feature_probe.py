"""Probe dense transcoder features that matter only through the recurrence.

For a few features of one gate's transcoder (e.g. the dense candidate features
that circuit/explained_fidelity.py found essential but unexplained):

  profile     firing rate and mean activation by phase, and the features'
              pairwise correlation
  regression  how well one-hot cue groups predict the activation *magnitude*
              (ridge, R^2 on held-out sequences): timing (L, t), the current
              input, recent inputs, the token being output, upcoming outputs,
              the set of tokens seen so far, all original tokens. Each group is
              added to timing alone and dropped from the full model.
  conditional which information matters for behaviour: replace the feature by
  ablation   its mean conditional on a cue (global, (L, t), (L, t, input),
              (L, t, output target), ...) and run the GRU. If conditioning on a
              cue restores top-1 agreement, that cue is what the feature
              carries that the model needs.
  timing      top-1 agreement by output index under global mean ablation, and
              where the hidden state starts to diverge
  decoder     each feature's direct effect on the output logits (readout of its
              decoder vector) and its alignment with the other features

    python -m circuit.feature_probe --rnn RNN.ckpt --transcoder C.ckpt --gate candidate \\
        --features 162 214 284 388 487 --traces T3.pt ... T9.pt --out probe.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from circuit.autointerp import TOKENS
from circuit.explained_fidelity import load_eval, load_transcoder, teacher_forced
from training.splice_eval import build_task_rnn

NONE = len(TOKENS)            # "no token" category (31 one-hot slots per token cue)


def cues(groups):
    """Per-step cue columns (flattened in teacher_forced order) and per-group step grids."""
    cols = {k: [] for k in ("L", "t", "phase", "in_tok", "prev_tok", "prev2_tok", "target", "next1", "next2", "seq")}
    seen, orig_cols, seq_base = [], [], 0
    for g in groups:
        tok = g["inputs"].argmax(-1)                                    # (n, T); <DEL> = 30
        n, T = tok.shape
        L = T // 2
        orig = tok[:, :L]
        t = torch.arange(T).expand(n, T)
        pad = lambda a, k: torch.cat([torch.full((n, k), NONE), a[:, :T - k]], 1)

        def out_at(offset):                                             # orig[t - L + offset], NONE outside
            idx = t - L + offset
            ok = (idx >= 0) & (idx < L)
            return torch.where(ok, orig.gather(1, idx.clamp(0, L - 1)), torch.full_like(idx, NONE))

        add = {"L": torch.full((n, T), L), "t": t, "phase": (t > L).long() + (t >= L).long(),
               "in_tok": tok, "prev_tok": pad(tok, 1), "prev2_tok": pad(tok, 2),
               "target": out_at(0), "next1": out_at(1), "next2": out_at(2),
               "seq": (torch.arange(n) + seq_base)[:, None].expand(n, T)}
        for k, v in add.items():
            cols[k].append(v.flatten())
        onehot = F.one_hot(tok.clamp(max=NONE - 1), NONE).cumsum(1).clamp(max=1)     # tokens seen up to t
        seen.append(onehot[..., :30].flatten(0, 1))
        o = torch.full((n, T, 9), NONE)
        o[:, :, :L] = orig[:, None, :]
        orig_cols.append(o.flatten(0, 1))
        seq_base += n
    out = {k: torch.cat(v).numpy() for k, v in cols.items()}
    out["seen"] = torch.cat(seen).numpy()
    out["orig"] = torch.cat(orig_cols).numpy()
    return out


def design(c, groups_used):
    """Sparse one-hot design matrix for the named cue groups."""
    from scipy import sparse
    blocks = []
    def onehot(codes, n_cat):
        rows = np.arange(len(codes))
        return sparse.csr_matrix((np.ones(len(codes)), (rows, codes)), shape=(len(codes), n_cat))
    for name in groups_used:
        if name == "timing":
            blocks.append(onehot((c["L"] - 3) * 18 + c["t"], 7 * 18))
        elif name == "recent":
            blocks += [onehot(c["prev_tok"], NONE + 1), onehot(c["prev2_tok"], NONE + 1)]
        elif name == "upcoming":
            blocks += [onehot(c["next1"], NONE + 1), onehot(c["next2"], NONE + 1)]
        elif name == "seen":
            blocks.append(sparse.csr_matrix(c["seen"].astype(np.float64)))
        elif name == "orig":
            blocks += [onehot(c["orig"][:, k], NONE + 1) for k in range(9)]
        else:                                                           # in_tok, target
            blocks.append(onehot(c[name], NONE + 1))
    return sparse.hstack(blocks).tocsr()


CUE_GROUPS = ["in_tok", "recent", "target", "upcoming", "seen", "orig"]


def regression(c, acts):
    from sklearn.linear_model import Ridge
    test = c["seq"] % 2 == 1
    out = {}
    def r2(groups_used, y):
        X = design(c, groups_used)
        m = Ridge(alpha=1.0).fit(X[~test], y[~test])
        resid = y[test] - m.predict(X[test])
        return float(1 - (resid ** 2).sum() / ((y[test] - y[test].mean()) ** 2).sum())
    for fid, y in acts.items():
        res = {"timing": r2(["timing"], y)}
        for g in CUE_GROUPS:
            res[f"timing+{g}"] = r2(["timing", g], y)
        res["all"] = r2(["timing"] + CUE_GROUPS, y)
        for g in CUE_GROUPS:
            res[f"all-{g}"] = r2(["timing"] + [x for x in CUE_GROUPS if x != g], y)
        out[fid] = res
        print(f"  feature {fid}: " + "  ".join(f"{k} {v:.2f}" for k, v in res.items()), flush=True)
    return out


@torch.no_grad()
def run_override(rnn, tc, x, gate, override):
    """GRU outputs and hidden states with chosen features of the gate's transcoder overridden.

    override: {feature: (n, T) replacement values} or {} for the plain transcoder;
    None runs the original GRU.
    """
    gru, readout = rnn.layers[0], rnn.layers[-1]
    n, T, _ = x.shape
    h = torch.zeros(n, gru.hidden_size)
    logits, hs = [], []
    ids = torch.tensor(list(override or {}), dtype=torch.long)
    for t in range(T):
        xt = x[:, t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        gate_in = torch.cat([r * h if gate == "candidate" else h, xt], -1)
        if override is None:
            val = None
        else:
            f = tc.act(tc.input_to_features(gate_in))
            if len(ids):
                f[:, ids] = torch.stack([override[i][:, t] for i in override.keys()], 1)
            val = tc.features_to_outputs(f)
        if gate == "reset":
            r = val if val is not None else r
        if gate == "update":
            z = val if val is not None else z
        n_t = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        if gate == "candidate" and val is not None:
            n_t = val
        h = (1 - z) * h + z * n_t
        hs.append(h)
        if t >= T - T // 2:
            logits.append(readout(h))
    return torch.stack(logits, 1), torch.stack(hs, 1)


def conditional_means(c, y, key):
    """Mean of y within each cell of the cue combination `key` (tuple of column names)."""
    codes = np.zeros(len(y), dtype=np.int64)
    for k in key:
        codes = codes * 64 + c[k].astype(np.int64)
    uniq, inv = np.unique(codes, return_inverse=True)
    means = np.bincount(inv, weights=y) / np.bincount(inv)
    return means[inv]


CONDITIONS = [("global", ()), ("(L, t)", ("L", "t")), ("(L, t, input)", ("L", "t", "in_tok")),
              ("(L, t, target)", ("L", "t", "target")), ("(L, t, next output)", ("L", "t", "next1")),
              ("(L, t, input, prev input)", ("L", "t", "in_tok", "prev_tok")),
              ("(L, t, target, next output)", ("L", "t", "target", "next1"))]


@torch.no_grad()
def behaviour(rnn, tc, groups, gate, c, acts, sets):
    """Top-1 agreement with the original GRU when feature sets are replaced by conditional means."""
    refs = [run_override(rnn, tc, g["inputs"], gate, None)[0] for g in groups]
    sizes = [g["inputs"].shape[0] * g["inputs"].shape[1] for g in groups]
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    def agreement(override_flat):
        agree = n = 0
        per_index = {}
        for gi, (g, ref) in enumerate(zip(groups, refs)):
            shape = g["inputs"].shape[:2]
            ov = {i: torch.from_numpy(v[bounds[gi]:bounds[gi + 1]]).float().reshape(shape)
                  for i, v in override_flat.items()}
            out, _ = run_override(rnn, tc, g["inputs"], gate, ov)
            same = (out.argmax(-1) == ref.argmax(-1))
            agree += same.sum().item(); n += same.numel()
            for j in range(same.shape[1]):
                a, b = per_index.get(j, (0, 0))
                per_index[j] = (a + same[:, j].sum().item(), b + same.shape[0])
        return agree / n, {j: a / b for j, (a, b) in sorted(per_index.items())}
    base, _ = agreement({})
    print(f"  plain transcoder: top1 {base:.3f}")
    results = {"plain_transcoder": base, "sets": []}
    for name, ids in sets:
        row = {"set": name, "features": ids, "conditions": {}}
        for cname, key in CONDITIONS:
            ov = {i: conditional_means(c, acts[i], key) for i in ids}
            top1, by_index = agreement(ov)
            row["conditions"][cname] = top1
            if cname == "global":
                row["global_by_output_index"] = by_index
        results["sets"].append(row)
        print(f"  {name:>10}: " + "  ".join(f"{k} {v:.3f}" for k, v in row["conditions"].items()), flush=True)
    return results


@torch.no_grad()
def divergence(rnn, tc, groups, gate, c, acts, ids):
    """Mean |h_ablated - h_plain| per step relative to |h_plain|, under global mean ablation."""
    sizes = [g["inputs"].shape[0] * g["inputs"].shape[1] for g in groups]
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    out = {}
    for gi, g in enumerate(groups):
        shape = g["inputs"].shape[:2]
        ov = {i: torch.full(shape, float(acts[i].mean())) for i in ids}
        _, h0 = run_override(rnn, tc, g["inputs"], gate, {})
        _, h1 = run_override(rnn, tc, g["inputs"], gate, ov)
        rel = ((h1 - h0).norm(dim=-1) / h0.norm(dim=-1).clamp_min(1e-6)).mean(0)
        out[int(shape[1] // 2)] = [round(float(v), 4) for v in rel]
    return out


def token_tables(c, acts, min_count=20):
    """Mean activation relative to what timing predicts, by token cue.

    Residual = activation - its mean at the same (L, t), so the tables show what
    tokens add on top of timing. Input-phase tables use steps before <DEL>;
    output tables use the steps that output a token (t >= L). Cells with fewer
    than min_count steps are None.
    """
    tokens = TOKENS[:30]
    inp = c["phase"] == 0
    out = c["t"] >= c["L"]
    def table(mask, a, b=None):
        idx = a[mask] if b is None else a[mask] * 30 + b[mask]
        size = 30 if b is None else 900
        counts = np.bincount(idx, minlength=size)
        sums = np.bincount(idx, weights=res[mask], minlength=size)
        m = np.where(counts >= min_count, sums / np.maximum(counts, 1), np.nan)
        m = m if b is None else m.reshape(30, 30)
        return [None if np.isnan(v) else round(float(v), 4) for v in np.ravel(m)] if b is None else \
            [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in m]
    report = {"tokens": tokens}
    for i, y in acts.items():
        res = y - conditional_means(c, y, ("L", "t"))
        has_prev = inp & (c["t"] >= 1)
        report[i] = {
            "mean": float(y.mean()), "residual_std": float(res.std()),
            "input": table(inp, c["in_tok"]),
            "prev_input": table(has_prev, c["prev_tok"]),
            "prev_x_input": table(has_prev, c["prev_tok"], c["in_tok"]),
            "target": table(out, c["target"]),
            "target_x_next": table(out & (c["next1"] < 30), c["target"], c["next1"]),
        }
        for key in ("input", "prev_input", "target"):
            vals = [(t, v) for t, v in zip(tokens, report[i][key]) if v is not None]
            vals.sort(key=lambda tv: tv[1])
            print(f"  feature {i} by {key:>10}: highest {' '.join(f'{t}{v:+.2f}' for t, v in vals[::-1][:6])}"
                  f" | lowest {' '.join(f'{t}{v:+.2f}' for t, v in vals[:6])}")
    return report


SECTIONS = ["profile", "regression", "decoder", "behaviour", "divergence", "tables"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--transcoder", required=True)
    ap.add_argument("--gate", default="candidate", choices=["reset", "update", "candidate"])
    ap.add_argument("--features", nargs="+", type=int, required=True)
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--max_sequences", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sections", nargs="*", default=SECTIONS, choices=SECTIONS)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    rnn = build_task_rnn("copy", input_size=31, hidden_size=128)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    rnn.eval()
    tc = load_transcoder(args.transcoder)
    groups = load_eval(args.traces, args.max_sequences)
    X, _ = teacher_forced(groups, args.gate)
    feats = torch.cat([tc.act(tc.input_to_features(X[i:i + 65536])) for i in range(0, len(X), 65536)])
    acts = {i: feats[:, i].numpy().astype(np.float64) for i in args.features}
    c = cues(groups)
    sections = set(args.sections)
    # sections not rerun keep their previous results
    report = json.load(open(args.out)) if os.path.exists(args.out) else {}
    report["args"] = vars(args)

    if "profile" in sections:
        print("profile")
        phases = ["input", "<DEL>", "copy"]
        report["profile"] = {}
        for i, y in acts.items():
            p = {ph: {"fire_rate": float((y[c["phase"] == k] > 0).mean()), "mean": float(y[c["phase"] == k].mean())}
                 for k, ph in enumerate(phases)}
            report["profile"][i] = p
            print(f"  feature {i}: " + "  ".join(f"{ph} rate {v['fire_rate']:.2f} mean {v['mean']:.3f}" for ph, v in p.items()))
        corr = np.corrcoef(np.stack([acts[i] for i in args.features]))
        report["correlation"] = corr.round(3).tolist()
        print("  correlation:\n" + "\n".join("   " + " ".join(f"{v:6.2f}" for v in row) for row in corr))

    if "regression" in sections:
        print("regression (held-out R^2 of activation magnitude)")
        report["regression"] = regression(c, acts)

    if "decoder" in sections:
        print("decoder")
        D = tc.features_to_outputs.weight                                     # (128, feats)
        W = rnn.layers[-1].weight                                             # (30, 128)
        report["decoder"] = {}
        for i in args.features:
            eff = (W @ D[:, i]).numpy()
            order = np.argsort(eff)
            report["decoder"][i] = {"norm": float(D[:, i].norm()),
                                    "top_up": [(TOKENS[k], float(eff[k])) for k in order[::-1][:5]],
                                    "top_down": [(TOKENS[k], float(eff[k])) for k in order[:5]]}
            print(f"  feature {i}: |d| {D[:, i].norm():.2f}  raises {[t for t, _ in report['decoder'][i]['top_up']]}"
                  f"  lowers {[t for t, _ in report['decoder'][i]['top_down']]}")
        Dn = F.normalize(D, dim=0)
        cos = (Dn.T @ Dn)
        sub = cos[args.features][:, args.features]
        report["decoder_cosine"] = sub.numpy().round(3).tolist()
        nearest = {}
        for i in args.features:
            others = cos[i].clone(); others[i] = -2
            top = torch.topk(others.abs(), 5)
            nearest[i] = [(int(j), float(cos[i, j])) for j in top.indices]
        report["decoder_nearest"] = nearest
        print("  decoder cosine between the features:\n" + "\n".join("   " + " ".join(f"{v:6.2f}" for v in row) for row in sub.numpy()))

    if "behaviour" in sections:
        print("conditional ablation (top-1 agreement with the original GRU)")
        sets = [(str(i), [i]) for i in args.features] + [("all", list(args.features))]
        report["behaviour"] = behaviour(rnn, tc, groups, args.gate, c, acts, sets)

    if "divergence" in sections:
        print("hidden-state divergence by step, all features globally mean-ablated")
        report["divergence"] = divergence(rnn, tc, groups, args.gate, c, acts, list(args.features))
        for L, v in report["divergence"].items():
            print(f"  L={L}: " + " ".join(f"{x:.3f}" for x in v))

    if "tables" in sections:
        print("token tables (activation minus its (L, t) mean)")
        report["tables"] = token_tables(c, acts)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
