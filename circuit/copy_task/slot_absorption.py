"""Is "slot filled" its own feature, or absorbed into token features?

Attribution graphs credit token-specific features (e.g. "first token is b")
with token-independent information ("slot 0 was filled"). Either the model
stores token x slot jointly and the features are faithful, or the candidate
transcoder folded a general slot feature into its token features (feature
absorption, Chanin et al. 2024). Per input step i of length-L sequences:

  model     what the GRU writes into h at step i, w = z_i * n_i, split into its
            mean over sequences (shared: the same whatever the token) and the
            rest; shared_frac = |E w|^2 / E |w|^2.
  carriers  the shared write E w as a sum over candidate features, decoder
            bias and error (E[z a_f] * d_f, ...), each projected on E w. Features
            are grouped by how they fire at step i: general (on for >= 50% of
            sequences), token-specific (rarer, and >= 80% of firings on <= 2
            input tokens), other. Shared write carried mainly by token-specific
            features = absorption.
  alignment for token-specific features, the fraction of their write that lies
            along the shared direction (1/H for a random direction).
  binding   h at the delimiter regressed on the tokens: slot-specific one-hots
            (conjunctive, L x V) vs a bag of tokens (factored, V). A factored
            code cannot store order.

    python -m circuit.copy_task.slot_absorption --rnn RNN.ckpt --traces T4.pt \\
        --candidate C.ckpt --feature_results features/<run>_autointerp --out absorption.json
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import StackDataset

from circuit.copy_task.autointerp import TOKENS
from circuit.copy_task.copy_circuits import load_labels, step_label
from circuit.copy_task.explained_fidelity import load_transcoder

torch.serialization.add_safe_globals([StackDataset])


def ridge_r2(X, Y, lam=1e-3):
    """Held-out R^2 (odd rows) of a ridge fit on even rows, pooled over output dims."""
    X = torch.cat([X, torch.ones(len(X), 1)], 1)
    tr, te = slice(0, None, 2), slice(1, None, 2)
    A = X[tr].T @ X[tr] + lam * torch.eye(X.shape[1])
    W = torch.linalg.solve(A, X[tr].T @ Y[tr])
    resid = Y[te] - X[te] @ W
    return float(1 - (resid ** 2).sum() / ((Y[te] - Y[te].mean(0)) ** 2).sum())


@torch.no_grad()
def step_report(cols, tc, i, labels, top):
    x, h, r, z, n = (cols[k][:, i].float() for k in ("inputs", "h_prevs", "r_ts", "z_ts", "h_new_ts"))
    f = tc.act(tc.input_to_features(torch.cat([r * h, x], -1)))       # (N, F)
    D, b = tc.features_to_outputs.weight, tc.features_to_outputs.bias  # (H, F), (H,)
    N = len(x)

    w = z * n                                                          # actual write into h
    shared = w.mean(0)
    shared_frac = float(shared.norm() ** 2 / (w.norm(dim=1) ** 2).mean())

    # Shared write as a sum of carriers, each projected on the shared direction.
    per_feat = (f.T @ z / N) * D.T                                     # (F, H): E[a_f z] * d_f
    bias = z.mean(0) * b
    error = (z * (n - tc.features_to_outputs(f))).mean(0)
    unit = shared / shared.norm() ** 2
    share = per_feat @ unit                                            # sums with bias/error to 1

    fires = (f > 0).float()
    rate = fires.mean(0)
    tok = x[:, :-1].argmax(-1) if x[:, :-1].sum(-1).all() else None    # None at the delimiter
    groups = np.full(f.shape[1], "other", dtype=object)
    groups[(rate >= 0.5).numpy()] = "general"
    top2 = torch.zeros(f.shape[1])
    if tok is not None:
        counts = torch.zeros(f.shape[1], 30).index_add_(1, tok, fires.T)   # firings per token
        top2 = counts.topk(2, dim=1).values.sum(1) / counts.sum(1).clamp_min(1)
        groups[((rate < 0.5) & (rate > 0) & (top2 >= 0.8)).numpy()] = "token_specific"
    groups[(rate == 0).numpy()] = "silent"

    carriers = {g: float(share[torch.from_numpy(groups == g)].sum()) for g in ("general", "token_specific", "other")}
    carriers["decoder_bias"] = float(bias @ unit)
    carriers["error"] = float(error @ unit)

    # Alignment of token-specific features' writes with the shared direction.
    spec = torch.from_numpy(groups == "token_specific")
    direction = shared / shared.norm()
    align = []
    for j in torch.nonzero(spec).flatten().tolist():
        on = f[:, j] > 0
        wj = z[on] * f[on, j:j + 1] * D[:, j]                          # its write when firing
        cos = (wj @ direction) / wj.norm(dim=1).clamp_min(1e-8)
        align.append((float((cos ** 2).mean()), float(cos.mean())))

    order = torch.argsort(share.abs(), descending=True)[:top].tolist()
    def describe(j):
        lab = labels.get(("hidden", j))
        d = {"feature": j, "share_of_shared_write": float(share[j]), "fire_rate": float(rate[j]),
             "group": groups[j], "top2_token_mass": float(top2[j])}
        if tok is not None and groups[j] == "token_specific":
            d["tokens"] = [TOKENS[k] for k in counts[j].topk(2).indices.tolist() if counts[j, k] > 0]
        return {**d, **({"label": lab[0]} if lab else {})}

    return {"shared_frac_of_write": shared_frac,
            "shared_carried_by": carriers,
            "n_features": {g: int((groups == g).sum()) for g in ("general", "token_specific", "other", "silent")},
            "token_specific_alignment": {
                "mean_cos2": float(np.mean([a[0] for a in align])) if align else None,
                "mean_cos": float(np.mean([a[1] for a in align])) if align else None,
                "random_cos2": 1.0 / D.shape[0]},
            "top_carriers": [describe(j) for j in order]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", required=True, help="trace file of one length")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--feature_results", default=None)
    parser.add_argument("--max_sequences", type=int, default=20000)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tc = load_transcoder(args.candidate)
    labels = load_labels(args.feature_results)
    data = torch.load(args.traces, map_location="cpu", mmap=True).datasets
    cols = {k: data[k][:args.max_sequences] for k in ("inputs", "h_prevs", "r_ts", "z_ts", "h_new_ts")}
    L = data["outputs"].shape[1]

    steps = {}
    for i in range(L + 1):
        steps[step_label(i, L)] = rep = step_report(cols, tc, i, labels, args.top)
        c = rep["shared_carried_by"]
        print(f"{step_label(i, L):>4}: shared {rep['shared_frac_of_write']:.2f} of the write | carried by "
              + ", ".join(f"{k} {v:+.2f}" for k, v in c.items())
              + f" | token-specific cos^2 {rep['token_specific_alignment']['mean_cos2'] or 0:.2f}")

    # Binding: h after the last input (entering the delimiter) from the tokens.
    h = cols["h_prevs"][:, L].float()
    tok = cols["inputs"][:, :L, :-1].float()                           # (N, L, V)
    binding = {"conjunctive_r2": ridge_r2(tok.flatten(1), h), "factored_r2": ridge_r2(tok.sum(1), h)}
    print(f"binding: h before <DEL> from slot-specific tokens R^2 {binding['conjunctive_r2']:.3f}, "
          f"from a bag of tokens R^2 {binding['factored_r2']:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "L": L, "steps": steps, "binding": binding}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
