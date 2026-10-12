"""Smaller copy-circuit experiments (real GRU), written to one JSON.

  trace_cosine    the token's trace in h (original minus token-swapped run, feedback
                  matched) at each step, cosine with the trace at its write step
  pop_split       at each pop step: the stored head token's logit, split into the part
                  kept from h_{t-1} and the part the candidate writes, with the matched
                  token fed back, a mismatched token, and no input term (memory only)
  early_feedback  at copy step L+k, feed back the right token, a token not in the
                  sequence, the token due now, the token due next, or an already popped
                  one; what is output, and what the candidate writes for each queued token
  dense_drop      candidate features mean-ablated in a transcoder splice (lengths 3-9):
                  the 5 dense features, the 15 other dense circuit carriers, random dense /
                  sparse controls, and each of the 15 alone
  token_cond      for six carriers, accuracy when removed, split by whether the output
                  token is in the feature's raising set (S+), lowering set (S-) or neither

    python -m circuit.copy_task.circuit_extras --rnn RNN.ckpt --trace_dir features/copy_transcoder_v2_1of5 \\
        --candidate C.ckpt --magnitude_scores features/copy_circuits/magnitude_scores_hidden.json --out extras.json
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import StackDataset

from circuit.copy_task.autointerp import TOKEN_IDX
from circuit.copy_task.circuit_tests import run
from circuit.copy_task.copy_circuits import step_label
from circuit.copy_task.explained_fidelity import Ablated, components, load_eval, load_transcoder, teacher_forced
from circuit.copy_task.explained_fidelity import run as splice_run
from training.splice_eval import build_task_rnn

torch.serialization.add_safe_globals([StackDataset])
FIVE = [162, 214, 284, 388, 487]
CIRCUIT_DENSE = [8, 32, 54, 77, 79, 91, 212, 231, 266, 274, 349, 368, 387, 440, 500]
SIX = [387, 77, 212, 231, 349, 368]


def load_length(trace_dir, L, n):
    d = torch.load(os.path.join(trace_dir, f"1M_128_high_seq{L}.pt"), map_location="cpu", mmap=True).datasets
    tok = d["inputs"][:, :L, :-1].argmax(-1)
    rows = torch.tensor([r for r in range(len(tok)) if len(set(tok[r].tolist())) == L][:n])
    return d["inputs"][rows].float(), tok[rows]


def hidden_entering(rnn, x, L):
    hs = []

    def keep(t, h, xt, r, z, n):
        hs.append(h.clone())
        return r, z, n
    run(rnn, x, L, keep)
    return torch.stack(hs, 1)                                            # (N, T, H): h_{t-1} at each step


def absent_tokens(orig, seed):
    present = torch.zeros(len(orig), 30, dtype=torch.bool).scatter_(1, orig, True)
    return torch.multinomial((~present).float(), 1, generator=torch.Generator().manual_seed(seed))[:, 0]


@torch.no_grad()
def trace_cosine(rnn, x, orig, L, seed):
    N, ar = len(x), torch.arange(len(x))
    hs = hidden_entering(rnn, x, L)
    out = {}
    for i in range(L):
        new = absent_tokens(orig, seed + i)
        x2 = x.clone()
        for t in [i] + ([L + 1 + i] if i + 1 < L else []):
            x2[:, t] = 0
            x2[ar, t, new] = 1
        hs2 = hidden_entering(rnn, x2, L)
        # trace after step t = h entering step t+1; the last step's h is not needed (no step after it)
        d = hs[:, 1:] - hs2[:, 1:]                                       # d[:, t] = trace after step t
        d0 = d[:, i]
        cos = lambda a, b: float(((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-8)).mean())
        out[i] = {step_label(t, L): cos(d[:, t], d0) for t in range(i, 2 * L - 1)}
    return out


@torch.no_grad()
def pop_split(rnn, x, orig, L, seed):
    gru, ro = rnn.layers[0], rnn.layers[-1]
    W_c = ro.weight - ro.weight.mean(0, keepdim=True)
    H = hidden_entering(rnn, x, L)
    u = absent_tokens(orig, seed)
    out = {}
    for k in range(1, L):
        t = L + k
        h, v = H[:, t], orig[:, k - 1]
        w = W_c[v]
        rec = {}
        for name, tok in (("matched", v), ("mismatched", u)):
            xt = torch.nn.functional.one_hot(tok, 31).float()
            r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
            z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
            mem = gru.hidden_to_new(r * h)
            n, n0 = torch.tanh(gru.input_to_new(xt) + mem), torch.tanh(mem)
            rec[name] = {"kept": float((w * (1 - z) * h).sum(-1).mean()), "written": float((w * z * n).sum(-1).mean()),
                         "written_memory_only": float((w * z * n0).sum(-1).mean()), "z_mean": float(z.mean())}
        out[step_label(t, L)] = rec
    return out


@torch.no_grad()
def early_feedback(rnn, x, orig, L, seed):
    gru, ro = rnn.layers[0], rnn.layers[-1]
    W_c = ro.weight - ro.weight.mean(0, keepdim=True)
    N, ar = len(x), torch.arange(len(x))
    H = hidden_entering(rnn, x, L)
    u = absent_tokens(orig, seed)
    out = {}
    for k in range(2, L - 1):
        t = L + k
        conds = {"right token orig[k-1]": orig[:, k - 1], "token not in sequence": u, "token due now orig[k]": orig[:, k],
                 "token due next orig[k+1]": orig[:, k + 1], "already popped orig[k-2]": orig[:, k - 2]}
        res = {}
        for name, fb in conds.items():
            x2 = x.clone()
            x2[:, t] = 0
            x2[ar, t, fb] = 1
            pred = run(rnn, x2, L).argmax(-1)
            h, xt = H[:, t], x2[:, t]
            r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
            z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
            mem = gru.hidden_to_new(r * h)
            n, n0 = torch.tanh(gru.input_to_new(xt) + mem), torch.tanh(mem)
            wl = lambda vv, nn: float((W_c[vv] * z * nn).sum(-1).mean())
            res[name] = {f"output k = orig[{lab}]": float((pred[:, k] == orig[:, j]).float().mean())
                         for lab, j in (("k-1", k - 1), ("k", k), ("k+1", k + 1))}
            res[name]["output k+1 correct"] = float((pred[:, k + 1] == orig[:, k + 1]).float().mean())
            res[name]["write"] = {lab: {"with input": wl(orig[:, j], n), "memory only": wl(orig[:, j], n0)}
                                  for lab, j in (("orig[k-1]", k - 1), ("orig[k]", k), ("orig[k+1]", k + 1))}
        out[f"copy step L+{k}"] = res
    return out


@torch.no_grad()
def splice_setup(trace_dir, candidate, n, seed):
    """Held-out sequences of lengths 3-9, the candidate transcoder, its feature means and step firing rates."""
    paths = [os.path.join(trace_dir, f"1M_128_high_seq{L}.pt") for L in range(3, 10)]
    groups = load_eval(paths, n, seed)
    tc = load_transcoder(candidate)
    X, Y = teacher_forced(groups, "candidate")
    means, alive, _ = components(X, Y, tc)
    rate = (tc.act(tc.input_to_features(X[:200000])) > 0).float().mean(0)
    return groups, tc, means, alive, rate


@torch.no_grad()
def dense_drop(rnn, setup, seed):
    groups, tc, means, alive, rate = setup
    dense_other = [f for f in range(tc.n_feats) if rate[f] >= 0.2 and f not in FIVE]
    sparse = [f for f in range(tc.n_feats) if 0 < rate[f] < 0.2 and alive[f]]
    rng = np.random.default_rng(seed)

    def acc(drop):
        keep = torch.ones(tc.n_feats, dtype=torch.bool)
        keep[list(drop)] = False
        fn = Ablated(keep, means, tc)
        res = []
        for g in groups:
            xx = g["inputs"]
            out = torch.cat([splice_run(rnn, xx[i:i + 4096], "candidate", fn) for i in range(0, len(xx), 4096)])
            Lg = out.shape[1]
            res.append(((out.argmax(-1) == xx[:, :Lg, :-1].argmax(-1)).float().mean().item(), Lg, len(xx)))
        tot = sum(a * m for a, _, m in res) / sum(m for _, _, m in res)
        return {"accuracy": tot, "by_length": {Lg: a for a, Lg, _ in res}, "dropped": list(map(int, drop))}
    pool = [f for f in dense_other if f not in CIRCUIT_DENSE]
    out = {"none": acc([]), "the 5": acc(FIVE), "the 15 dense circuit carriers": acc(CIRCUIT_DENSE),
           "the 5 + the 15": acc(FIVE + CIRCUIT_DENSE), "all other dense": acc(dense_other)}
    for k in range(3):
        out[f"random 15 other dense #{k}"] = acc(rng.choice(pool, 15, replace=False).tolist())
        out[f"random 15 sparse #{k}"] = acc(rng.choice(sparse, 15, replace=False).tolist())
    for f in CIRCUIT_DENSE:
        out[f"only {f}"] = acc([f])
    return out


@torch.no_grad()
def token_cond(rnn, setup, magnitude_scores):
    groups, tc, means, _, _ = setup
    sets = {r["feature"]: r["token_sets"] for r in json.load(open(magnitude_scores))["features"]}

    def preds(drop):
        keep = torch.ones(tc.n_feats, dtype=torch.bool)
        keep[list(drop)] = False
        fn = Ablated(keep, means, tc)
        P, T = [], []
        for g in groups:
            xx = g["inputs"]
            out = torch.cat([splice_run(rnn, xx[i:i + 4096], "candidate", fn) for i in range(0, len(xx), 4096)])
            Lg = out.shape[1]
            P.append(out.argmax(-1).flatten())
            T.append(xx[:, :Lg, :-1].argmax(-1).flatten())
        return torch.cat(P), torch.cat(T)
    out = {}
    for f in SIX:
        p, tgt = preds([f])
        plus = torch.tensor([TOKEN_IDX[t] for t in sets[f]["S+"][:4]])
        minus = torch.tensor([TOKEN_IDX[t] for t in sets[f]["S-"][:4]])
        inp, inm = torch.isin(tgt, plus), torch.isin(tgt, minus)
        a = lambda m: float((p[m] == tgt[m]).float().mean())
        out[f] = {"S+": sets[f]["S+"][:4], "S-": sets[f]["S-"][:4], "acc_output_in_S+": a(inp), "acc_output_in_S-": a(inm),
                  "acc_other_outputs": a(~(inp | inm)), "acc_overall": float((p == tgt).float().mean())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--magnitude_scores", required=True)
    ap.add_argument("--sections", nargs="+", default=["trace_cosine", "pop_split", "early_feedback", "dense_drop", "token_cond"])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n_dense", type=int, default=10000, help="held-out sequences (lengths 3-9) for dense_drop / token_cond")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    rnn = build_task_rnn("copy", 31, 128)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    res = {}
    x4, o4 = load_length(args.trace_dir, 4, args.n)
    if "trace_cosine" in args.sections:
        res["trace_cosine_L4"] = trace_cosine(rnn, x4, o4, 4, args.seed)
        print("trace_cosine:", json.dumps(res["trace_cosine_L4"]))
    if "pop_split" in args.sections:
        res["pop_split_L4"] = pop_split(rnn, x4, o4, 4, args.seed)
        print("pop_split:", json.dumps(res["pop_split_L4"]))
    if "early_feedback" in args.sections:
        for L in (4, 6):
            x, o = (x4, o4) if L == 4 else load_length(args.trace_dir, L, args.n)
            res[f"early_feedback_L{L}"] = early_feedback(rnn, x, o, L, args.seed)
        print("early_feedback:", json.dumps(res["early_feedback_L4"]))
    if {"dense_drop", "token_cond"} & set(args.sections):
        setup = splice_setup(args.trace_dir, args.candidate, args.n_dense, args.seed)
    if "dense_drop" in args.sections:
        res["dense_drop"] = dense_drop(rnn, setup, args.seed)
        print("dense_drop:", {k: round(v["accuracy"], 3) for k, v in res["dense_drop"].items()})
    if "token_cond" in args.sections:
        res["token_cond"] = token_cond(rnn, setup, args.magnitude_scores)
        print("token_cond:", json.dumps(res["token_cond"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), **res}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
