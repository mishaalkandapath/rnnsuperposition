"""Token-conditional versions of three analyses that averaged over all sequences (real GRU).

A feature that matters only when its token is present looks unimportant when its effect is averaged over every
sequence. These sections redo three such analyses with the average taken only where the feature takes part:

  importance  every candidate feature mean-ablated alone (kind 2, per-step mean); accuracy drop over all sequences
              and over the sequences where the feature is active (activation > 0 at some step), plus how often
              the error is a repeat of the previous output (a failed pop)
  pop         the clamp test of circuit/copy_task/clamp_tests.py on every "copy input X" feature (label
              'phase == 2' and 'in_tok == tok("X")', any verdict): off -> is X repeated? on -> is X erased early?
              Also how much each one writes when active (decoder norm x activation, and its one-step direct
              effect on X's logit), to tell used features from inert duplicates with the same label
  ablations   every candidate feature: decoder norm x activation (and the part the update gate lets into h), and
              zero / time-shift ablations (its per-step mean taken from the step before or after), which remove
              timing features that a per-step mean cannot
  flows       the exact contrastive flow (circuit/copy_task/feature_circuits.change_flow) and write angles
              (circuit/copy_task/rotation_edges.feature_angles) per (sequence, slot), averaged per feature over all
              (sequence, slot) pairs and over only the pairs it takes part in (|flow| > --min_flow at some step)

    python -m circuit.copy_task.conditional_analyses --rnn RNN.ckpt --traces T4.pt T7.pt \\
        --transcoders reset=R.ckpt update=U.ckpt candidate=C.ckpt --firing features/<run>_autointerp \\
        --sections importance pop flows --out_dir features/copy_circuits
"""
import argparse
import glob
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import StackDataset

from circuit.copy_task.autointerp import TOKEN_IDX, TOKENS
from circuit.copy_task.circuit_tests import clean_acts, feature_hook, run
from circuit.copy_task.clamp_tests import SIMPLE_PREDICATE, firing_test
from circuit.copy_task.copy_circuits import gru_trace, swap_token
from circuit.copy_task.explained_fidelity import load_transcoder
from circuit.copy_task.feature_circuits import change_flow
from circuit.copy_task.rotation_edges import circ_mean, feature_angles
from circuit.edge_att import CircuitTracer
from training.splice_eval import build_task_rnn

torch.serialization.add_safe_globals([StackDataset])


def load_data(path, n):
    data = torch.load(path, map_location="cpu", mmap=True).datasets
    L = data["outputs"].shape[1]
    x = data["inputs"][:n].float()
    return x, x[:, :L, :-1].argmax(-1), L


def importance(rnn, tcs, traces, n):
    out = {}
    tc = tcs["candidate"]
    for path in traces:
        x, orig, L = load_data(path, n)
        acts = clean_acts(rnn, x, L, tcs)["candidate"]                 # (N, T, F)
        mean = acts.mean(0)
        active = (acts > 0).any(1)                                     # (N, F)
        base = run(rnn, x, L).argmax(-1) == orig
        prev_differs = torch.zeros_like(orig, dtype=torch.bool)
        prev_differs[:, 1:] = orig[:, 1:] != orig[:, :-1]
        rec = defaultdict(list)
        start = time.time()
        for f in range(tc.n_feats):
            pred = run(rnn, x, L, feature_hook(tcs, "candidate", [f], lambda t: mean[t][None])).argmax(-1)
            ok = pred == orig
            rep = torch.zeros_like(ok)
            rep[:, 1:] = (pred[:, 1:] == orig[:, :-1]) & prev_differs[:, 1:]
            a = active[:, f]
            rec["overall_drop"].append(float(base.float().mean() - ok.float().mean()))
            rec["active_drop"].append(float(base[a].float().mean() - ok[a].float().mean()) if a.any() else 0.0)
            rec["inactive_drop"].append(float(base[~a].float().mean() - ok[~a].float().mean()) if (~a).any() else 0.0)
            rec["active_repeat"].append(float(rep[a].float().mean()) if a.any() else 0.0)
            rec["seq_rate"].append(float(a.float().mean()))
            # per output token: the drop on outputs of that token (treats dense and sparse features alike)
            n_tok = torch.bincount(orig.flatten(), minlength=30).float()
            d_tok = (torch.bincount(orig.flatten(), weights=(base.float() - ok.float()).flatten(), minlength=30)
                     / n_tok.clamp_min(1))
            rec["drop_by_output_token"].append(d_tok.tolist())
        out[f"L{L}"] = dict(rec)
        print(f"importance L={L}: {time.time() - start:.0f}s")
    return out


def write_sizes(rnn, tc, x, L, feats, tokens):
    """How much each feature actually writes when it is active (clean real-GRU run): decoder norm x mean activation,
    and the one-step direct effect on its token's logit, (z * d_f) . W_out[X] * a (z = update gate)."""
    gru, W = rnn.layers[0], rnn.layers[-1].weight
    h = torch.zeros(len(x), gru.hidden_size)
    acts, zs = [], []
    for t in range(2 * L):
        xt = x[:, t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        acts.append(tc.act(tc.input_to_features(torch.cat([r * h, xt], -1))))
        zs.append(z)
        h = (1 - z) * h + z * n
    acts, zs = torch.stack(acts, 1), torch.stack(zs, 1)
    D = tc.features_to_outputs.weight
    out = {}
    for f, X in zip(feats, tokens):
        m = acts[:, :, f] > 0
        if not m.any():
            continue
        a = acts[:, :, f][m]
        direct = (zs[m] * D[:, f]) @ W[X] * a
        out[f] = {"decoder_norm_x_activation": float(D[:, f].norm() * a.mean()), "mean_activation": float(a.mean()),
                  "direct_dlogit_X": float(direct.mean())}
    return out


def other_ablations(rnn, tcs, traces, n):
    """Every candidate feature: how much it writes when active, and zero / time-shift ablations, which can remove
    a feature that a per-step mean cannot (a feature with the same value in every sequence at a step)."""
    tc = tcs["candidate"]
    gru = rnn.layers[0]
    D = tc.features_to_outputs.weight
    out = {}
    for path in traces:
        x, orig, L = load_data(path, n)
        T = 2 * L
        h = torch.zeros(len(x), gru.hidden_size)
        acts, zs = [], []
        for t in range(T):
            xt = x[:, t]
            r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
            z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
            nn_ = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
            acts.append(tc.act(tc.input_to_features(torch.cat([r * h, xt], -1))))
            zs.append(z)
            h = (1 - z) * h + z * nn_
        acts, zs = torch.stack(acts, 1), torch.stack(zs, 1)
        mean = acts.mean(0)
        active = (acts > 0).any(1)
        base = run(rnn, x, L).argmax(-1) == orig
        n_tok = torch.bincount(orig.flatten(), minlength=30).float().clamp_min(1)
        modes = {"zero": lambda t: torch.zeros_like(mean[t])[None],
                 "shift_earlier": lambda t: mean[max(t - 1, 0)][None],      # value of the step before
                 "shift_later": lambda t: mean[min(t + 1, T - 1)][None]}    # value of the step after
        rec = defaultdict(list)
        start = time.time()
        for f in range(tc.n_feats):
            m = acts[:, :, f] > 0
            a = acts[:, :, f][m]
            rec["decoder_norm_x_activation"].append(float(D[:, f].norm() * a.mean()) if m.any() else 0.0)
            rec["gated_write"].append(float(((zs[m] * D[:, f]).norm(dim=-1) * a).mean()) if m.any() else 0.0)
            for name, repl in modes.items():
                ok = run(rnn, x, L, feature_hook(tcs, "candidate", [f], repl)).argmax(-1) == orig
                d_tok = torch.bincount(orig.flatten(), weights=(base.float() - ok.float()).flatten(), minlength=30) / n_tok
                af = active[:, f]
                rec[f"{name}_overall_drop"].append(float(base.float().mean() - ok.float().mean()))
                rec[f"{name}_active_drop"].append(float(base[af].float().mean() - ok[af].float().mean()) if af.any() else 0.0)
                rec[f"{name}_worst_token_drop"].append(float(d_tok.max()))
        out[f"L{L}"] = dict(rec)
        print(f"ablations L={L}: {time.time() - start:.0f}s")
    return out


def pop(rnn, tcs, traces, n, firing, seed):
    pool = sorted(f for f, r in firing.items() if "phase == 2" in (r.get("predicate") or "")
                  and len(SIMPLE_PREDICATE.findall(r["predicate"])) == 1 and "prev_tok" not in r["predicate"])
    rng = np.random.default_rng(seed)
    out = {str(f): {"label": firing[f]["label"], "predicate": firing[f]["predicate"],
                    "verdict": firing[f].get("final_verdict"),
                    "token": SIMPLE_PREDICATE.findall(firing[f]["predicate"])[0]} for f in pool}
    for path in traces:
        x, _, L = load_data(path, n)
        acts = clean_acts(rnn, x, L, tcs)["candidate"]
        base_lg = run(rnn, x, L)
        for f in pool:
            r = firing[f]
            X = TOKEN_IDX[out[str(f)]["token"]]
            out[str(f)][f"L{L}"] = firing_test(rnn, tcs["candidate"], x, L, acts, base_lg, f, r["predicate"], X, rng)
        sizes = write_sizes(rnn, tcs["candidate"], x, L, pool, [TOKEN_IDX[out[str(f)]["token"]] for f in pool])
        for f, s in sizes.items():
            out[str(f)][f"L{L}"]["write"] = s
        print(f"pop L={L}: {len(pool)} features")
    return out


def flows(rnn, tcs, trace_path, n, min_flow, seed):
    tracer = CircuitTracer(rnn, tcs["update"], tcs["candidate"], device="cpu", reset_transcoder=tcs["reset"])
    x, orig, L = load_data(trace_path, n)
    acts = clean_acts(rnn, x, L, tcs)["candidate"]
    seq_rate = (acts > 0).any(1).float().mean(0)
    rng = np.random.default_rng(seed)
    flow_sum = defaultdict(float)                                       # feature -> sum of |flow| over steps
    takes_part = defaultdict(int)                                       # feature -> (sequence, slot) pairs
    by_step = defaultdict(lambda: [0.0, 0])                             # (slot, t, f) -> [sum |flow|, pairs]
    angles = defaultdict(lambda: ([], []))
    start = time.time()
    for s in range(len(x)):
        o = orig[s].tolist()
        run_o = tracer.run_forward_pass(gru_trace(rnn, x[s]))
        for i in range(L):
            new = int(rng.choice([v for v in range(30) if v not in o]))
            run_s = tracer.run_forward_pass(gru_trace(rnn, swap_token(x[s], L, i, new)))
            fl = change_flow(tracer, run_o, run_s, L, o, i)
            per_f = defaultdict(float)
            for (t, f), v in fl["nodes"].items():
                if abs(v) > 0:
                    per_f[f] += abs(v)
                    if abs(v) > min_flow:
                        by_step[(i, t, f)][0] += abs(v)
                        by_step[(i, t, f)][1] += 1
            for f, v in per_f.items():
                flow_sum[f] += v
                if v > min_flow:
                    takes_part[f] += 1
            fa, _ = feature_angles(tracer, run_o, run_s, L, i, o, new)
            for t, g, ang, w, _ in fa:
                angles[g][0].append(ang)
                angles[g][1].append(w)
        if s % 50 == 49:
            print(f"flows: {s + 1}/{len(x)} sequences ({time.time() - start:.0f}s)")
    pairs = len(x) * L
    feats = []
    for f in sorted(flow_sum):
        k = takes_part[f]
        ang, w = angles[f]
        m, R = circ_mean(ang, w) if ang and sum(w) > 0 else (None, None)
        feats.append({"feature": int(f), "seq_rate": float(seq_rate[f]),
                      "mean_abs_flow_all": flow_sum[f] / pairs,
                      "share_of_pairs": k / pairs,
                      "mean_abs_flow_when_taking_part": flow_sum[f] / k if k else 0.0,
                      "write_angle": m, "write_R": R,
                      "write_weight_all": float(sum(w)) / pairs,
                      "write_weight_when_taking_part": float(sum(w)) / k if k else 0.0})
    steps = [{"slot": i, "t": t, "feature": f, "mean_abs_flow_when_taking_part": v / c, "share_of_pairs": c / len(x)}
             for (i, t, f), (v, c) in by_step.items()]
    return {"L": L, "n": len(x), "min_flow": min_flow, "features": feats, "by_step": steps}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--traces", nargs="+", required=True, help="flows uses the first one")
    ap.add_argument("--transcoders", nargs=3, required=True, metavar="GATE=PATH")
    ap.add_argument("--firing", required=True, help="firing autointerp results dir (hidden_<f>.json)")
    ap.add_argument("--sections", nargs="+", default=["importance", "pop", "flows"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n_flow", type=int, default=300)
    ap.add_argument("--min_flow", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    paths = dict(item.split("=", 1) for item in args.transcoders)
    rnn = build_task_rnn("copy", 31, 128)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    tcs = {k: load_transcoder(paths[k]) for k in ("reset", "update", "candidate")}
    firing = {}
    for p in glob.glob(os.path.join(args.firing, "hidden_*.json")):
        r = json.load(open(p))
        firing[r["feature"]] = r
    os.makedirs(args.out_dir, exist_ok=True)

    def save(name, res):
        path = os.path.join(args.out_dir, name)
        json.dump({"args": vars(args), **res}, open(path, "w"), indent=1)
        print(f"wrote {path}")

    if "importance" in args.sections:
        save("conditional_importance.json", importance(rnn, tcs, args.traces, args.n))
    if "ablations" in args.sections:
        save("conditional_ablations.json", other_ablations(rnn, tcs, args.traces, args.n))
    if "pop" in args.sections:
        save("conditional_pop.json", {"features": pop(rnn, tcs, args.traces, min(args.n, 1000), firing, args.seed)})
    if "flows" in args.sections:
        save("conditional_flows.json", flows(rnn, tcs, args.traces[0], args.n_flow, args.min_flow, args.seed))


if __name__ == "__main__":
    main()
