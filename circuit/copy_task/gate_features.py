"""What the update and reset gates do, in features and their connections.

The update and reset transcoders write each gate value as an exact sum of
gate features + bias + error, and the gate features read [h_{t-1}, x_t], where
h_{t-1} is the sum of past candidate-feature writes carried forward. So:

  drivers    what drives each gate feature (non-contrastive, averaged over the
             steps where it fires): connections from candidate features at
             earlier steps (W_gate_h[f] . (C(s, t-1) * z_s * d_g) a_g, by feature
             and steps back), the current input token, and the bias.
  reads      which reset features let the token reach each candidate feature
             (contrastive: original minus token-swapped run). A candidate feature
             reads memory as W_h[g] . (r * h_{t-1}); its change splits exactly into
             reset features (a_f M_r[:, f] * dh_{t-1}), reset bias, reset error and the
             reset-gate change (dr * hbar). Shares of the reading (projection on the
             total change), by role of the candidate feature (keeper / flipper /
             other) and steps since the write.
  writes     which update features admit which carriers' writes into the token's
             trace (zbar * Da_g d_g split over update features), and which update
             features erase the stored trace (zbar * dh_{t-1}), as shares of the new /
             old trace, by steps since the write (the pop is one step after output).

    python -m circuit.copy_task.gate_features --rnn RNN.ckpt --traces T4.pt \\
        --transcoders reset=R.ckpt update=U.ckpt candidate=C.ckpt \\
        --groups features/copy_circuits/rotation_phases_L4.json \\
        --feature_results features/<run>_autointerp --magnitude_results features/<run>_magnitude --out gate_features.json
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import StackDataset

from circuit.copy_task.copy_circuits import gru_trace, step_label, swap_token
from circuit.edge_att import CircuitTracer
from circuit.copy_task.explained_fidelity import load_transcoder
from circuit.copy_task.token_circuits import label_of, load_label_sets
from training.splice_eval import build_task_rnn

torch.serialization.add_safe_globals([StackDataset])
GATE_FEATURES = {"update": [717, 650, 580, 297, 177, 901, 212], "reset": [31, 434, 312, 147, 76, 322, 477]}


def carry_matrix(z):
    T, H = z.shape
    C = torch.ones(T, T, H)
    for s in range(T):
        for t in range(s + 1, T):
            C[s, t] = C[s, t - 1] * (1 - z[t])
    return C


@torch.no_grad()
def drivers(tracer, run, L, acc):
    """Accumulate, for the listed gate features, what drives them at each step where they fire."""
    T = run["inputs"].shape[0]
    z, f_n = run["z_ts"], run["f_n"]
    C = carry_matrix(z)
    dec = tracer.hidden_transcoder.features_to_outputs.weight
    mats = {"update": (tracer.W_z_h, tracer.W_z_x, tracer.update_transcoder, "f_z"),
            "reset": (tracer.W_r_h, tracer.W_r_x, tracer.reset_transcoder, "f_r")}
    for t in range(1, T):
        for gate, feats in GATE_FEATURES.items():
            W_h, W_x, tc, key = mats[gate]
            for f in feats:
                a = float(run[key][t, f])
                if a <= 0:
                    continue
                rec = acc[(gate, f)]
                rec["n"] += 1
                rec["preact"] += float(run["pf_" + key[-1]][t, f])
                rec["input"][int(run["inputs"][t].argmax())] += float(W_x[f] @ run["inputs"][t])
                rec["bias"] += float(tc.input_to_features.bias[f])
                for s in range(t):
                    g = torch.nonzero(f_n[s] > 0).flatten()
                    if not len(g):
                        continue
                    w = (W_h[f] * C[s, t - 1] * z[s]) @ dec[:, g]           # per unit activation of g at s
                    for gg, v in zip(g.tolist(), (w * f_n[s, g]).tolist()):
                        rec["features"][(gg, t - s)] += v
                    rec["rest"] += float(W_h[f] @ (C[s, t - 1] * z[s] * (tracer.hidden_transcoder.features_to_outputs.bias
                                                                          + run["e_n"][s])))


@torch.no_grad()
def reads_writes(tracer, ro, rs, L, i, groups, acc_r, acc_w):
    """Contrastive reset-feature reads and update-feature writes / erasures for slot i."""
    T = ro["inputs"].shape[0]
    W_h = tracer.W_n_h
    M_r, M_z = tracer.M_r, tracer.M_z
    b_r = tracer.reset_transcoder.features_to_outputs.bias
    b_z = tracer.update_transcoder.features_to_outputs.bias
    dec = tracer.hidden_transcoder.features_to_outputs.weight
    grp = torch.tensor([{"keeper": 0, "flipper": 1}.get(groups.get(g), 2) for g in range(W_h.shape[0])])
    for t in range(i, min(T, L + i + 2)):
        k = t - i
        # --- reads (from t = i+1: the token is in h_{t-1}) ---
        if t > i:
            dh = ro["h_prevs"][t] - rs["h_prevs"][t]
            ar = (ro["f_r"][t] + rs["f_r"][t]) / 2
            er = (ro["e_r"][t] + rs["e_r"][t]) / 2
            dr = ro["r_ts"][t] - rs["r_ts"][t]
            hb = (ro["h_prevs"][t] + rs["h_prevs"][t]) / 2
            act = (ro["f_n"][t] > 0) | (rs["f_n"][t] > 0)
            per_f = W_h @ (M_r * dh[:, None]) * ar[None, :]                 # (Fc, Fr)
            pieces = {"bias": W_h @ (b_r * dh), "error": W_h @ (er * dh), "reset change": W_h @ (dr * hb)}
            dm = per_f.sum(1) + sum(pieces.values())
            for gi in range(3):
                sel = act & (grp == gi)
                if not sel.any():
                    continue
                rec = acc_r[(gi, k)]
                den = float((dm[sel] ** 2).sum())
                rec["den"] += den
                rec["features"] += (per_f[sel] * dm[sel][:, None]).sum(0)
                for key, v in pieces.items():
                    rec[key] += float((v[sel] * dm[sel]).sum())
        # --- writes and erasures ---
        dcur = ro["h_ts"][t] - rs["h_ts"][t]
        dprev = ro["h_prevs"][t] - rs["h_prevs"][t]
        az = (ro["f_z"][t] + rs["f_z"][t]) / 2
        ez = (ro["e_z"][t] + rs["e_z"][t]) / 2
        da = ro["f_n"][t] - rs["f_n"][t]
        rec = acc_w[k]
        norm_cur = float(dcur @ dcur) + 1e-12
        for gi in range(3):
            Wg = dec[:, grp == gi] @ da[grp == gi]                           # this group's write (before z)
            v = Wg * dcur / norm_cur
            rec[f"admit_features_{gi}"] += az * (M_z.T @ v)
            rec[f"admit_bias_{gi}"] += float(b_z @ v)
            rec[f"admit_error_{gi}"] += float(ez @ v)
        if t > i and float(dprev @ dprev) > 1e-10:
            w2 = dprev ** 2 / float(dprev @ dprev)
            rec["erase_features"] += az * (M_z.T @ w2)
            rec["erase_bias"] += float(b_z @ w2)
            rec["erase_error"] += float(ez @ w2)
            rec["n_erase"] += 1
        rec["n"] += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--transcoders", nargs=3, required=True, metavar="GATE=PATH")
    ap.add_argument("--groups", required=True, help="rotation_edges.py --phases output (keeper / flipper groups)")
    ap.add_argument("--feature_results", default=None)
    ap.add_argument("--magnitude_results", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    paths = dict(item.split("=", 1) for item in args.transcoders)
    rnn = build_task_rnn("copy", 31, 128)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    tc, upd, rst = (load_transcoder(paths[k]) for k in ("candidate", "update", "reset"))
    tracer = CircuitTracer(rnn, upd, tc, device="cpu", reset_transcoder=rst)
    labels = load_label_sets(args.feature_results, args.magnitude_results)
    groups = {f["feature"]: f["group"] for f in json.load(open(args.groups))["features"]}
    data = torch.load(args.traces, map_location="cpu", mmap=True).datasets
    L = data["outputs"].shape[1]
    tokens = data["inputs"][:, :L, :-1].argmax(-1)
    rng = np.random.default_rng(args.seed)
    distinct = np.flatnonzero([len(set(r)) == L for r in tokens.tolist()])
    rows = np.sort(rng.choice(distinct, size=min(args.n, len(distinct)), replace=False))

    acc_d = defaultdict(lambda: {"n": 0, "preact": 0.0, "bias": 0.0, "rest": 0.0,
                                 "input": defaultdict(float), "features": defaultdict(float)})
    acc_r = defaultdict(lambda: {"den": 0.0, "features": torch.zeros(rst.n_feats), "bias": 0.0, "error": 0.0, "reset change": 0.0})
    acc_w = defaultdict(lambda: defaultdict(lambda: 0.0))
    start = time.time()
    for row in rows:
        x = data["inputs"][row].float()
        orig = tokens[row].tolist()
        ro = tracer.run_forward_pass(gru_trace(rnn, x))
        drivers(tracer, ro, L, acc_d)
        for i in range(L):
            new = int(rng.choice([v for v in range(30) if v not in orig]))
            rs = tracer.run_forward_pass(gru_trace(rnn, swap_token(x, L, i, new)))
            reads_writes(tracer, ro, rs, L, i, groups, acc_r, acc_w)
    print(f"{len(rows)} sequences, {time.time() - start:.0f}s")
    lab = lambda gate, f: label_of(labels, gate, f)["label"]
    G = ["keepers", "flippers", "other candidate features"]

    # drivers
    out_d = {}
    print("\n=== what drives the gate features (mean over the steps where each fires) ===")
    for (gate, f), rec in sorted(acc_d.items()):
        n = rec["n"]
        feats = sorted(rec["features"].items(), key=lambda kv: -abs(kv[1]))[:args.top]
        inp = sorted(rec["input"].items(), key=lambda kv: -abs(kv[1]))[:3]
        by_back = defaultdict(float)
        for (g, back), v in rec["features"].items():
            by_back[back] += v / n
        out_d[f"{gate} {f}"] = {"label": lab(gate, f), "n_steps": n, "mean_preact": rec["preact"] / n,
                                "bias": rec["bias"] / n, "bias_and_error_writes": rec["rest"] / n,
                                "input": {(TOK[k] if k < len(TOK) else k): v / n for k, v in inp},
                                "by_steps_back": dict(sorted(by_back.items())),
                                "top_candidate_features": [{"feature": g, "steps_back": b, "mean": v / n, "label": lab("hidden", g)}
                                                           for (g, b), v in feats]}
        o = out_d[f"{gate} {f}"]
        print(f"{gate} {f} [{o['label'][:50]}]: fires on {n} steps, preact {o['mean_preact']:+.2f} = bias {o['bias']:+.2f} "
              f"+ input {sum(o['input'].values()):+.2f} + candidate features {sum(o['by_steps_back'].values()):+.2f} + bias/error writes {o['bias_and_error_writes']:+.2f}")
        print("    by steps back: " + " ".join(f"-{b}:{v:+.2f}" for b, v in list(o["by_steps_back"].items())[:6]))
        print("    top: " + "; ".join(f"{d['feature']}(-{d['steps_back']}) {d['mean']:+.2f} [{d['label'][:28]}]" for d in o["top_candidate_features"][:5]))

    # reads
    out_r = {}
    print("\n=== reset features: share of each candidate group's token reading (by steps since the write) ===")
    for (gi, k), rec in sorted(acc_r.items()):
        den = rec["den"] + 1e-12
        sh = rec["features"] / den
        top = torch.argsort(sh.abs(), descending=True)[:args.top].tolist()
        out_r[f"{G[gi]} +{k}"] = {"features_total": float(sh.sum()), "bias": rec["bias"] / den, "error": rec["error"] / den,
                                  "reset change": rec["reset change"] / den,
                                  "top": [{"feature": f, "share": float(sh[f]), "label": lab("reset", f)} for f in top]}
        o = out_r[f"{G[gi]} +{k}"]
        print(f"{G[gi]:>24} +{k}: features {o['features_total']:+.2f}, bias {o['bias']:+.2f}, error {o['error']:+.2f}, change {o['reset change']:+.2f} | "
              + "; ".join(f"{d['feature']} {d['share']:+.2f} [{d['label'][:24]}]" for d in o["top"][:4]))

    # writes / erasures
    out_w = {}
    print("\n=== update features: admitting each group's write into the trace, and erasing the stored trace ===")
    for k, rec in sorted(acc_w.items()):
        n = rec["n"]
        o = {}
        for gi in range(3):
            sh = rec[f"admit_features_{gi}"] / n
            top = torch.argsort(sh.abs(), descending=True)[:4].tolist()
            o[f"admit {G[gi]}"] = {"features_total": float(sh.sum()), "bias": rec[f"admit_bias_{gi}"] / n,
                                   "error": rec[f"admit_error_{gi}"] / n,
                                   "top": [{"feature": f, "share": float(sh[f]), "label": lab("update", f)} for f in top]}
        if rec["n_erase"]:
            ne = rec["n_erase"]
            sh = rec["erase_features"] / ne
            top = torch.argsort(sh.abs(), descending=True)[:args.top].tolist()
            o["erase"] = {"features_total": float(sh.sum()), "bias": rec["erase_bias"] / ne, "error": rec["erase_error"] / ne,
                          "top": [{"feature": f, "share": float(sh[f]), "label": lab("update", f)} for f in top]}
        out_w[f"+{k}"] = o
        line = f"+{k}: " + " | ".join(f"admit {G[gi].split()[0]}: feats {o[f'admit {G[gi]}']['features_total']:+.2f} bias {o[f'admit {G[gi]}']['bias']:+.2f}" for gi in range(3))
        print(line)
        if "erase" in o:
            e = o["erase"]
            print(f"      erase (share of the stored trace): features {e['features_total']:+.2f}, bias {e['bias']:+.2f} | "
                  + "; ".join(f"{d['feature']} {d['share']:+.2f} [{d['label'][:26]}]" for d in e["top"][:4]))
        for gi in range(2):
            print(f"      admit {G[gi]}: " + "; ".join(f"{d['feature']} {d['share']:+.2f} [{d['label'][:26]}]"
                                                    for d in o[f"admit {G[gi]}"]["top"][:3]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "L": L, "drivers": out_d, "reads": out_r, "writes": out_w}, open(args.out, "w"), indent=1,
              default=str)
    print(f"wrote {args.out}")


from circuit.copy_task.autointerp import TOKENS as TOK  # noqa: E402

if __name__ == "__main__":
    main()
