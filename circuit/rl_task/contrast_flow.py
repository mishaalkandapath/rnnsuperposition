"""Contrastive attribution graphs for the two-step-task GRU: how one input of a trial reaches the next choice.

RL version of the copy task's exact contrastive flow (circuit/copy_task/feature_circuits.py change_flow; method in
docs/interp_circuits.md section 2). For each held-out (odd-episode) trace window whose core trial has a next trial,
the window is run twice in the real GRU from its carried state: as recorded, and with one input of the core trial c
changed (the later inputs, including the agent's later choices, are kept as recorded):

  state    the state reached, shown at c.D2, swapped S1 <-> S2 (a common trial becomes rare and vice versa)
  action   the choice, shown at c.D2, swapped a1 <-> a2 (same state reached)
  reward   the reward, shown at (c+1).D1, swapped 1 <-> 0

Target: the choice at (c+1).GO, logit a1 - logit a2 (or the value, --target value). Nodes are the features of all
three transcoders (reset r:, update z:, candidate n:) at every step. Unlike the copy task's graphs the gates are not
frozen: a gate change carries much of the flow here. Each product in the GRU step is split at the two runs'
midpoint, which is exact, so the change is an exact linear system in the features' activation changes; replacing
each JumpReLU by its secant slope between the runs (DeepLIFT rescale) splits the target's change exactly into
  input       the changed input, entering the features of its step
  error_r/z/n the change of each transcoder's error
  jump        activation changes with no preactivation change (normally 0)
with each feature's total effect (its flow) and the edges between features. Unchanged features carry no flow.

Per window: the target change, its split by source, every node's flow (its total effect on the target change) and
activation change. Aggregated per contrast and core-trial pattern (choice x state x reward): mean target change, how
often the next choice flips, flow by source and step, and the features carrying the most flow at each step.

    python -m circuit.rl_task.contrast_flow --out rl_features/rl_circuits/contrast_flow.json [--n_windows 400]
"""
import argparse
import glob
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.copy_task.explained_fidelity import load_transcoder
from circuit.edge_att import CircuitTracer
from training.splice_eval import build_task_rnn

RNN = "data/models/rl_train/run_two_step_init_low_r1/run_two_step_init_low_r1.ckpt"
TCS = {"reset": "rl_transcoders/reset_w128_mlsparse_nonorm_e100.ckpt",
       "update": "rl_transcoders/update_w256_hlsparse_nonorm_final.ckpt",
       "candidate": "rl_transcoders/candidate_w256_mlsparse_ctd_e70.ckpt"}
STORE = "rl_features/rl_chosen/chosen_rl_v1_r128_store"
PAD = 5
S1, S2, A1, A2, REW = 1, 2, 4, 5, 6                      # input columns
CONTRASTS = ("state", "action", "reward")
PHASES = ("D1", "GO", "D2")


def load_model():
    rnn = build_task_rnn("rl", input_size=8, hidden_size=48)
    rnn.load_state_dict(torch.load(RNN, map_location="cpu"))
    rnn.eval()
    tcs = {k: load_transcoder(p) for k, p in TCS.items()}
    import pickle
    meta = pickle.load(open(os.path.join(STORE, "metadata.pkl"), "rb"))
    tracer = CircuitTracer(rnn, tcs["update"], tcs["candidate"], device="cpu", reset_transcoder=tcs["reset"],
                           task="rl", input_baseline=torch.tensor(meta["input_baseline"]))
    return rnn, tcs, tracer


@torch.no_grad()
def gru_run(rnn, x, h0):
    """Real GRU on one window x (T, 8) from h0 (H,): the tracer's item dict (T, ·)."""
    gru = rnn.layers[0]
    h, rec = h0, defaultdict(list)
    for t in range(x.shape[0]):
        xt = x[t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        rec["h_prevs"].append(h); rec["r_ts"].append(r); rec["z_ts"].append(z); rec["h_new_ts"].append(n)
        h = (1 - z) * h + z * n
    item = {k: torch.stack(v) for k, v in rec.items()}
    item["inputs"] = x
    return item


def load_windows(trace_dir, n, seed):
    """Held-out windows (odd episodes) whose core trial is valid and has a next trial: unpadded inputs (T, 8),
    carried state, core trial index (in the unpadded window) and the core trial's choice / state / reward."""
    out = []
    for path in sorted(glob.glob(os.path.join(trace_dir, "*_rl.pt"))):
        d = torch.load(path, map_location="cpu", weights_only=False).datasets
        keep = (d["has_next"] & ~d["d1_error"] & (d["seq_id"] % 2 == 1)).nonzero().flatten()
        for i in keep.tolist():
            x = d["inputs"][i]
            n_pad = int((x == PAD).flatten(1).all(-1).sum())
            xs = x[n_pad:].reshape(-1, 8)
            c = int(d["core_trial"][i]) - n_pad
            d2, d1n = xs[3 * c + 2], xs[3 * c + 3]
            choice = 1 if d2[A1] == 1 else 2 if d2[A2] == 1 else 0
            state = 1 if d2[S1] == 1 else 2 if d2[S2] == 1 else 0
            reward = float(d1n[REW])
            if choice == 0 or state == 0 or reward not in (0.0, 1.0) or xs[3 * c + 4, 7] != 1:
                continue
            out.append({"x": xs, "h0": d["h_prevs"][i][n_pad:].reshape(-1, 48)[0], "c": c,
                        "choice": choice, "state": state, "reward": int(reward),
                        "trial": int(d["trial_in_episode"][i]), "file": os.path.basename(path)})
    rng = np.random.default_rng(seed)
    by_pat = defaultdict(list)
    for w in out:
        by_pat[(w["choice"], w["state"], w["reward"])].append(w)
    picked = []
    for k in sorted(by_pat):                             # the same number per pattern (as available)
        ws = by_pat[k]
        picked += [ws[j] for j in rng.choice(len(ws), size=min(n, len(ws)), replace=False)]
    return picked


def edit(w, contrast):
    x, c = w["x"].clone(), w["c"]
    if contrast == "state":
        x[3 * c + 2, [S1, S2]] = x[3 * c + 2, [S2, S1]]
    elif contrast == "action":
        x[3 * c + 2, [A1, A2]] = x[3 * c + 2, [A2, A1]]
    elif contrast == "reward":
        x[3 * c + 3, REW] = 1 - x[3 * c + 3, REW]
    return x


KINDS = (("r", "f_r", "pf_r"), ("z", "f_z", "pf_z"), ("n", "f_n", "pf_n"))     # reset, update, candidate


def full_flow(tracer, ao, asw, w_T, tau, init_delta=None):
    """Exact contrastive flow of target(original) - target(edited) through the features of all three gates.

    Every product in the GRU step is split at the two runs' midpoint (exact for a product of two factors):
      d(r h) = rbar dh + dr hbar,   dh_t = (1 - zbar) dh_{t-1} + zbar dn + dz (nbar - hbar_{t-1}),
    and every gate value is linear in its features (dr = M_r da_r + de_r, likewise z and n), so the change is a
    linear system in the features' activation changes da, closed by each JumpReLU's secant slope da = kappa dp.
    Sources: the input change, the three transcoders' error changes, the change of the carried state entering the
    window (init; zero when both runs start from the same state), and activation jumps where dp ~ 0. init_delta
    replaces the init source by a part of that change (e.g. its component in a subspace): the system (slopes, gates)
    stays that of the two runs, so the flows of complementary parts add up to the init flow.
    """
    T = tau + 1
    mats = {"r": (tracer.M_r, tracer.W_r_h, tracer.W_r_x), "z": (tracer.M_z, tracer.W_z_h, tracer.W_z_x),
            "n": (tracer.M_n, tracer.W_n_h, tracer.W_n_x)}
    nodes, da, dp = [], [], []
    for k, fk, pk in KINDS:
        nk = sorted({(t, f) for a in (ao, asw) for t, f in torch.nonzero(a[fk][:T] > 0).tolist()})
        nodes += [(k, t, f) for t, f in nk]
        if nk:
            tt, ff = torch.tensor([t for t, _ in nk]), torch.tensor([f for _, f in nk])
            da.append(ao[fk][tt, ff] - asw[fk][tt, ff]); dp.append(ao[pk][tt, ff] - asw[pk][tt, ff])
    da, dp = torch.cat(da), torch.cat(dp)
    n_nodes, H = len(nodes), ao["h_ts"].shape[1]
    kind_of = {k: torch.tensor([i for i, nd in enumerate(nodes) if nd[0] == k], dtype=torch.long) for k in "rzn"}
    pos = {k: (torch.tensor([nodes[i][1] for i in kind_of[k].tolist()], dtype=torch.long),
               torch.tensor([nodes[i][2] for i in kind_of[k].tolist()], dtype=torch.long)) for k in "rzn"}
    mid = lambda key: (ao[key][:T] + asw[key][:T]) / 2
    rb, zb, nb, hb = mid("r_ts"), mid("z_ts"), mid("h_new_ts"), mid("h_prevs")
    dx = ao["inputs"][:T] - asw["inputs"][:T]
    de = {k: ao[f"e_{k}"][:T] - asw[f"e_{k}"][:T] for k in "rzn"}
    zero = torch.zeros(T, 8)

    def lin(act, src):
        """(dp at the nodes, target change) given activation changes `act` and the sources in `src`."""
        dense = {}
        for k in "rzn":
            Fk = mats[k][0].shape[1]
            dense[k] = torch.zeros(T, Fk, dtype=act.dtype).index_put((pos[k][0], pos[k][1]), act[kind_of[k]])
        x = dx if "input" in src else zero
        e = {k: de[k] if f"error_{k}" in src else torch.zeros(T, H) for k in "rzn"}
        d0 = (ao["h_prevs"][0] - asw["h_prevs"][0]) if init_delta is None else init_delta
        dh = d0.to(act.dtype) if "init" in src else torch.zeros(H, dtype=act.dtype)
        out = {k: [] for k in "rzn"}
        for t in range(T):
            Mr, Wrh, Wrx = mats["r"]; Mz, Wzh, Wzx = mats["z"]; Mn, Wnh, Wnx = mats["n"]
            out["r"].append(Wrh @ dh + Wrx @ x[t]); out["z"].append(Wzh @ dh + Wzx @ x[t])
            dr = Mr @ dense["r"][t] + e["r"][t]
            dz = Mz @ dense["z"][t] + e["z"][t]
            out["n"].append(Wnh @ (rb[t] * dh + dr * hb[t]) + Wnx @ x[t])
            dn = Mn @ dense["n"][t] + e["n"][t]
            dh = (1 - zb[t]) * dh + zb[t] * dn + dz * (nb[t] - hb[t])
        dpv = torch.zeros(n_nodes, dtype=act.dtype)
        for k in "rzn":
            dpv[kind_of[k]] = torch.stack(out[k])[pos[k][0], pos[k][1]]
        return dpv, w_T @ dh

    with torch.no_grad():
        ok = dp.abs() > 1e-6
        kappa = torch.where(ok, da / torch.where(ok, dp, torch.ones_like(dp)), torch.zeros_like(dp))
        jump = torch.where(ok, torch.zeros_like(da), da)                 # activation change with dp ~ 0
        sources = {"input": ("input",), "error_r": ("error_r",), "error_z": ("error_z",), "error_n": ("error_n",),
                   "init": ("init",)}
        b, d = {}, {}
        for name, src in sources.items():
            b[name], d[name] = lin(torch.zeros(n_nodes), src)
    A, c = torch.func.jacfwd(lambda a: lin(a, ()))(torch.zeros(n_nodes))
    with torch.no_grad():
        K = torch.diag(kappa)
        g = torch.linalg.solve((torch.eye(n_nodes) - K @ A).T, c)          # target sensitivity to an activation change
        Dt = kappa * g                                                     # ... to a preactivation change
        into = {name: Dt * v for name, v in b.items()}
        direct = dict(d)
        direct["jump"] = float(g @ jump)
        total = float(sum(v.sum() for v in into.values())) + float(sum(direct.values()))
        real = float((ao["h_ts"][tau] - asw["h_ts"][tau]) @ w_T)
        dp_pred = A @ da + sum(b.values())
        return {"nodes": nodes, "flow": g * da, "da": da, "into": into, "edges": Dt[:, None] * A * da[None, :],
                "direct": c * da, "target": {f"direct_{k}": float(v) for k, v in direct.items()},
                "d_target": real, "check": total - real, "preact_check": float((dp_pred - dp).abs().max())}


def step_name(t, c):
    off = t // 3 - c
    return f"c{'' if off == 0 else f'{off:+d}'}.{PHASES[t % 3]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", default="rl_features/rl_trace_sequences")
    ap.add_argument("--n_windows", type=int, default=300, help="windows per core-trial pattern")
    ap.add_argument("--contrasts", nargs="+", default=list(CONTRASTS))
    ap.add_argument("--target", choices=("choice", "value"), default="choice")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rnn, tcs, tracer = load_model()
    w_T = tracer.W_choice if args.target == "choice" else tracer.W_value
    windows = load_windows(args.traces, args.n_windows, args.seed)
    print(f"{len(windows)} windows ({time.time() - t0:.0f}s)", flush=True)
    result = {"target": args.target, "n_windows_per_pattern": args.n_windows, "contrasts": {}}
    for contrast in args.contrasts:
        groups = defaultdict(lambda: {"n": 0, "d_target": [], "flip": [], "src": defaultdict(float),
                                      "node": defaultdict(float), "da": defaultdict(float),
                                      "count": defaultdict(int), "edges": defaultdict(float),
                                      "direct": defaultdict(float), "checks": []})
        for k, w in enumerate(windows):
            c = w["c"]
            tau = 3 * c + 4
            acts = [tracer.run_forward_pass(gru_run(rnn, x, w["h0"])) for x in (w["x"], edit(w, contrast))]
            f = full_flow(tracer, acts[0], acts[1], w_T, tau)
            g = groups[f"a{w['choice']}->S{w['state']} r={w['reward']}"]
            g["n"] += 1
            g["d_target"].append(f["d_target"])
            choice_o, choice_s = (float(a["choice"][tau]) for a in acts)
            g["flip"].append(np.sign(choice_o) != np.sign(choice_s))
            g["checks"].append(max(abs(f["check"]), f["preact_check"]))
            for key, v in f["into"].items():
                for (_, t, _), x in zip(f["nodes"], v.tolist()):
                    g["src"][f"{key}@{step_name(t, c)}"] += x
            for key, v in f["target"].items():
                g["src"][key] += v
            for (kd, t, fi), fl, da, dr in zip(f["nodes"], f["flow"].tolist(), f["da"].tolist(), f["direct"].tolist()):
                if da != 0 or fl != 0:
                    nm = f"{step_name(t, c)}/{kd}:{fi}"
                    g["node"][nm] += fl; g["da"][nm] += da; g["count"][nm] += 1; g["direct"][nm] += dr
            E = f["edges"]
            big = torch.nonzero(E.abs() > 0.05)
            for gi, fi in big.tolist():
                (kg, tg, fg), (kf, tf, ff) = f["nodes"][gi], f["nodes"][fi]
                g["edges"][f"{step_name(tf, c)}/{kf}:{ff} -> {step_name(tg, c)}/{kg}:{fg}"] += float(E[gi, fi])
            if k % 200 == 0:
                print(f"{contrast}: {k}/{len(windows)} ({time.time() - t0:.0f}s)", flush=True)
        out = {}
        for name, g in sorted(groups.items()):
            n = g["n"]
            mean = lambda dct: {k: round(v / n, 4) for k, v in sorted(dct.items(), key=lambda kv: -abs(kv[1]))
                                if abs(v / n) >= 1e-3}
            node = mean(g["node"])
            by_step = defaultdict(list)
            for nm, v in node.items():
                by_step[nm.split("/")[0]].append((nm.split("/")[1], v))
            out[name] = {
                "n": n, "d_target_mean": round(float(np.mean(g["d_target"])), 3),
                "d_target_sd": round(float(np.std(g["d_target"])), 3),
                "choice_flips": round(float(np.mean(g["flip"])), 3),
                "max_check": float(np.max(g["checks"])),
                "flow_by_source": mean(g["src"]),
                "flow_through_step": {s: round(sum(v for _, v in fs), 3) for s, fs in sorted(by_step.items())},
                "top_features_by_step": {s: [{"feature": fi, "flow": v,
                                              "da": round(g["da"][f"{s}/{fi}"] / n, 4),
                                              "direct": round(g["direct"][f"{s}/{fi}"] / n, 4),
                                              "windows_changed": g["count"][f"{s}/{fi}"]}
                                             for fi, v in fs[:args.top]] for s, fs in sorted(by_step.items())},
                "top_edges": dict(list(mean(g["edges"]).items())[:40]),
            }
            print(f"{contrast} {name}: n={n} d_target={out[name]['d_target_mean']:+.2f} "
                  f"flips={out[name]['choice_flips']:.2f} check={out[name]['max_check']:.1e}", flush=True)
        result["contrasts"][contrast] = out
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
