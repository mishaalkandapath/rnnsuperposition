"""Does the sum of the words-only interpretations reproduce the model's behaviour throughout? (features vs units)

For each gate, the behaviour-matched sets (circuit/rl_task/matched_sets.py) are the fewest features / fewest GRU
units that recover 90% of the model's behaviour with every other component at its phase mean. Here each component
in a set is replaced, at every step of the shared held-out windows (circuit/rl_task/simulation_scoring.py
shared_windows), by the activation its words-only explanation implies: the blind simulator's 0-10 prediction,
mapped back with the component's own range (the same 1st-99th percentile scale the explainer saw; nothing fitted
to explanations). The rest of the gate stays at its phase mean, as in the selection. Conditions per gate and source:

  set kept      the set's real activations (the reference that meets the 90% target)
  set -> words  the set replaced by the words-implied activations
  set -> mean   the set at its phase means too (nothing kept)

Behaviour against the original GRU on the shared windows: switch / GO agreement and value nMAE; recovery =
(words - mean) / (kept - mean). Features run in the spliced transcoder (as in matched_sets), units in the real GRU.

    python -m circuit.rl_task.story_test --rnn RNN.ckpt --traces rl_features/rl_trace_sequences \\
        --transcoders reset=R.ckpt update=U.ckpt candidate=C.ckpt --simulations rl_features/rl_simulation_shared \\
        --out rl_features/rl_fidelity/story_test.json
"""
import argparse
import json
import os

import numpy as np
import torch

from circuit.copy_task.explained_fidelity import load_transcoder
from circuit.rl_task.explained_fidelity import behaviour_metrics, run_all, teacher_forced
from circuit.rl_task.explanation_substitution import eval_windows
from circuit.rl_task.matched_sets import phase_means, phase_of
from circuit.rl_task.simulation_scoring import Component, shared_windows
from training.splice_eval import build_task_rnn

FEATURE_GATE = {"reset": "reset", "update": "update", "candidate": "hidden"}
UNIT_GATE = {"reset": "unit_reset", "update": "unit_update", "candidate": "unit_candidate"}
METRICS = ("switch_agreement", "go_agreement", "value_nmae")


@torch.no_grad()
def run(rnn, x, h0, gate, keep, means, tc, override):
    """Readouts with the gate's components outside `keep` at their phase means, and components in `override`
    ({component: (n, T) values}) set to those values; tc None = GRU units (the real gate value)."""
    gru, readout = rnn.layers[0], rnn.layers[-1]
    h, out = h0, []
    ids = list(override)
    keepf = keep.float()
    for t in range(x.shape[1]):
        xt = x[:, t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        gate_in = torch.cat([r * h if gate == "candidate" else h, xt], -1)
        m = means[phase_of(xt)]
        if tc is None:
            val = {"reset": r, "update": z}.get(gate)
            if gate == "candidate":
                val = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
            v = val * keepf + m * (1 - keepf)
        else:
            f = tc.act(tc.input_to_features(gate_in))
            f = f * keepf + m * (1 - keepf)
        if ids:
            ov = torch.stack([override[i][:, t] for i in ids], 1)
            if tc is None:
                v = v.clone(); v[:, ids] = ov
            else:
                f = f.clone(); f[:, ids] = ov
        if tc is not None:
            v = tc.features_to_outputs(f)
        if gate == "reset":
            r = v
            n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        else:
            n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        if gate == "update":
            z = v
        if gate == "candidate":
            n = v
        h = (1 - z) * h + z * n
        out.append(readout(h))
    return torch.stack(out, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--store", default="rl_features/rl_chosen/chosen_rl_v1_r128_store")
    ap.add_argument("--transcoders", nargs="+", required=True, metavar="GATE=CKPT")
    ap.add_argument("--matched", default="rl_features/rl_fidelity/matched_sets.json")
    ap.add_argument("--simulations", required=True, help="score_shared records (<gate>_<id>.json with predictions)")
    ap.add_argument("--simulations_refined", nargs="*", default=[],
                    help="score_shared records of refined explanations, used instead where present (first directory "
                         "that has the component wins)")
    ap.add_argument("--gates", nargs="*", default=["reset", "update", "candidate"])
    ap.add_argument("--sources", nargs="*", default=["features", "units"], choices=["features", "units"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    rnn = build_task_rnn("rl", input_size=8, hidden_size=48)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    rnn.eval()
    from circuit.rl_task.autointerp import Interp
    ip = Interp(args.store)
    wins = shared_windows(ip)
    groups, _, _ = eval_windows(args.store, args.traces, len(wins), picks=wins)
    refs = [run_all(rnn, g) for g in groups]                         # the original GRU
    rs = np.asarray(ip.cols["row_start"])
    rows = np.concatenate([np.arange(rs[w], rs[w + 1]) for w in wins])   # the order score_shared predicts in
    matched = json.load(open(args.matched))
    transcoders = dict(s.split("=", 1) for s in args.transcoders)
    # phase means come from the matched-set selection's convention; computed here on a large held-out sample
    from circuit.rl_task.explained_fidelity import load_eval
    big = load_eval(args.traces, 8000, seed=1)
    out = {}
    for gate in args.gates:
        Xb, Yb = teacher_forced(big, gate)
        tc = load_transcoder(transcoders[gate])
        out[gate] = {}
        for source in args.sources:
            t = tc if source == "features" else None
            means, alive = phase_means(Xb, Yb, t)
            width = len(alive)
            ids = matched[gate][source]["ids"]
            keep = torch.zeros(width, dtype=torch.bool)
            keep[ids] = True
            name = FEATURE_GATE[gate] if source == "features" else UNIT_GATE[gate]
            # words-implied activations per (group, window, step)
            override, missing, sims, n_refined = {}, [], [], 0
            for c in ids:
                p = os.path.join(args.simulations, f"{name}_{c}.json")
                for d in args.simulations_refined:
                    pr = os.path.join(d, f"{name}_{c}.json")
                    if os.path.exists(pr):
                        p, n_refined = pr, n_refined + 1
                        break
                if not os.path.exists(p):
                    missing.append(c)
                    continue
                rec = json.load(open(p))
                comp = Component(args.store, name, c)
                pred = np.asarray(rec["predictions"], dtype=np.float64)
                vals = np.zeros(ip.n_rows)
                vals[rows] = comp.lo + pred / 10 * (comp.hi - comp.lo)   # back to the activation scale
                override[c] = [torch.from_numpy(vals[g["rows"]]).float() for g in groups]
                sims.append(rec["scores"]["correlation"])
            res = {}
            for cond, k, ov in (("set kept", keep, {}), ("set -> words", keep, override),
                                ("set -> mean", torch.zeros(width, dtype=torch.bool), {})):
                outs = [run(rnn, g["inputs"], g["h0"], gate, k, means, t, {c: v[gi] for c, v in ov.items()})
                        for gi, g in enumerate(groups)]
                res[cond] = {m: v for m, v in behaviour_metrics(outs, refs).items() if m in METRICS}
            rec = {}
            for m in METRICS:
                kept, words, mean = res["set kept"][m], res["set -> words"][m], res["set -> mean"][m]
                rec[m] = (words - mean) / (kept - mean) if abs(kept - mean) > 1e-9 else None
            res["recovery"] = rec
            res["n"], res["n_missing_simulations"], res["n_refined"] = len(ids), len(missing), n_refined
            res["median_simulation_correlation"] = float(np.median(sims)) if sims else None
            out[gate][source] = res
            print(f"{gate:9s} {source:8s} ({len(ids)}, {len(missing)} missing): " + " | ".join(
                f"{c}: switch {v['switch_agreement']:.3f} go {v['go_agreement']:.3f} value {v['value_nmae']:.3f}"
                for c, v in res.items() if c in ("set kept", "set -> words", "set -> mean")) +
                f" | recovery " + " ".join(f"{m.split('_')[0]} {v:+.2f}" for m, v in rec.items() if v is not None),
                flush=True)
            json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
