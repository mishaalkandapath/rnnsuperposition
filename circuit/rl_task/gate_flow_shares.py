"""How much of the next-choice change flows through each gate's features, step by step (exact contrastive flow).

Same graphs as contrast_flow.py: held-out windows, the core trial c edited (state reached, action taken, or reward
swapped), target the next choice logit a1 - a2 at (c+1).GO, features of all three transcoders as nodes. A node's
flow is how much of the target change passes through that feature (its total effect, contrast_flow.full_flow).
Per step (c.D2, (c+1).D1, (c+1).GO) and gate, mean over windows of:
  signed    sum of the gate's node flows
  abs       sum of |node flow|
  abs share the gate's abs over all three gates' abs at that step
Reset features act on h only through the candidate (r gates what the candidate reads), so their flow is nested in
the candidate's; update and candidate features write h in parallel (z-bar dn and dz (n-bar - h-bar)), so those two are
the like-for-like comparison.

    python -m circuit.rl_task.gate_flow_shares --out rl_features/rl_circuits/gate_flow_shares.json [--n_windows 100]
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from circuit.rl_task.contrast_flow import CONTRASTS, edit, full_flow, gru_run, load_model, load_windows, step_name

GATES = {"r": "reset", "z": "update", "n": "candidate"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", default="rl_features/rl_trace_sequences")
    ap.add_argument("--n_windows", type=int, default=100, help="windows per core-trial pattern")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rnn, tcs, tracer = load_model()
    windows = load_windows(args.traces, args.n_windows, args.seed)
    out = {"n_windows": len(windows), "contrasts": {}}
    for contrast in CONTRASTS:
        signed, absf, d_target = defaultdict(list), defaultdict(list), []
        for w in windows:
            c = w["c"]
            tau = 3 * c + 4
            acts = [tracer.run_forward_pass(gru_run(rnn, x, w["h0"])) for x in (w["x"], edit(w, contrast))]
            f = full_flow(tracer, acts[0], acts[1], tracer.W_choice, tau)
            d_target.append(abs(f["d_target"]))
            s, a = defaultdict(float), defaultdict(float)
            sign = np.sign(f["d_target"]) or 1.0                  # orient so the window's change is positive
            for (kd, t, _), fl in zip(f["nodes"], f["flow"].tolist()):
                key = (step_name(t, c), GATES[kd])
                s[key] += sign * fl; a[key] += abs(fl)
            for step in ("c.D2", "c+1.D1", "c+1.GO"):
                for g in GATES.values():
                    signed[(step, g)].append(s[(step, g)]); absf[(step, g)].append(a[(step, g)])
        res = {"abs_target_change_mean": round(float(np.mean(d_target)), 3), "by_step": {}}
        for step in ("c.D2", "c+1.D1", "c+1.GO"):
            tot = sum(np.mean(absf[(step, g)]) for g in GATES.values())
            res["by_step"][step] = {g: {"signed": round(float(np.mean(signed[(step, g)])), 3),
                                        "abs": round(float(np.mean(absf[(step, g)])), 3),
                                        "abs_share": round(float(np.mean(absf[(step, g)]) / max(tot, 1e-9)), 3)}
                                    for g in GATES.values()}
        out["contrasts"][contrast] = res
        print(contrast, json.dumps(res, indent=1), f"({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
