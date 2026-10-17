"""Steering the credit-assignment step by its feature labels (real GRU): do the next-D1 conjunction detectors alone
decide which state a reward is credited to?

At (c+1).D1 the reward input meets the remembered state c.D2 (credit_factorial.py). For each (state, reward) cell
of the core trial, its detectors are the features (all three gates) at (c+1).D1 whose mean activation in that cell
is at least --ratio times their mean in every other cell and at least 0.05, measured on half of the windows (each
window rerun in the 4 cells with its recorded action, as in credit_factorial.py); likewise at (c+1).GO, and at c.D2
by state alone (the reward is not yet shown). On the other half, with the input left as recorded, at the steered
steps (D1; D1 + GO; D2 + D1 + GO):

  state swap   the recorded cell's detectors set to 0, the other state's (same reward) set to their mean there
  reward swap  the same with the other reward's detectors (same state; D2 is not steered)
  double swap  the other state's and the other reward's detectors: as many features moved, but a model-based agent
               credits the same action (a reward in S1 and no reward in S2 both favour a1), so the choice should
               barely move

Each also with only the recorded cell's detectors switched off, or only the other cell's switched on.
Predicted: the next choice moves as if the core trial had reached the other state / had the other reward. Score:
the share of that real input swap's effect on the next choice, (y_steer - y) / (y_swap - y), and how often the
choice follows the swap where it changes the choice; and every steering's shift in units of the real state swap's
effect, (y_steer - y) / (y_state_swap - y). Control: as many random other active features at that step
set to 0 and as many random other inactive ones set to their mean active value (swap sizes matched in count).
Gates at the steered step are reconstructions plus the run's own transcoder error (circuit_checks.run_patched).

    python -m circuit.rl_task.credit_steering --out rl_features/rl_circuits/credit_steering.json [--n_windows 100]
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.circuit_checks import run_patched
from circuit.rl_task.contrast_flow import load_model, load_windows, gru_run
from circuit.rl_task.credit_factorial import set_trial

CELLS = [(s, r) for s in (1, 2) for r in (1, 0)]
GK = {"r": "f_r", "z": "f_z", "n": "f_n"}


def cell_acts(rnn, tracer, w):
    """Clean passes of window w in its 4 (state, reward) cells, recorded action."""
    return {cell: tracer.run_forward_pass(gru_run(rnn, set_trial(w, w["choice"], *cell)[:3 * w["c"] + 5], w["h0"]))
            for cell in CELLS}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", default="rl_features/rl_trace_sequences")
    ap.add_argument("--n_windows", type=int, default=100, help="windows per core-trial pattern")
    ap.add_argument("--ratio", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rnn, tcs, tracer = load_model()
    rng = np.random.default_rng(args.seed)
    windows = load_windows(args.traces, args.n_windows, args.seed)
    order = rng.permutation(len(windows))
    sel, test = [windows[i] for i in order[::2]], [windows[i] for i in order[1::2]]
    # detectors from the selection half, per step offset (0 = c.D2, 1 = (c+1).D1, 2 = (c+1).GO)
    sums = {(o, cell): {g: 0.0 for g in GK} for o in range(3) for cell in CELLS}
    for w in sel:
        for cell, a in cell_acts(rnn, tracer, w).items():
            for o in range(3):
                for g, key in GK.items():
                    sums[(o, cell)][g] = sums[(o, cell)][g] + a[key][3 * w["c"] + 2 + o]
    mean = {key: {g: v / len(sel) for g, v in d.items()} for key, d in sums.items()}
    active_mean = {g: torch.stack([mean[k][g] for k in mean]).max(0).values for g in GK}
    groups = {}
    for o in range(3):
        for cell in CELLS:
            groups[(o, cell)] = {}
            for g in GK:
                rivals = [c for c in CELLS if (c[0] != cell[0] if o == 0 else c != cell)]
                others = torch.stack([mean[(o, c)][g] for c in rivals]).max(0).values
                groups[(o, cell)][g] = torch.nonzero((mean[(o, cell)][g] >= 0.05)
                                                     & (mean[(o, cell)][g] >= args.ratio * others)).flatten()
    names = ("c.D2", "c+1.D1", "c+1.GO")
    result = {"n_selection": len(sel), "n_test": len(test), "ratio": args.ratio,
              "detectors": {names[o]: {f"S{s}" + ("" if o == 0 else f" r={r}"): {g: v.tolist() for g, v in groups[(o, (s, r))].items()}
                                       for s, r in CELLS if o > 0 or r == 1} for o in range(3)}}
    print(json.dumps(result["detectors"]), f"({time.time() - t0:.0f}s)", flush=True)
    res = defaultdict(list)
    for k, w in enumerate(test):
        c, t, tau = w["c"], 3 * w["c"] + 3, 3 * w["c"] + 4
        acts = cell_acts(rnn, tracer, w)
        cell = (w["state"], w["reward"])
        a0 = acts[cell]
        x = set_trial(w, w["choice"], *cell)[:tau + 1][None]
        y0 = float(a0["choice"][tau])
        err = {g: a0[f"e_{g}"][None] for g in GK}
        swaps = [("state_swap", (3 - cell[0], cell[1]), at) for at in ((1,), (1, 2), (0, 1, 2))] + \
                [("reward_swap", (cell[0], 1 - cell[1]), at) for at in ((1,), (1, 2))] + \
                [("double_swap", (3 - cell[0], 1 - cell[1]), at) for at in ((1,), (1, 2), (0, 1, 2))]
        y_state = float(acts[(3 - cell[0], cell[1])]["choice"][tau])
        for name, other, at in swaps:
            steps = torch.zeros(1, tau + 1, dtype=torch.bool)
            for o in at:
                steps[0, 3 * c + 2 + o] = True
            y_swap = float(acts[other]["choice"][tau])
            for gates, control, part in [(g, ctl, pt) for g in ("all", "n", "r", "z") for ctl in (False, True)
                                         for pt in ("both", "off", "on")]:
                if part != "both" and (control or gates not in ("all", "n")):
                    continue
                mask = {g: torch.zeros(1, tau + 1, a0[GK[g]].shape[1], dtype=torch.bool) for g in GK}
                value = {g: a0[GK[g]][None].clone() for g in GK}
                for g, o in [(g, o) for g in ("rzn" if gates == "all" else gates) for o in at]:
                    t = 3 * c + 2 + o
                    off, on = groups[(o, cell)][g], groups[(o, other)][g]
                    if control:
                        act_now = a0[GK[g]][t] > 0
                        pool_off = torch.nonzero(act_now).flatten()
                        pool_on = torch.nonzero(~act_now & (active_mean[g] > 0.05)).flatten()
                        pool_off = pool_off[~torch.isin(pool_off, torch.cat([off, on]))]
                        pool_on = pool_on[~torch.isin(pool_on, torch.cat([off, on]))]
                        off = pool_off[torch.tensor(rng.permutation(len(pool_off))[:len(off)], dtype=torch.long)]
                        on = pool_on[torch.tensor(rng.permutation(len(pool_on))[:len(on)], dtype=torch.long)]
                    if part != "on":
                        mask[g][0, t, off] = True
                        value[g][0, t, off] = 0.0
                    if part != "off":
                        mask[g][0, t, on] = True
                        value[g][0, t, on] = active_mean[g][on] if control else mean[(o, other)][g][on]
                y = float(run_patched(rnn, tcs, x, w["h0"][None], mask, value, steps, err)[0, -1])
                key = f"{name}/{'+'.join(names[o] for o in at)}/{gates}" + ("/control" if control else "") + \
                    ("" if part == "both" else f"/{part}_only")
                if abs(y_swap - y0) >= 1:
                    res[key + "/share"].append((y - y0) / (y_swap - y0))
                if np.sign(y_swap) != np.sign(y0):
                    res[key + "/choice_follows"].append(float(np.sign(y) == np.sign(y_swap)))
                if abs(y_state - y0) >= 1:
                    res[key + "/shift_vs_state_swap"].append((y - y0) / (y_state - y0))
                    if gates == "all" and not control and part == "both":
                        res[f"{name}/real_input/shift_vs_state_swap"].append((y_swap - y0) / (y_state - y0))
        if k % 100 == 0:
            print(f"{k}/{len(test)} ({time.time() - t0:.0f}s)", flush=True)
    result["results"] = {key: {"rate" if key.endswith("follows") else "median":
                               round(float(np.mean(v) if key.endswith("follows") else np.median(v)), 3), "n": len(v)}
                         for key, v in sorted(res.items())}
    for key, v in result["results"].items():
        print(key, v)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
