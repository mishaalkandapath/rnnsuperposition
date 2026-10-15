"""Behaviour-matched sets: the fewest features, and the fewest GRU units, that reproduce the model's behaviour.

Per gate and source (transcoder features / the gate's GRU units), a set is chosen by greedy backward elimination on
a selection sample (seed 1), minimising (1 - switch) + (1 - GO) + value nMAE, and kept as long as it meets a common
target on all three: each metric recovers at least --recovery of the way from nothing kept to the full feature
transcoder (so features and units must reach the same absolute behaviour). Components not kept are set to their
mean at the current PHASE (phase is read from the gate's input: the GO signal, the state shown at D2), not to one
global mean, which would put phase-structured components off-distribution at every step. Each round removes the
cheapest components (by single-removal cost) while the target holds; a final pass removes one at a time until none
can go. Sets are reported on the held-out evaluation sample, with how many of the features that matter
(rl_fidelity/matter_sets.json) they contain.

    python -m circuit.rl_task.matched_sets --rnn RNN.ckpt --traces rl_features/rl_trace_sequences \\
        --transcoders reset=R.ckpt update=U.ckpt candidate=C.ckpt --out rl_features/rl_fidelity/matched_sets.json
"""
import argparse
import json
import time

import numpy as np
import torch

from circuit.copy_task.explained_fidelity import load_transcoder
from circuit.rl_task.explained_fidelity import load_eval, run_all, splice, teacher_forced
from training.splice_eval import build_task_rnn

METRICS = ("switch_agreement", "go_agreement", "value_nmae")


def phase_of(x):
    """Phase per step from the input (..., 8): state one-hot 0:3, action 3:6, reward 6, GO signal 7."""
    go = x[..., 7] > 0.5
    d2 = x[..., 1:3].sum(-1) > 0.5
    return torch.where(go, 1, torch.where(d2, 2, 0))


class PhaseAblated:
    """Gate with only `keep` components live, the rest at their mean for the current phase.
    tc: transcoder (features) or None (GRU units: the real gate value with ablated units)."""

    def __init__(self, keep, means, tc=None):
        self.keep, self.means, self.tc = keep.float(), means, tc        # means: (3, width)

    def __call__(self, gate_input, value):
        m = self.means[phase_of(gate_input[:, -8:])]
        if self.tc is None:
            return value * self.keep + m * (1 - self.keep)
        f = self.tc.act(self.tc.input_to_features(gate_input))
        return self.tc.features_to_outputs(f * self.keep + m * (1 - self.keep))


def phase_means(X, Y, tc):
    ph = phase_of(X[:, -8:])
    if tc is None:
        vals = Y
    else:
        vals = torch.cat([tc.act(tc.input_to_features(X[i:i + 65536])) for i in range(0, len(X), 65536)])
    alive = (vals > 0).any(0) if tc is not None else torch.ones(vals.shape[1], dtype=torch.bool)
    return torch.stack([vals[ph == k].mean(0) for k in range(3)]), alive


def cost(s):
    return (1 - s["switch_agreement"]) + (1 - s["go_agreement"]) + s["value_nmae"]


def meets(s, t):
    return s["switch_agreement"] >= t["switch_agreement"] and s["go_agreement"] >= t["go_agreement"] \
        and s["value_nmae"] <= t["value_nmae"]


def backward(rnn, groups, refs, gate, make, alive, target, log, frac=0.1):
    kept = [int(i) for i in torch.nonzero(alive).flatten()]
    width = len(alive)
    mask = lambda ids: torch.zeros(width, dtype=torch.bool).index_fill_(0, torch.tensor(ids, dtype=torch.long), True)
    ev = lambda ids: splice(rnn, groups, refs, gate, make(mask(ids)))
    if not meets(ev(kept), target):
        return kept, False
    t0 = time.time()
    while len(kept) > 1:
        costs = sorted((cost(ev([j for j in kept if j != i])), i) for i in kept)
        n_drop = max(1, int(np.ceil(frac * len(kept))))
        trial = [i for i in kept if i not in {i for _, i in costs[:n_drop]}]
        if n_drop > 1 and meets(ev(trial), target):
            kept = trial
            log(f"    {len(kept)} kept ({time.time() - t0:.0f}s)")
            continue
        # one at a time: drop the cheapest component whose removal still meets the target
        for _, i in costs:
            trial = [j for j in kept if j != i]
            if meets(ev(trial), target):
                kept = trial
                break
        else:
            break
        log(f"    {len(kept)} kept ({time.time() - t0:.0f}s)")
    return kept, True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--transcoders", nargs="+", required=True, metavar="GATE=CKPT")
    ap.add_argument("--matter", default="rl_features/rl_fidelity/matter_sets.json")
    ap.add_argument("--recovery", type=float, default=0.9)
    ap.add_argument("--gates", nargs="*", default=["reset", "update", "candidate"])
    ap.add_argument("--n_select", type=int, default=6000)
    ap.add_argument("--max_windows", type=int, default=20000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    log = lambda s: print(s, flush=True)

    rnn = build_task_rnn("rl", input_size=8, hidden_size=48)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    rnn.eval()
    ev = load_eval(args.traces, args.max_windows, seed=0)
    sel = load_eval(args.traces, args.n_select, seed=1)
    ev_refs, sel_refs = [run_all(rnn, g) for g in ev], [run_all(rnn, g) for g in sel]
    matter = json.load(open(args.matter))
    transcoders = dict(s.split("=", 1) for s in args.transcoders)
    out = {"args": vars(args)}
    for gate in args.gates:
        Xe, Ye = teacher_forced(ev, gate)
        Xs, Ys = teacher_forced(sel, gate)
        tc = load_transcoder(transcoders[gate])
        res = {}
        # target: recovery of the way from nothing kept to the full feature transcoder (selection sample)
        fm_s, f_alive = phase_means(Xs, Ys, tc)
        full = splice(rnn, sel, sel_refs, gate, PhaseAblated(torch.ones(tc.n_feats, dtype=torch.bool), fm_s, tc))
        none = splice(rnn, sel, sel_refs, gate, PhaseAblated(torch.zeros(tc.n_feats, dtype=torch.bool), fm_s, tc))
        target = {m: none[m] + args.recovery * (full[m] - none[m]) for m in METRICS}
        log(f"{gate}: target switch >= {target['switch_agreement']:.3f}, GO >= {target['go_agreement']:.3f}, "
            f"value <= {target['value_nmae']:.3f} (full features: {full['switch_agreement']:.3f} / "
            f"{full['go_agreement']:.3f} / {full['value_nmae']:.3f})")
        for source in ("units", "features"):
            if source == "features":
                means_s, alive, t = fm_s, f_alive, tc
                means_e = phase_means(Xe, Ye, tc)[0]
            else:
                means_s, alive = phase_means(Xs, Ys, None)
                means_e, t = phase_means(Xe, Ye, None)[0], None
            make = lambda keep, means=means_s, t=t: PhaseAblated(keep, means, t)
            log(f"  {source}: {int(alive.sum())} components")
            kept, ok = backward(rnn, sel, sel_refs, gate, make, alive, target, log)
            keep = torch.zeros(len(alive), dtype=torch.bool)
            keep[kept] = True
            s = splice(rnn, ev, ev_refs, gate, PhaseAblated(keep, means_e, t))
            in_matter = sorted(set(kept) & set(matter.get(f"{gate}/{source}", [])))
            res[source] = {"ids": sorted(kept), "n": len(kept), "n_total": int(alive.sum()), "met_target": ok,
                           "eval": {m: s[m] for m in METRICS}, "n_that_matter_in_set": len(in_matter),
                           "n_that_matter": len(matter.get(f"{gate}/{source}", []))}
            log(f"  {source}: {len(kept)} of {int(alive.sum())} kept -> switch {s['switch_agreement']:.3f} go "
                f"{s['go_agreement']:.3f} value {s['value_nmae']:.3f} (eval); {len(in_matter)} of the "
                f"{res[source]['n_that_matter']} that matter are in it")
        res["target"], res["full_features"], res["none"] = target, full, none
        out[gate] = res
        json.dump(out, open(args.out, "w"), indent=1)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
