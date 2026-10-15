"""How much behaviour do the explained features carry, by explanation type (first-round explanations)?

Per gate, in the spliced transcoder with every feature outside the kept set at its phase mean (as matched_sets.py),
on 6000 held-out windows: switch / GO agreement and value nMAE against the original GRU for
  all kept / none kept
  firing: kept                  only the features with an accepted firing explanation (rl_autointerp) kept
  firing: replaced by explanation   those features replaced by their predicate x mean level
  firing or magnitude: kept     features with an accepted firing or magnitude explanation (rl_magnitude) kept
  firing or magnitude: replaced by explanation   those replaced by their explanation (magnitude short form
                                where one exists, else the predicate)
Recovery in the findings doc = (x - none kept) / (all kept - none kept).

    python -m circuit.rl_task.firing_vs_magnitude --out rl_features/rl_fidelity/firing_vs_magnitude.json
"""
import argparse
import json
import os

import numpy as np
import torch

from circuit.copy_task.explained_fidelity import load_transcoder
from circuit.rl_task.explained_fidelity import behaviour_metrics, load_eval, run_all, teacher_forced
from circuit.rl_task.explanation_substitution import eval_windows, predictions
from circuit.rl_task.matched_sets import phase_means
from circuit.rl_task.story_test import run
from training.splice_eval import build_task_rnn

FG = {"reset": "reset", "update": "update", "candidate": "hidden"}
M = ("switch_agreement", "go_agreement", "value_nmae")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rnn", default="data/models/rl_train/run_two_step_init_low_r1/run_two_step_init_low_r1.ckpt")
    ap.add_argument("--traces", default="rl_features/rl_trace_sequences")
    ap.add_argument("--store", default="rl_features/rl_chosen/chosen_rl_v1_r128_store")
    ap.add_argument("--transcoders", nargs="+", metavar="GATE=CKPT", default=[
        "reset=rl_transcoders/reset_w128_mlsparse_nonorm_e100.ckpt",
        "update=rl_transcoders/update_w256_hlsparse_nonorm_final.ckpt",
        "candidate=rl_transcoders/candidate_w256_mlsparse_ctd_e70.ckpt"])
    ap.add_argument("--feature_results", default="rl_features/rl_autointerp")
    ap.add_argument("--feature_magnitude", default="rl_features/rl_magnitude")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    rnn = build_task_rnn("rl", 8, 48)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    groups, ip, _ = eval_windows(args.store, args.traces, 6000, seed=0)
    refs = [run_all(rnn, g) for g in groups]
    big = load_eval(args.traces, 8000, seed=1)                     # phase means, as in matched_sets / story_test
    paths = dict(s.split("=", 1) for s in args.transcoders)
    firing_only = argparse.Namespace(**{**vars(args), "feature_magnitude": "/nonexistent"})
    out = {}
    for gate in ("reset", "update", "candidate"):
        tc = load_transcoder(paths[gate])
        Xb, Yb = teacher_forced(big, gate)
        means, alive = phase_means(Xb, Yb, tc)

        def verdict(d, f):
            p = os.path.join(d, f"{FG[gate]}_{f}.json")
            return json.load(open(p)).get("final_verdict") if os.path.exists(p) else None
        firing = [f for f in range(tc.n_feats) if verdict(args.feature_results, f) == "explained"]
        magnitude = [f for f in range(tc.n_feats) if verdict(args.feature_magnitude, f) == "explained"]
        both = sorted(set(firing) | set(magnitude))
        res = {"n_firing": len(firing), "n_firing_or_magnitude": len(both), "n_alive": int(alive.sum())}

        def behave(keep_ids, override):
            keep = torch.zeros(tc.n_feats, dtype=torch.bool)
            keep[keep_ids] = True
            outs = [run(rnn, g["inputs"], g["h0"], gate, keep, means, tc, {c: v[gi] for c, v in override.items()})
                    for gi, g in enumerate(groups)]
            s = behaviour_metrics(outs, refs)
            return {m: s[m] for m in M}
        res["all kept"] = behave(list(range(tc.n_feats)), {})
        res["none kept"] = behave([], {})
        for name, ids, use_mag in (("firing", firing, False), ("firing or magnitude", both, True)):
            res[f"{name}: kept"] = behave(ids, {})
            ov = {}
            for f in ids:
                p = predictions(args.store, ip, gate, "features", f, args if use_mag else firing_only)
                if p is None:
                    continue
                vals = np.asarray(p[0], dtype=np.float32)
                ov[f] = [torch.from_numpy(vals[g["rows"]]).float() for g in groups]
            res[f"{name}: replaced by explanation"] = behave(ids, ov)
        out[gate] = res
        print(f"\n{gate}: {len(firing)} explained by firing, {len(both)} by firing or magnitude "
              f"(of {int(alive.sum())} firing features)")
        for k, v in res.items():
            if isinstance(v, dict):
                print(f"  {k:42s} switch {v['switch_agreement']:.3f} go {v['go_agreement']:.3f} "
                      f"value {v['value_nmae']:.3f}", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
