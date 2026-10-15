"""Evidence for re-explaining components that fail per-component substitution (the revisit round).

For each component (default: those failing the substitute section of conditional_tests.json): its recorded
explanation, the substitution result, its activation in its worst previous-trial pattern (mean / std by phase,
test split) and the clamp push (on minus off, from the diagnose section).

rl_fidelity/revisit_evidence.json is a snapshot from before the revisit round (Oct 13): it holds the 6 components
that failed then (update 5, 18; candidate 8, 204; units candidate 0, 3) with their explanations at the time.
conditional_tests.json and the explanation records have changed since, so re-running gives today's failures.

    python -m circuit.rl_task.revisit_evidence --out rl_features/rl_fidelity/revisit_evidence.json \\
        [--components update/features/5 candidate/units/0 ...]
"""
import argparse
import json
import os

import numpy as np

from circuit.rl_task.autointerp_magnitude import PATTERNS, Mag

G = {"reset": "reset", "update": "update", "candidate": "hidden"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tests", default="rl_features/rl_fidelity/conditional_tests.json")
    ap.add_argument("--store", default="rl_features/rl_chosen/chosen_rl_v1_r128_store")
    ap.add_argument("--components", nargs="*", default=None, metavar="GATE/SOURCE/ID")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = json.load(open(args.tests))
    sub = {(r["gate"], r["source"], r["id"]): r for r in d["substitute"]}
    dg = {(r["gate"], r["source"], r["id"]): r for r in d.get("diagnose", [])}
    imp = {(r["gate"], r["source"], r["id"]): r for r in d["importance"]["components"]}
    comps = ([(g, s, int(i)) for g, s, i in (c.split("/") for c in args.components)] if args.components
             else [k for k, r in sub.items() if not r["pass"]])
    ev = {}
    for gate, src, i in comps:
        r = sub.get((gate, src, i))
        g = G[gate] if src == "features" else f"unit_{gate}"
        mp = (f"rl_features/rl_magnitude/{g}_{i}.json" if src == "features"
              else f"rl_features/rl_magnitude_units/{g}_{i}.json")
        rec = json.load(open(mp)) if os.path.exists(mp) else json.load(open(f"rl_features/rl_autointerp/{g}_{i}.json"))
        m = Mag(args.store, g, i, observed_only=True)
        worst = r["worst_pattern"] if r else imp[(gate, src, i)]["mean"]["worst_pattern_go"][0]
        test = m.rows["test"]
        sel = test[m.last[test] == PATTERNS.index(worst)]
        a, ph = m.act[sel], np.asarray(m.ip.cols["phase"])[sel]
        e = {"explanation": rec.get("short_terms") or rec.get("predicate"), "label": rec.get("final_label") or rec.get("label"),
             "worst_pattern": worst,
             "activation_in_worst_pattern_mean_std_by_phase": {
                 "D1 GO D2".split()[p]: (round(float(a[ph == p].mean()), 3), round(float(a[ph == p].std()), 3))
                 for p in range(3)}}
        if r:
            e.update(kind=r["kind"], go_agreement=[round(r["ablated_go"], 3), round(r["substituted_go"], 3)],
                     recovery=None if r["recovery"] is None else round(r["recovery"], 2))
        if (gate, src, i) in dg:
            e["clamp_push_on_minus_off"] = {k: round(dg[(gate, src, i)]["on_single"][f"d_{k}"]
                                                     - dg[(gate, src, i)]["off_single"][f"d_{k}"], 3)
                                            for k in ("logit", "value")}
        ev[f"{g}/{i}"] = e
        print(f"{g}/{i}: {json.dumps(e)}")
    json.dump(ev, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
