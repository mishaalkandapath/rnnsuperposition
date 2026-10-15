"""The components that matter individually: features and GRU units whose ablation alone hurts behaviour.

From the importance section of rl_fidelity/conditional_tests.json (each component ablated alone, to its
per-(phase, episode-decile) mean and to zero, in the real GRU). A component matters if, under either ablation, any of:
  worst choice   GO-choice drop after its worst previous-trial pattern >= 0.05
  averaged       GO-choice drop over all windows >= 0.01, or switch-agreement drop >= 0.05
  worst value    value-error (nMAE) increase at its worst pattern >= 0.05
  active choice  over the windows where it takes part: GO drop >= 0.05 or switch drop >= 0.10
  active value   over those windows: value-error increase >= 0.05
The active-window criteria have no minimum number of windows (update 131 qualifies on a single window).

    python -m circuit.rl_task.matter_sets --out rl_features/rl_fidelity/matter_sets.json
"""
import argparse
import json
from collections import Counter


def matters(r):
    hit = set()
    for mode in ("mean", "zero"):
        m = r.get(mode)
        if not m:
            continue
        if (m["worst_pattern_go"][1] or 0) >= 0.05:
            hit.add("worst choice")
        if m["overall"]["go"] >= 0.01 or m["overall"]["switch"] >= 0.05:
            hit.add("averaged")
        if (m["worst_pattern_value"][1] or 0) >= 0.05:
            hit.add("worst value")
        a = m.get("active")
        if a:
            if a["go"] >= 0.05 or a["switch"] >= 0.10:
                hit.add("active choice")
            if a["value"] >= 0.05:
                hit.add("active value")
    return hit


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tests", default="rl_features/rl_fidelity/conditional_tests.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sel, why = {}, Counter()
    for r in json.load(open(args.tests))["importance"]["components"]:
        h = matters(r)
        if h:
            sel.setdefault(f"{r['gate']}/{r['source']}", []).append(r["id"])
            why.update(f"{r['source']}: {k}" for k in h)
    json.dump(sel, open(args.out, "w"))
    print({k: len(v) for k, v in sel.items()}, dict(why))


if __name__ == "__main__":
    main()
