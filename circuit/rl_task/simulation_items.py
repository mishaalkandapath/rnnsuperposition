"""Item lists for the blind-simulation workflows (circuit/rl_task/autointerp_workflows/).

  simulation_items.json  {"features": [[gate, id], ...], "units": [...]}: every member of the behaviour-matched sets
                         (rl_fidelity/matched_sets.json), for simulation_scoring.js (each component's own windows)
  shared_items.json      [[gate, id, words], ...]: the same items with the words-only explanation each got in that
                         run (rl_simulation/, single quotes removed for shell quoting), for simulate_shared.js

    python -m circuit.rl_task.simulation_items --out_dir rl_features/rl_fidelity            # simulation_items.json
    python -m circuit.rl_task.simulation_items --out_dir rl_features/rl_fidelity --shared   # shared_items.json (after
                                                                                           # the first run)
"""
import argparse
import json
import os

FG = {"reset": "reset", "update": "update", "candidate": "hidden"}
UG = {"reset": "unit_reset", "update": "unit_update", "candidate": "unit_candidate"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--matched", default="rl_features/rl_fidelity/matched_sets.json")
    ap.add_argument("--simulations", default="rl_features/rl_simulation")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shared", action="store_true")
    args = ap.parse_args()
    if not args.shared:
        d = json.load(open(args.matched))
        items = {"features": [[FG[g], i] for g in FG for i in d[g]["features"]["ids"]],
                 "units": [[UG[g], i] for g in UG for i in d[g]["units"]["ids"]]}
        json.dump(items, open(os.path.join(args.out_dir, "simulation_items.json"), "w"))
        print(len(items["features"]), "features,", len(items["units"]), "units")
        return
    items = json.load(open(os.path.join(args.out_dir, "simulation_items.json")))
    out = []
    for g, i in items["features"] + items["units"]:
        r = json.load(open(os.path.join(args.simulations, f"{g}_{i}.json")))
        out.append([g, i, r["explanation"].replace("'", "")])
    json.dump(out, open(os.path.join(args.out_dir, "shared_items.json"), "w"))
    print(len(out), "items")


if __name__ == "__main__":
    main()
