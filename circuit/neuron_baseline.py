"""Neuron baseline: the GRU's own gate units as a feature store.

Builds a store (circuit/feature_store.py) whose "features" are thresholded GRU
units, so the interpretation pipeline (circuit/autointerp.py, circuit/scorecards.py)
runs on neurons exactly as on transcoder features:

  neuron_update/i    z_t[i]           (update gate)
  neuron_reset/i     r_t[i]           (reset gate)
  neuron_cand_pos/i  n_t[i]           (candidate, high end)
  neuron_cand_neg/i  -n_t[i]          (candidate, low end)

Each unit is "active" on the top fraction of its steps; the fraction is matched
to the median firing density of the corresponding transcoder's features, so
neurons and features are compared at the same sparsity. Activation = value -
threshold on active steps.

    python -m circuit.neuron_baseline TRACE_FILES... --feature_store STORE --out OUT_STORE
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import StackDataset

from circuit.autointerp import TOKENS
from circuit.feature_store import FeatureStoreWriter

torch.serialization.add_safe_globals([StackDataset])

# neuron gate name -> (trace key, sign, transcoder gate whose density it matches)
UNITS = {"neuron_update": ("z_ts", 1, "update"), "neuron_reset": ("r_ts", 1, "reset"),
         "neuron_cand_pos": ("h_new_ts", 1, "hidden"), "neuron_cand_neg": ("h_new_ts", -1, "hidden")}


def matched_densities(store_path: str) -> dict:
    """Median fraction of steps on which a feature of each transcoder fires."""
    index = json.load(open(f"{store_path}/store.json"))
    n_steps = int(np.load(f"{store_path}/autointerp_steps/t.npy", mmap_mode="r").shape[0])
    return {gate: float(np.median([c[1] for c in counts.values()])) / n_steps
            for gate, counts in index["counts"].items()}


def load_traces(paths):
    """Sequences (store order) and per-key (rows, units) arrays, rows = (sequence, step).

    Arrays are preallocated from a first pass over shapes so loading never
    holds two full copies (concatenation would).
    """
    shapes = []
    for path in paths:
        cols = torch.load(path, map_location="cpu", mmap=True).datasets
        shapes.append(tuple(cols["z_ts"].shape))
        del cols
    n_rows = sum(n * t for n, t, _ in shapes)
    values = {k: np.empty((n_rows, shapes[0][2]), dtype=np.float32) for k in ("z_ts", "r_ts", "h_new_ts")}
    sequences, row = [], 0
    for path, (n, t, _) in zip(paths, shapes):
        cols = torch.load(path, map_location="cpu", mmap=True).datasets
        sequences += [tuple(TOKENS[i] for i in r) for r in cols["inputs"].argmax(-1).tolist()]
        for k in values:
            values[k][row:row + n * t] = cols[k].reshape(n * t, -1).numpy()
        row += n * t
        del cols
        print(f"loaded {path}", flush=True)
    return sequences, values


def build_store(out, sequences, values, density, traces, density_label):
    """Write one neuron store; density = {transcoder gate: fraction of steps active}."""
    lengths = np.array([len(s) for s in sequences])
    row_start = np.concatenate([[0], np.cumsum(lengths)])
    row_seq = np.repeat(np.arange(len(sequences)), lengths)
    row_t = np.arange(len(row_seq)) - row_start[row_seq]
    writer = FeatureStoreWriter(out, sequences, metadata={
        "kind": "neuron_baseline", "traces": traces, "density": density, "density_rule": density_label,
        "matched_densities": density, "units": {
            k: {"trace_key": v[0], "sign": v[1], "density_from": v[2]} for k, v in UNITS.items()}})
    thresholds = {}
    for gate, (key, sign, match) in UNITS.items():
        vals = sign * values[key]
        k = max(1, int(round(density[match] * len(vals))))
        thr = np.partition(vals, len(vals) - k, axis=0)[len(vals) - k]      # per-unit top-k cut
        thresholds[gate] = thr.tolist()
        for i in range(vals.shape[1]):
            rows = np.nonzero(vals[:, i] > thr[i])[0]
            acts = {}
            for r in rows:
                entry = acts.setdefault(sequences[row_seq[r]], {"positions": [], "magnitudes": []})
                entry["positions"].append(int(row_t[r]))
                entry["magnitudes"].append(float(vals[r, i] - thr[i]))
            if acts:
                writer.write_feature(gate, i, acts)
        print(f"{out}: {gate}: {vals.shape[1]} units, ~{k} active steps each (of {len(vals)})", flush=True)
        del vals
    writer.close(gates=list(UNITS))
    with open(f"{out}/thresholds.json", "w") as f:
        json.dump(thresholds, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+", help="trace files (one sequence length each)")
    ap.add_argument("--feature_store", help="transcoder feature store, for matched densities")
    ap.add_argument("--out", help="output store for matched densities")
    ap.add_argument("--fixed", nargs="*", default=[], metavar="FRACTION=OUT",
                    help="also build stores where every unit is active on this fraction of steps, e.g. 0.01=store_d01")
    args = ap.parse_args()
    if not args.out and not args.fixed:
        ap.error("nothing to build: pass --out (matched densities) and/or --fixed")

    sequences, values = load_traces(args.traces)
    if args.out:
        if not args.feature_store:
            ap.error("--out (matched densities) needs --feature_store")
        density = matched_densities(args.feature_store)
        print("matched densities:", {k: f"{v:.4%}" for k, v in density.items()}, flush=True)
        build_store(args.out, sequences, values, density, args.traces, "matched to median transcoder feature")
    for spec in args.fixed:
        frac, out = spec.split("=")
        frac = float(frac)
        build_store(out, sequences, values, {g: frac for g in ("reset", "update", "hidden")}, args.traces,
                    f"fixed {frac:.2%} of steps for every unit")


if __name__ == "__main__":
    main()
