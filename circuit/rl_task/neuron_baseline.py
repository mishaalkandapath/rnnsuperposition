"""Neuron baseline for the RL agent: the GRU's own gate units as a feature store.

RL version of circuit/copy_task/neuron_baseline.py. Builds a store whose
"features" are thresholded GRU units over the same trace windows (and the same
store keys) as a transcoder feature store, so circuit.rl_task.autointerp and
the agent workflows run on units exactly as on features:

  neuron_update/i    z_t[i]           (update gate)
  neuron_reset/i     r_t[i]           (reset gate)
  neuron_cand_pos/i  n_t[i]           (candidate, high end)
  neuron_cand_neg/i  -n_t[i]          (candidate, low end)

Each unit is "active" on the top fraction of steps; activation = value -
threshold there. The fraction is either matched to the median firing density
of the corresponding transcoder's features (--out) or fixed (--fixed). The new
store links to the transcoder store's step table (autointerp_steps), since
rows and keys are identical.

    python -m circuit.rl_task.neuron_baseline --feature_store STORE --traces rl_features/rl_trace_sequences \\
        --out rl_features/rl_neuron_matched_store --fixed 0.01=rl_features/rl_neuron_d01_store 0.05=...
"""
import argparse
import json
import os

import numpy as np

from circuit.copy_task.neuron_baseline import UNITS
from circuit.feature_store import FeatureStoreWriter, open_feature_store
from circuit.rl_task.autointerp import _load_traces, match_windows, n_padding

KEYS = ("inputs", "outputs", "core_trial", "start_high", "pattern", "z_ts", "r_ts", "h_new_ts")


def matched_densities(store_path: str) -> dict:
    """Median fraction of steps on which a feature of each transcoder fires."""
    index = json.load(open(os.path.join(store_path, "store.json")))
    n_rows = int(np.load(os.path.join(store_path, "autointerp_steps", "step.npy"), mmap_mode="r").shape[0])
    return {gate: float(np.median([c[1] for c in counts.values()])) / n_rows
            for gate, counts in index["counts"].items()}


def unit_values(store_path: str, trace_dir: str):
    """{trace key: (rows, units)} aligned with the store's step table (padding stripped)."""
    _, sequences, _ = open_feature_store(store_path)
    windows = [d for _, d in _load_traces(trace_dir, KEYS)]
    chosen = match_windows(sequences, windows)
    row_start = np.load(os.path.join(store_path, "autointerp_steps", "row_start.npy"))
    n_units = windows[0]["z_ts"].shape[-1]
    values = {k: np.empty((int(row_start[-1]), n_units), dtype=np.float32) for k in ("z_ts", "r_ts", "h_new_ts")}
    for j, (wi, i) in enumerate(chosen):
        d = windows[wi]
        n_pad = n_padding(d["inputs"][i])
        lo, hi = int(row_start[j]), int(row_start[j + 1])
        for k in values:
            v = d[k][i][n_pad:].reshape(-1, n_units)
            if len(v) != hi - lo:
                raise ValueError(f"window {j}: {len(v)} steps but the step table has {hi - lo}")
            values[k][lo:hi] = v
    return sequences, values, row_start


def build_store(out, feature_store, sequences, values, row_start, density, density_label):
    row_seq = np.repeat(np.arange(len(sequences)), np.diff(row_start))
    writer = FeatureStoreWriter(out, sequences, metadata={
        "kind": "neuron_baseline", "task": "rl", "feature_store": feature_store, "density": density,
        "density_rule": density_label,
        "units": {k: {"trace_key": v[0], "sign": v[1], "density_from": v[2]} for k, v in UNITS.items()}})
    thresholds = {}
    for gate, (key, sign, match) in UNITS.items():
        vals = sign * values[key]
        k = max(1, int(round(density[match] * len(vals))))
        thr = np.partition(vals, len(vals) - k, axis=0)[len(vals) - k]      # per-unit top-k cut
        thresholds[gate] = thr.tolist()
        for i in range(vals.shape[1]):
            rows = np.nonzero(vals[:, i] > thr[i])[0]
            if not len(rows):
                continue
            seqs = row_seq[rows]
            starts = np.r_[0, np.nonzero(np.diff(seqs))[0] + 1, len(rows)]
            pos, mag = rows - row_start[seqs], vals[rows, i] - thr[i]
            writer.write_feature(gate, i, {
                sequences[seqs[a]]: {"positions": pos[a:b].tolist(), "magnitudes": mag[a:b].tolist()}
                for a, b in zip(starts[:-1], starts[1:])})
        print(f"{out}: {gate}: {vals.shape[1]} units, ~{k} active steps each (of {len(vals)})", flush=True)
    writer.close(gates=list(UNITS))
    with open(os.path.join(out, "thresholds.json"), "w") as f:
        json.dump(thresholds, f)
    link = os.path.join(out, "autointerp_steps")
    if not os.path.exists(link):
        os.symlink(os.path.abspath(os.path.join(feature_store, "autointerp_steps")), link)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--feature_store", required=True, help="transcoder feature store (keys, step table, densities)")
    ap.add_argument("--traces", required=True, help="directory of *_rl.pt trace-window files")
    ap.add_argument("--out", help="output store for densities matched to the transcoders")
    ap.add_argument("--fixed", nargs="*", default=[], metavar="FRACTION=OUT",
                    help="also build stores where every unit is active on this fraction of steps, e.g. 0.01=store_d01")
    ap.add_argument("--raw", help="also save the raw unit values (r, z, n; rows of the step table) here, for "
                                  "magnitude explanations (circuit/rl_task/autointerp_magnitude.py)")
    args = ap.parse_args()
    if not args.out and not args.fixed and not args.raw:
        ap.error("nothing to build: pass --out (matched densities), --fixed and/or --raw")

    sequences, values, row_start = unit_values(args.feature_store, args.traces)
    print(f"loaded unit values: {len(row_start) - 1} windows, {int(row_start[-1])} steps", flush=True)
    if args.raw:
        os.makedirs(args.raw, exist_ok=True)
        for key, name in (("r_ts", "reset"), ("z_ts", "update"), ("h_new_ts", "candidate")):
            np.save(os.path.join(args.raw, f"{name}.npy"), values[key])
        json.dump({"feature_store": args.feature_store, "traces": args.traces, "rows": int(row_start[-1]),
                   "files": {"reset": "r_t", "update": "z_t", "candidate": "n_t"}},
                  open(os.path.join(args.raw, "meta.json"), "w"))
        print(f"saved raw unit values to {args.raw}", flush=True)
    if args.out:
        density = matched_densities(args.feature_store)
        print("matched densities:", {k: f"{v:.4%}" for k, v in density.items()}, flush=True)
        build_store(args.out, args.feature_store, sequences, values, row_start, density,
                    "matched to median transcoder feature")
    for spec in args.fixed:
        frac, out = spec.split("=")
        frac = float(frac)
        build_store(out, args.feature_store, sequences, values, row_start,
                    {g: frac for g in ("reset", "update", "hidden")}, f"fixed {frac:.2%} of steps for every unit")


if __name__ == "__main__":
    main()
