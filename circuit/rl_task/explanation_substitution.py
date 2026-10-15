"""Do the explanations carry what the model needs? Substitute them for the components they explain.

The fidelity tests (explained_fidelity.py, ranked_fidelity.py) only use which
components are labelled explained. Here the explanation's content is used:
each explained component's activation is replaced, at every step, by what its
explanation predicts, and the GRU runs with it (inputs teacher-forced from the
recorded windows of held-out episodes).

  prediction   magnitude explanation (circuit/rl_task/autointerp_magnitude.py):
               timing baseline + fitted terms; otherwise a firing predicate:
               predicate x mean activation where it holds (fitted on dev).
               GRU unit predictions are clipped to the gate's range.
  explained    final verdict explained by any recorded explanation (E)

Conditions per gate and source (transcoder features / GRU units):
  full                  nothing replaced (features: the transcoder; units: the GRU)
  E -> mean             explained components mean-ablated, the rest real
  E -> timing           explained components set to their timing baseline
                        (mean at each phase x episode decile): what timing alone carries
  E -> explanation      explained components set to their explanation's prediction
  E -> explanation, no model-output terms
                        the same, with terms that read the model's own value / policy
                        (value, choice, model_act) held at their mean: what the
                        explanation carries from the task variables alone
  E -> observed-only explanation (with belief)
                        components whose explanation read model outputs use their
                        observed-only re-explanation (which may use the ideal-observer
                        belief instead); the rest use their original explanation
  all -> mean / all -> explanation (others mean)
                        every component replaced

Recovery of the E conditions = (m - [E -> mean]) / (full - [E -> mean]).

    python -m circuit.rl_task.explanation_substitution --rnn RNN.ckpt --traces rl_features/rl_trace_sequences \\
        --transcoders reset=R.ckpt update=U.ckpt candidate=C.ckpt --out rl_features/rl_fidelity/substitution.json
"""
import argparse
import json
import os
import re

import numpy as np
import torch

from circuit.copy_task.explained_fidelity import load_transcoder
from circuit.feature_store import open_feature_store
from circuit.rl_task.autointerp import Interp, _load_traces, match_windows, n_padding
from circuit.rl_task.autointerp_magnitude import Mag
from circuit.rl_task.explained_fidelity import behaviour_metrics, run_all
from circuit.rl_task.feature_probe import run_override
from circuit.rl_task.ranked_fidelity import MAG_UNIT_GATE, UNIT_GATES, verdict_tiers
from training.splice_eval import build_task_rnn

FEATURE_GATE = {"reset": "reset", "update": "update", "candidate": "hidden"}
UNIT_RANGE = {"reset": (0.0, 1.0), "update": (0.0, 1.0), "candidate": (-1.0, 1.0)}
MODEL_OUTPUT = re.compile(r"\b(value|choice|model_act)\b")
METRICS = ("switch_agreement", "go_agreement", "value_nmae")


def eval_windows(store, trace_dir, n_windows, seed=0, picks=None):
    """Held-out (odd-episode) windows as length groups, with their step-table rows (picks: explicit store
    window indices instead of a random sample)."""
    _, sequences, _ = open_feature_store(store)
    windows = [d for _, d in _load_traces(trace_dir, ("inputs", "outputs", "core_trial", "start_high", "pattern",
                                                      "h_prevs"))]
    chosen = match_windows(sequences, windows)
    ip = Interp(store)
    row_start, episode = np.asarray(ip.cols["row_start"]), np.asarray(ip.cols["episode"])
    test = np.nonzero(episode[row_start[:-1]] % 2 == 1)[0]
    pick = np.sort(np.asarray(picks)) if picks is not None else \
        np.sort(np.random.default_rng(seed).choice(test, size=min(n_windows, len(test)), replace=False))
    by_len = {}
    for j in pick:
        wi, i = chosen[j]
        x = windows[wi]["inputs"][i]
        n_pad = n_padding(x)
        xs = x[n_pad:].reshape(-1, x.shape[-1])
        h0 = windows[wi]["h_prevs"][i][n_pad:].reshape(-1, windows[wi]["h_prevs"].shape[-1])[0]
        rows = np.arange(row_start[j], row_start[j + 1])
        assert len(rows) == len(xs), (j, len(rows), len(xs))
        g = by_len.setdefault(len(xs), {"inputs": [], "h0": [], "rows": []})
        g["inputs"].append(xs); g["h0"].append(h0); g["rows"].append(rows)
    groups = [{"inputs": torch.tensor(np.stack(g["inputs"]), dtype=torch.float32),
               "h0": torch.tensor(np.stack(g["h0"]), dtype=torch.float32), "rows": np.stack(g["rows"])}
              for _, g in sorted(by_len.items())]
    # values are kept only at the eval rows: g["pos"] indexes into eval_rows
    eval_rows, start = [], 0
    for g in groups:
        eval_rows.append(g["rows"].ravel())
        g["pos"] = (start + np.arange(g["rows"].size)).reshape(g["rows"].shape)
        start += g["rows"].size
    eval_rows = np.concatenate(eval_rows)
    print(f"eval: {len(pick)} held-out windows from the step table, {len(eval_rows)} steps", flush=True)
    return groups, ip, eval_rows


def predictions(store, ip, gate, source, i, args):
    """(prediction, prediction without model-output terms, timing baseline, recorded activation, mean, uses model output, kind)
    over all step-table rows, from the best available explanation of component i (None if none)."""
    if source == "features":
        g = FEATURE_GATE[gate]
        mag_path, firing_path = os.path.join(args.feature_magnitude, f"{g}_{i}.json"), \
            os.path.join(args.feature_results, f"{g}_{i}.json")
    else:
        g = MAG_UNIT_GATE[gate]
        mag_path, firing_path = os.path.join(args.unit_magnitude, f"{g}_{i}.json"), None
    m = Mag(store, g, i)
    dev = m.rows["dev"]
    mean = float(m.act[dev].mean())
    refined = os.path.join(getattr(args, "feature_refined", "") or "", f"{g}_{i}.json")
    if source == "features" and getattr(args, "feature_refined", "") and os.path.exists(refined):
        # refined explanation (circuit/rl_task/refined_explanations.py), used as worded: a firing explanation is zero
        # outside its predicate and its fitted terms inside it (no timing baseline); a magnitude one is the usual fit
        rec = json.load(open(refined))
        exprs, X = m.terms(rec["terms"])
        if rec.get("predicate"):
            w = np.linalg.lstsq(X[dev], m.act[dev], rcond=None)[0]
            pred = X @ w
        else:
            exprs, X, w = m.fit(rec["terms"])
            pred = m.predict(X, w)
        return pred, pred, m.timing, m.act, mean, False, "refined"
    if os.path.exists(mag_path):
        rec = json.load(open(mag_path))
        short = "short_terms" in rec                     # the explanation is the short form, not the free fit
        spec = rec["short_terms"] if short else rec["terms"]
        spec = spec if isinstance(spec, str) else " ;; ".join(spec)
        exprs, X, w = m.fit(spec, short)
        pred = m.predict(X, w)
        uses = [bool(MODEL_OUTPUT.search(e)) for e in exprs]
        Xf = X.copy()
        for k, u in enumerate(uses):
            if u:
                Xf[:, k] = X[dev, k].mean()
        return pred, m.predict(Xf, w), m.timing, m.act, mean, any(uses), "magnitude"
    if firing_path and os.path.exists(firing_path):
        rec = json.load(open(firing_path))
        if rec.get("predicate"):
            p = ip.predicate(rec["predicate"])
            level = float(m.act[dev][p[dev]].mean()) if p[dev].any() else 0.0
            pred = p * level
            uses = bool(MODEL_OUTPUT.search(rec["predicate"]))
            return pred, (np.full_like(pred, mean) if uses else pred), m.timing, m.act, mean, uses, "predicate"
    return None


def observed_prediction(store, gate, source, i, args):
    """Prediction of an observed-only re-explanation (no model-output terms, may use belief), or None."""
    d = args.feature_magnitude_observed if source == "features" else args.unit_magnitude_observed
    g = FEATURE_GATE[gate] if source == "features" else MAG_UNIT_GATE[gate]
    path = os.path.join(d, f"{g}_{i}.json") if d else None
    if not path or not os.path.exists(path):
        return None
    rec = json.load(open(path))
    spec = rec["terms"] if isinstance(rec["terms"], str) else " ;; ".join(rec["terms"])
    m = Mag(store, g, i, observed_only=True)
    exprs, X, w = m.fit(spec)
    return m.predict(X, w)


@torch.no_grad()
def run_units(rnn, x, h0, gate, override):
    """GRU readouts with chosen units of one gate replaced by override[unit] (n, T)."""
    gru, readout = rnn.layers[0], rnn.layers[-1]
    h, out = h0, []
    ids = list(override)
    for t in range(x.shape[1]):
        xt = x[:, t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        if gate == "reset" and ids:
            r = r.clone(); r[:, ids] = torch.stack([override[u][:, t] for u in ids], 1)
        if gate == "update" and ids:
            z = z.clone(); z[:, ids] = torch.stack([override[u][:, t] for u in ids], 1)
        n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        if gate == "candidate" and ids:
            n = n.clone(); n[:, ids] = torch.stack([override[u][:, t] for u in ids], 1)
        h = (1 - z) * h + z * n
        out.append(readout(h))
    return torch.stack(out, 1)


def run_condition(rnn, tc, groups, gate, values, chunk=4096):
    """values: {component: array over the eval rows}; tc None = GRU units."""
    outs = []
    for g in groups:
        ov = {c: torch.from_numpy(np.asarray(v)[g["pos"]]).float() for c, v in values.items()}
        parts = []
        for s in range(0, len(g["inputs"]), chunk):
            sub = {c: o[s:s + chunk] for c, o in ov.items()}
            x, h0 = g["inputs"][s:s + chunk], g["h0"][s:s + chunk]
            parts.append(run_units(rnn, x, h0, gate, sub) if tc is None else run_override(rnn, tc, x, h0, gate, sub))
        outs.append(torch.cat(parts))
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--store", default="rl_features/rl_chosen/chosen_rl_v1_r128_store")
    ap.add_argument("--transcoders", nargs="+", required=True, metavar="GATE=CKPT")
    ap.add_argument("--feature_results", default="rl_features/rl_autointerp")
    ap.add_argument("--feature_magnitude", default="rl_features/rl_magnitude")
    ap.add_argument("--unit_results", nargs="*", default=["rl_features/rl_neuron_d01_autointerp",
                                                         "rl_features/rl_neuron_matched_autointerp",
                                                         "rl_features/rl_neuron_d05_autointerp"])
    ap.add_argument("--unit_magnitude", default="rl_features/rl_magnitude_units")
    ap.add_argument("--feature_magnitude_observed", default="",
                    help="optional separate observed-only re-explanations (no model-output terms; may use belief); "
                         "the current magnitude results are observed-only already")
    ap.add_argument("--unit_magnitude_observed", default="")
    ap.add_argument("--gates", nargs="*", default=["reset", "update", "candidate"])
    ap.add_argument("--sources", nargs="*", default=["features", "units"], choices=["features", "units"])
    ap.add_argument("--n_windows", type=int, default=8000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    rnn = build_task_rnn("rl", input_size=8, hidden_size=48)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    rnn.eval()
    groups, ip, eval_rows = eval_windows(args.store, args.traces, args.n_windows)
    refs = [run_all(rnn, g) for g in groups]
    transcoders = dict(s.split("=", 1) for s in args.transcoders)
    results = []
    for gate in args.gates:
        for source in args.sources:
            tc = load_transcoder(transcoders[gate]) if source == "features" else None
            width = tc.n_feats if tc is not None else 48
            counts = json.load(open(os.path.join(args.store, "store.json")))["counts"][FEATURE_GATE[gate]]
            alive = [i for i in range(width) if (source == "units" or str(i) in counts)]
            tiers = verdict_tiers(gate, source, width, args)
            E = [i for i in alive if tiers[i] == 0]
            pred, pred_nm, timing, actual, mean, pred_obs = {}, {}, {}, {}, {}, {}
            uses, kinds, n_reexplained = 0, {}, 0
            for i in alive:
                p = predictions(args.store, ip, gate, source, i, args)
                if p is None:
                    continue
                pred[i], pred_nm[i], timing[i], actual[i] = (np.asarray(a[eval_rows], dtype=np.float32) for a in p[:4])
                mean[i], u, kind = p[4:]
                if i in E:
                    uses += u
                    kinds[kind] = kinds.get(kind, 0) + 1
                    # observed-only: the re-explanation if there is one; else the original if it reads no model
                    # output; else the original with those terms held at their mean
                    o = observed_prediction(args.store, gate, source, i, args) if u else None
                    n_reexplained += o is not None
                    pred_obs[i] = np.asarray(o[eval_rows], dtype=np.float32) if o is not None else \
                        (pred[i] if not u else pred_nm[i])
            if source == "units":                      # unit predictions stay inside the gate's range
                lo, hi = UNIT_RANGE[gate]
                pred = {i: np.clip(v, lo, hi) for i, v in pred.items()}
                pred_nm = {i: np.clip(v, lo, hi) for i, v in pred_nm.items()}
                pred_obs = {i: np.clip(v, lo, hi) for i, v in pred_obs.items()}
            E = [i for i in E if i in pred]
            for i in alive:                              # components without any explanation: mean only
                if i not in mean:
                    m = Mag(args.store, FEATURE_GATE[gate] if source == "features" else MAG_UNIT_GATE[gate], i)
                    mean[i] = float(m.act[m.rows["dev"]].mean())
            const = lambda c: np.full(len(eval_rows), c, dtype=np.float32)
            conds = {
                "full": {},
                "check: E -> recorded values (= full)": {i: actual[i] for i in E},
                "E -> mean": {i: const(mean[i]) for i in E},
                "E -> timing": {i: timing[i] for i in E},
                "E -> explanation": {i: pred[i] for i in E},
                "E -> explanation, no model-output terms": {i: pred_nm[i] for i in E},
                **({"E -> observed-only explanation (with belief)": {i: pred_obs[i] for i in E}} if n_reexplained else {}),
                "all -> mean": {i: const(mean[i]) for i in alive},
                "all -> explanation (others mean)": {i: (pred[i] if i in E else const(mean[i])) for i in alive},
            }
            res = {"gate": gate, "source": source, "n_alive": len(alive), "n_explained": len(E),
                   "explanation_kinds": kinds, "n_using_model_outputs": int(uses), "n_reexplained": int(n_reexplained),
                   "conditions": {}}
            for name, values in conds.items():
                s = behaviour_metrics(run_condition(rnn, tc, groups, gate, values), refs)
                res["conditions"][name] = s
            base, full = res["conditions"]["E -> mean"], res["conditions"]["full"]
            for name in ("E -> timing", "E -> explanation", "E -> explanation, no model-output terms",
                         "E -> observed-only explanation (with belief)"):
                if name not in res["conditions"]:
                    continue
                s = res["conditions"][name]
                rec = {}
                for mtr in METRICS:
                    den = (full[mtr] - base[mtr]) if mtr != "value_nmae" else (base[mtr] - full[mtr])
                    num = (s[mtr] - base[mtr]) if mtr != "value_nmae" else (base[mtr] - s[mtr])
                    rec[mtr] = float(num / den) if abs(den) > 1e-9 else None
                s["recovery_vs_E_mean"] = rec
            results.append(res)
            print(f"\n{gate} {source}: {len(E)} explained of {len(alive)} ({kinds}; {uses} use model outputs)", flush=True)
            for name, s in res["conditions"].items():
                rec = s.get("recovery_vs_E_mean")
                tail = "" if not rec else "  | recovery " + " ".join(
                    f"{k.split('_')[0]} {v:+.2f}" for k, v in rec.items() if v is not None)
                print(f"  {name:<42} switch {s['switch_agreement']:.3f} go {s['go_agreement']:.3f} value "
                      f"{s['value_nmae']:.3f} diverged {s['nonfinite_frac']:.3f}{tail}", flush=True)
            json.dump({"args": vars(args), "results": results}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
