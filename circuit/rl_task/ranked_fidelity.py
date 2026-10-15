"""Budget-matched fidelity: do the best-explained components carry the behaviour?

circuit/rl_task/explained_fidelity.py keeps every component above an F1
threshold. With 48 GRU units that keeps nearly all of them (at 5% density 45-48
units have F1 >= 0.5), so fidelity is high whatever the explanations say. Here
every source (transcoder features of a gate, or that gate's GRU units) is
instead ranked, and the top-k are kept (the rest mean-ablated) at matched
budgets: fractions of the source's firing components and absolute counts.

  explanation score  best held-out score of any recorded explanation: predicate
                     F1 (features: rl_autointerp; units: max over the 1%, matched
                     and 5% runs and both candidate ends) or magnitude R^2 beyond
                     timing (rl_magnitude, rl_magnitude_units). Ties broken at random.
  orders             explained (by score), random (--n_random draws), oracle (by
                     single-drop behaviour loss on a separate selection sample:
                     switch drop + GO drop + value-error increase)
  keep top-k         behaviour with only the top-k kept
  drop top-k         behaviour with the top-k removed (the complement test)

Each metric is normalised to recovery = (m - none) / (full - none) (value
error: (none - m) / (none - full)); full = all components (features: the
transcoder; units: the exact GRU), none = all mean-ablated. The summary per
source is the explained order's position between random (0) and oracle (1),
averaged over budgets.

    python -m circuit.rl_task.ranked_fidelity --rnn RNN.ckpt --traces rl_features/rl_trace_sequences \\
        --transcoders reset=R.ckpt update=U.ckpt candidate=C.ckpt --out rl_features/rl_fidelity/ranked.json
"""
import argparse
import json
import os

import numpy as np
import torch

from circuit.copy_task.explained_fidelity import Ablated, components, load_transcoder
from circuit.rl_task.explained_fidelity import load_eval, run_all, splice, teacher_forced
from training.splice_eval import build_task_rnn

FEATURE_GATE = {"reset": "reset", "update": "update", "candidate": "hidden"}
UNIT_GATES = {"reset": ["neuron_reset"], "update": ["neuron_update"], "candidate": ["neuron_cand_pos", "neuron_cand_neg"]}
MAG_UNIT_GATE = {"reset": "unit_reset", "update": "unit_update", "candidate": "unit_candidate"}
METRICS = ("switch_agreement", "go_agreement", "value_nmae")


def _score(path, key):
    if not os.path.exists(path):
        return 0.0
    r = json.load(open(path))
    s = (r.get("test_scores") or {}).get(key)
    return float(s) if s is not None else 0.0


def explanation_scores(gate, source, width, args):
    """Best held-out explanation score per component (0 if none)."""
    out = np.zeros(width)
    for i in range(width):
        if source == "features":
            g = FEATURE_GATE[gate]
            vals = [_score(os.path.join(args.feature_results, f"{g}_{i}.json"), "f1"),
                    _score(os.path.join(args.feature_magnitude, f"{g}_{i}.json"), "r2_beyond_timing")]
        else:
            vals = [_score(os.path.join(d, f"{ug}_{i}.json"), "f1") for d in args.unit_results for ug in UNIT_GATES[gate]]
            vals.append(_score(os.path.join(args.unit_magnitude, f"{MAG_UNIT_GATE[gate]}_{i}.json"), "r2_beyond_timing"))
        out[i] = max(vals)
    return out


def verdict_tiers(gate, source, width, args):
    """0 = final verdict explained by any recorded explanation (firing or magnitude), 1 = partially explained,
    2 = otherwise. Scores of different kinds (F1, R^2) are not comparable, so verdicts order first."""
    def verdict(path):
        return json.load(open(path)).get("final_verdict") if os.path.exists(path) else None
    out = np.full(width, 2)
    for i in range(width):
        if source == "features":
            g = FEATURE_GATE[gate]
            vs = [verdict(os.path.join(args.feature_results, f"{g}_{i}.json")),
                  verdict(os.path.join(args.feature_magnitude, f"{g}_{i}.json"))]
        else:
            vs = [verdict(os.path.join(d, f"{ug}_{i}.json")) for d in args.unit_results for ug in UNIT_GATES[gate]]
            vs.append(verdict(os.path.join(args.unit_magnitude, f"{MAG_UNIT_GATE[gate]}_{i}.json")))
        out[i] = 0 if "explained" in vs else 1 if "partially_explained" in vs else 2
    return out


def loss(s, full):
    return (full["switch_agreement"] - s["switch_agreement"]) + (full["go_agreement"] - s["go_agreement"]) + \
           (s["value_nmae"] - full["value_nmae"])


def single_drops(rnn, groups, refs, gate, make_fn, width, alive, cached=None):
    """Behaviour loss when each component alone is removed (selection sample)."""
    full = splice(rnn, groups, refs, gate, make_fn(torch.ones(width, dtype=torch.bool)))
    out = np.zeros(width)
    for i in torch.nonzero(alive).flatten().tolist():
        if cached is not None and i in cached:
            d = cached[i]
            s = {"switch_agreement": full["switch_agreement"] - d["switch_drop"],
                 "go_agreement": full["go_agreement"] - d["go_drop"], "value_nmae": d["value_nmae"]}
        else:
            keep = torch.ones(width, dtype=torch.bool)
            keep[i] = False
            s = splice(rnn, groups, refs, gate, make_fn(keep))
        out[i] = loss(s, full)
    return out


def cached_feature_drops(gate):
    """Earlier single-drop runs on the same selection sample (seed 1, 8000 windows), if present."""
    if gate == "reset":
        p = "rl_features/rl_fidelity/reset_sets.json"
        drops = json.load(open(p))["drops"] if os.path.exists(p) else []
    else:
        p = "rl_features/rl_fidelity/single_drops.json"
        drops = json.load(open(p)).get(gate, {}).get("drops", []) if os.path.exists(p) else []
    return {d["feature"]: d for d in drops}


def recovery(s, full, none):
    out = {}
    for m in METRICS:
        den = (full[m] - none[m]) if m != "value_nmae" else (none[m] - full[m])
        num = (s[m] - none[m]) if m != "value_nmae" else (none[m] - s[m])
        out[m] = float(num / den) if abs(den) > 1e-9 else float("nan")
    return out


def evaluate_source(rnn, ev, ev_refs, sel, sel_refs, gate, source, make_ev, make_sel, width, alive, score, args, rng,
                    tiers=None, previous=None):
    """previous: (rows, summary) of an earlier run of this source; its orders and rows are reused."""
    pool = torch.nonzero(alive).flatten().numpy()
    n_alive = len(pool)
    if previous is not None:
        # reuse only what does not depend on the explanations (oracle and random orders); explanation orders
        # are rebuilt from the current scores
        prev_rows, prev_summary = previous
        prev_rows = [r for r in prev_rows if not r["order"].startswith("explained")]
        previous = (prev_rows, prev_summary)
        oracle_loss = np.array(prev_summary["oracle_loss"])
        orders = {}
        for r in prev_rows:
            orders.setdefault(r["order"], {})[r["k"]] = r["ids"]
        # rebuild full orders for reused names from their longest prefix (enough for budgets up to its k)
        orders = {n: by_k[max(by_k)] for n, by_k in orders.items()}
        tie = rng.random(width)
        orders["explained"] = sorted(pool, key=lambda i: (-score[i], tie[i]))
    else:
        cached = cached_feature_drops(gate) if source == "features" else None
        oracle_loss = single_drops(rnn, sel, sel_refs, gate, make_sel, width, alive, cached)
        tie = rng.random(width)
        orders = {"explained": sorted(pool, key=lambda i: (-score[i], tie[i])),
                  "oracle": sorted(pool, key=lambda i: -oracle_loss[i])}
        for r in range(args.n_random):
            orders[f"random #{r}"] = list(rng.permutation(pool))
    if previous is not None:                             # full-length orders for any new budget
        orders["oracle"] = sorted(pool, key=lambda i: -oracle_loss[i])
        prefix = list(orders["explained"])
        orders["explained"] = prefix + sorted((i for i in pool if i not in set(prefix)), key=lambda i: -score[i])
        for r in range(args.n_random):
            name = f"random #{r}"
            prefix = list(orders.get(name, []))
            rest = [i for i in rng.permutation(pool) if i not in set(prefix)]
            orders[name] = prefix + rest
    budgets = sorted({max(1, int(round(f * n_alive))) for f in args.fractions} | {k for k in args.counts if k < n_alive})
    if tiers is not None:
        tie = rng.random(width)
        orders["explained (verdict first)"] = sorted(pool, key=lambda i: (tiers[i], -score[i], tie[i]))
        n_explained = int((tiers[pool] == 0).sum())
        if 0 < n_explained < n_alive:
            budgets = sorted(set(budgets) | {n_explained})
    mask = lambda ids: torch.zeros(width, dtype=torch.bool).index_fill_(0, torch.tensor(ids, dtype=torch.long), True) \
        if len(ids) else torch.zeros(width, dtype=torch.bool)
    full = splice(rnn, ev, ev_refs, gate, make_ev(torch.ones(width, dtype=torch.bool)))
    none = splice(rnn, ev, ev_refs, gate, make_ev(torch.zeros(width, dtype=torch.bool)))
    rows = list(previous[0]) if previous is not None else []
    done = {(r["k"], r["order"], r["mode"]) for r in rows}
    for k in budgets:
        for name, order in orders.items():
            for mode in ("keep", "drop"):
                if (k, name, mode) in done:
                    continue
                keep = mask(order[:k]) if mode == "keep" else (~mask(order[:k]))
                s = splice(rnn, ev, ev_refs, gate, make_ev(keep))
                row = {"gate": gate, "source": source, "k": k, "fraction": k / n_alive, "order": name, "mode": mode,
                       "ids": [int(i) for i in order[:k]], "mean_score": float(np.mean(score[order[:k]])),
                       **{m: s[m] for m in METRICS}, "recovery": recovery(s, full, none),
                       "nonfinite_frac": s["nonfinite_frac"]}
                rows.append(row)
        rec = {n: next(r for r in rows if r["k"] == k and r["order"] == n and r["mode"] == "keep")["recovery"]
               for n in ("explained", "oracle")}
        rand = np.mean([r["recovery"]["switch_agreement"] for r in rows if r["k"] == k and r["mode"] == "keep"
                        and r["order"].startswith("random")])
        print(f"  {gate:9s} {source:8s} k={k:3d} ({k / n_alive:4.0%}) keep: switch recovery explained "
              f"{rec['explained']['switch_agreement']:+.2f} random {rand:+.2f} oracle {rec['oracle']['switch_agreement']:+.2f}"
              f" | value recovery explained {rec['explained']['value_nmae']:+.2f} oracle {rec['oracle']['value_nmae']:+.2f}",
              flush=True)
    summary = {"gate": gate, "source": source, "n_alive": n_alive, "budgets": budgets, "full": full, "none": none,
               "scores": score.tolist(), "oracle_loss": oracle_loss.tolist(), "position": {}}
    if tiers is not None:
        summary["tiers"] = tiers.tolist()
        summary["n_explained"] = int((tiers[pool] == 0).sum())
    explanation_orders = [n for n in ("explained", "explained (verdict first)") if n in orders]
    for name in explanation_orders:
        for mode in ("keep", "drop"):
            for m in METRICS:
                pos = []
                for k in budgets:
                    get = lambda n: [r["recovery"][m] for r in rows if r["k"] == k and r["mode"] == mode and r["order"] == n]
                    e, o = get(name)[0], get("oracle")[0]
                    rnd = np.mean([r["recovery"][m] for r in rows if r["k"] == k and r["mode"] == mode
                                   and r["order"].startswith("random")])
                    if abs(o - rnd) > 0.02:                      # skip budgets where oracle ~ random (no room)
                        pos.append((e - rnd) / (o - rnd))
                key = f"{mode} {m}" if name == "explained" else f"{name}: {mode} {m}"
                summary["position"][key] = float(np.mean(pos)) if pos else None
    print(f"  {gate} {source}: position between random (0) and oracle (1): " +
          ", ".join(f"{k} {v:+.2f}" for k, v in summary["position"].items() if v is not None), flush=True)
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--transcoders", nargs="+", required=True, metavar="GATE=CKPT")
    ap.add_argument("--feature_results", default="rl_features/rl_autointerp")
    ap.add_argument("--feature_magnitude", default="rl_features/rl_magnitude")
    ap.add_argument("--unit_results", nargs="*", default=["rl_features/rl_neuron_d01_autointerp",
                                                         "rl_features/rl_neuron_matched_autointerp",
                                                         "rl_features/rl_neuron_d05_autointerp"])
    ap.add_argument("--unit_magnitude", default="rl_features/rl_magnitude_units")
    ap.add_argument("--gates", nargs="*", default=["reset", "update", "candidate"])
    ap.add_argument("--sources", nargs="*", default=["features", "units"], choices=["features", "units"])
    ap.add_argument("--fractions", nargs="*", type=float, default=[0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--counts", nargs="*", type=int, default=[2, 5, 10, 20])
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--max_windows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiers", action="store_true",
                    help="also order by verdict first (explained by any explanation, then partial, then the rest), "
                         "with the size of the explained set as an extra budget")
    ap.add_argument("--reuse", help="earlier output of this script: reuse its orders, oracle and rows")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    reuse = json.load(open(args.reuse)) if args.reuse else None
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(args.seed)

    rnn = build_task_rnn("rl", input_size=8, hidden_size=48)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    rnn.eval()
    ev = load_eval(args.traces, args.max_windows, seed=0)
    sel = load_eval(args.traces, 8000, seed=1)                     # selection sample for the oracle order
    ev_refs, sel_refs = [run_all(rnn, g) for g in ev], [run_all(rnn, g) for g in sel]
    transcoders = dict(s.split("=", 1) for s in args.transcoders)
    rows, summaries = [], []
    for gate in args.gates:
        Xe, Ye = teacher_forced(ev, gate)
        Xs, Ys = teacher_forced(sel, gate)
        for source in args.sources:
            # means from each sample's own steps (as in the earlier single-drop runs)
            if source == "features":
                tc = load_transcoder(transcoders[gate])
                means, alive, _ = components(Xe, Ye, tc)
                means_sel = components(Xs, Ys, tc)[0]
                make_ev = lambda keep, tc=tc, means=means: Ablated(keep, means, tc)
                make_sel = lambda keep, tc=tc, means=means_sel: Ablated(keep, means, tc)
                width = tc.n_feats
            else:
                means, alive, _ = components(Xe, Ye)
                means_sel = components(Xs, Ys)[0]
                make_ev = lambda keep, means=means: Ablated(keep, means)
                make_sel = lambda keep, means=means_sel: Ablated(keep, means)
                width = Ye.shape[1]
            score = explanation_scores(gate, source, width, args)
            tiers = verdict_tiers(gate, source, width, args) if args.tiers else None
            previous = None
            if reuse:
                prev_s = next((x for x in reuse["summaries"] if x["gate"] == gate and x["source"] == source), None)
                if prev_s:
                    previous = ([x for x in reuse["rows"] if x["gate"] == gate and x["source"] == source], prev_s)
            r, s = evaluate_source(rnn, ev, ev_refs, sel, sel_refs, gate, source, make_ev, make_sel, width, alive,
                                   score, args, rng, tiers, previous)
            rows += r
            summaries.append(s)
            json.dump({"args": vars(args), "rows": rows, "summaries": summaries}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
