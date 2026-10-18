"""How much of the GRU's computation runs through explained features or units?

For one gate at a time, keep a subset of components (transcoder features, or
the GRU's own gate units) and mean-ablate the rest, then measure

  fve      teacher-forced: fraction of variance of the true gate value
           explained by the ablated gate, on the stored trace inputs;
  splice   the ablated gate runs inside the GRU with the recurrence, so errors
           compound: KL(original || ablated) at the output steps, top-1
           agreement with the original, and copy accuracy.

Kept subsets:
  full / none   all components / none (only means)
  f1>=tau       components whose explanation scores held-out F1 >= tau
  random        same number of components drawn at random from those that fire
                (control: does choosing explained ones beat chance?)
  core          the fewest components, by single-ablation importance, that
                recover --core_frac of the full set's fve. Its F1 breakdown says
                whether the components that matter are the explained ones.

Transcoder features replace the gate (as in training/splice_eval.py); GRU units
are the real gate with ablated units. Candidate units have two explanations
(high end, low end); a unit counts as explained if either end is (favours units).

Units were interpreted on the trace files given here, with odd store indices
as their held-out split; the eval uses those held-out sequences.

    python -m circuit.copy_task.explained_fidelity --rnn RNN.ckpt --traces T3.pt ... T9.pt \\
        --transcoders reset=R.ckpt update=U.ckpt candidate=C.ckpt \\
        --feature_results features/<run>_autointerp --feature_store features/<run>_store \\
        --units matched=features/<run>_neuron_baseline_autointerp d01=... d05=... \\
        --unit_store features/<run>_neuron_d01_store --out fidelity.json
"""
import argparse
import hashlib
import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import StackDataset

from models.transcoders import Transcoder
from training.splice_eval import build_task_rnn

torch.serialization.add_safe_globals([StackDataset])

# splice target -> trace key of the true gate value, feature-results gate, unit-results gates
GATES = {"reset": ("r_ts", "reset", ["neuron_reset"]),
         "update": ("z_ts", "update", ["neuron_update"]),
         "candidate": ("h_new_ts", "hidden", ["neuron_cand_pos", "neuron_cand_neg"])}
CHUNK = 65536


def load_eval(paths, max_sequences, seed=0):
    """Held-out sequences (odd global index, files in the given order), grouped by file."""
    sizes = [torch.load(p, map_location="cpu", mmap=True).datasets["inputs"].shape[0] for p in paths]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    odd = np.arange(1, starts[-1], 2)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(odd, size=min(max_sequences, len(odd)), replace=False))
    groups = []
    for path, a, b in zip(paths, starts[:-1], starts[1:]):
        rows = torch.from_numpy(chosen[(chosen >= a) & (chosen < b)] - a)
        if not len(rows):
            continue
        cols = torch.load(path, map_location="cpu", mmap=True).datasets
        groups.append({k: cols[k][rows].float().clone() for k in ("inputs", "h_prevs", "r_ts", "z_ts", "h_new_ts")})
        del cols
    print(f"eval: {len(chosen)} held-out sequences, {sum(g['inputs'].shape[0] * g['inputs'].shape[1] for g in groups)} steps")
    return groups


def teacher_forced(groups, gate):
    """Flattened (gate input, true gate value) over all eval steps."""
    xs, ys = [], []
    for g in groups:
        h, x = g["h_prevs"], g["inputs"]
        state = g["r_ts"] * h if gate == "candidate" else h
        xs.append(torch.cat([state, x], -1).flatten(0, 1))
        ys.append(g[GATES[gate][0]].flatten(0, 1))
    return torch.cat(xs), torch.cat(ys)


class Ablated:
    """Gate value with only `keep` components live; the rest at their mean."""

    def __init__(self, keep, means, transcoder=None):
        self.keep, self.means, self.tc = keep.float(), means, transcoder

    def __call__(self, gate_input, value):
        if self.tc is None:                       # GRU units: ablate the true value
            return value * self.keep + self.means * (1 - self.keep)
        f = self.tc.act(self.tc.input_to_features(gate_input))
        return self.tc.features_to_outputs(f * self.keep + self.means * (1 - self.keep))


@torch.no_grad()
def fve(X, Y, fn):
    sse = sum(((Y[i:i + CHUNK] - fn(X[i:i + CHUNK], Y[i:i + CHUNK])) ** 2).sum().item()
              for i in range(0, len(X), CHUNK))
    return 1 - sse / ((Y - Y.mean(0)) ** 2).sum().item()


@torch.no_grad()
def run(rnn, x, gate=None, fn=None):
    """Logits at the output steps with the target gate replaced by fn (None: original GRU)."""
    gru, readout = rnn.layers[0], rnn.layers[-1]
    h = torch.zeros(x.shape[0], gru.hidden_size)
    steps, out = x.shape[1], []
    for t in range(steps):
        xt = x[:, t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        if gate == "reset":
            r = fn(torch.cat([h, xt], -1), r)
        if gate == "update":
            z = fn(torch.cat([h, xt], -1), z)
        n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        if gate == "candidate":
            n = fn(torch.cat([r * h, xt], -1), n)
        h = (1 - z) * h + z * n
        if t >= steps - steps // 2:
            out.append(readout(h))
    return torch.stack(out, 1)


@torch.no_grad()
def splice(rnn, groups, references, gate, fn, per_sequence=False):
    """Splice metrics; per_sequence also returns (agreeing outputs, outputs) per eval sequence, for bootstraps."""
    kl = agree = correct = n = 0.0
    seq_agree, seq_n = [], []
    for g, ref in zip(groups, references):
        x = g["inputs"]
        out = torch.cat([run(rnn, x[i:i + 4096], gate, fn) for i in range(0, len(x), 4096)])
        true = x[:, :ref.shape[1], :-1].argmax(-1)
        log_p, log_q = F.log_softmax(ref, -1), F.log_softmax(out, -1)
        kl += (log_p.exp() * (log_p - log_q)).sum().item()
        same = ref.argmax(-1) == out.argmax(-1)
        agree += same.sum().item()
        correct += (out.argmax(-1) == true).sum().item()
        n += true.numel()
        seq_agree.append(same.sum(1).numpy())
        seq_n.append(np.full(len(same), same.shape[1]))
    res = {"splice_kl": kl / n, "top1_agreement": agree / n, "task_acc": correct / n}
    if per_sequence:
        return res, np.concatenate(seq_agree), np.concatenate(seq_n)
    return res


def bootstrap(agree, n_out, n_boot=2000, seed=0):
    """Bootstrap replicates of top-1 agreement, resampling evaluation sequences (same draws for every selection,
    so differences between selections are paired)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(agree), size=(n_boot, len(agree)))
    return agree[idx].sum(1) / n_out[idx].sum(1)


def ci(reps):
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def compare_selections(rnn, groups, refs, gate, ids, feature_f1, feature_fn, unit_sources, taus):
    """Each F1 selection of transcoder features, alone and with a fixed set added (e.g. the dense features no
    predicate explains), against the same F1 selection of GRU units at each density. All are scored on the same
    evaluation sequences; significance is a paired bootstrap over sequences: the 95% interval of
    (features - units) and the two-sided p-value 2 * min(P(diff <= 0), P(diff >= 0)).

    unit_sources: {name: (f1, make_fn)}."""
    width = len(feature_f1)
    s = torch.zeros(width, dtype=torch.bool)
    s[ids] = True

    def evaluate(keep, fn):
        res, a, n = splice(rnn, groups, refs, gate, fn(keep), per_sequence=True)
        return res["top1_agreement"], bootstrap(a, n)

    def entry(top1, reps, k, n_total):
        return {"top1": top1, "ci": ci(reps), "n_kept": k, "n_total": n_total}

    out = {"gate": gate, "set": list(ids), "selections": []}
    top1, reps = evaluate(torch.ones(width, dtype=torch.bool), feature_fn)
    out["full"] = entry(top1, reps, width, width)
    for tau in taus:
        row = {"tau": tau}
        reps_of = {}
        for name, keep in (("features", feature_f1 >= tau), ("features + set", (feature_f1 >= tau) | s)):
            top1, reps = evaluate(keep, feature_fn)
            row[name] = entry(top1, reps, int(keep.sum()), width)
            reps_of[name] = reps
        for name, (uf1, ufn) in unit_sources.items():
            keep = uf1 >= tau
            top1, reps = evaluate(keep, ufn)
            row[f"units {name}"] = entry(top1, reps, int(keep.sum()), len(uf1))
            for fname in ("features", "features + set"):
                diff = reps_of[fname] - reps
                row[f"units {name}"][f"{fname} minus units"] = {
                    "diff": row[fname]["top1"] - top1, "ci": ci(diff),
                    "p": float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))}
        out["selections"].append(row)
        msg = " | ".join(f"{k} {v['top1']:.3f} ({v['n_kept']})" for k, v in row.items() if isinstance(v, dict))
        print(f"  F1 >= {tau}: {msg}", flush=True)
    return out


@torch.no_grad()
def components(X, Y, tc=None):
    """Means, which components fire, and single-ablation importance (fve lost)."""
    sst = ((Y - Y.mean(0)) ** 2).sum()
    if tc is None:
        means = Y.mean(0)
        return means, torch.ones(Y.shape[1], dtype=torch.bool), ((Y - means) ** 2).sum(0) / sst
    feats = lambda i: tc.act(tc.input_to_features(X[i:i + CHUNK]))
    total, fired = 0, 0
    for i in range(0, len(X), CHUNK):
        f = feats(i)
        total, fired = total + f.sum(0), fired + (f > 0).sum(0)
    means = total / len(X)
    D = tc.features_to_outputs.weight                                        # (out, feats)
    imp = 0
    for i in range(0, len(X), CHUNK):
        f = feats(i)
        a, e = f - means, Y[i:i + CHUNK] - tc.features_to_outputs(f)          # residual of the full transcoder
        # sse(i ablated) - sse(full) = sum_t 2 a_ti d_i.e_t + a_ti^2 |d_i|^2
        imp = imp + 2 * (a * (e @ D)).sum(0) + (a ** 2).sum(0) * (D ** 2).sum(0)
    return means, fired > 0, imp / sst


def core_set(X, Y, importance, make_fn, target):
    """Smallest top-importance set whose fve reaches target (bisection on k)."""
    order = torch.argsort(importance, descending=True)
    mask = lambda k: torch.zeros(len(order), dtype=torch.bool).index_fill_(0, order[:k], True)
    lo, hi = 0, len(order)
    while lo < hi:
        mid = (lo + hi) // 2
        if fve(X, Y, make_fn(mask(mid))) >= target:
            hi = mid
        else:
            lo = mid + 1
    return mask(lo)


def load_f1(results_dir, result_gates, width):
    """Held-out F1 per component (0 where there is no explanation); max over result gates."""
    f1 = np.zeros((len(result_gates), width))
    for j, rg in enumerate(result_gates):
        for i in range(width):
            p = os.path.join(results_dir, f"{rg}_{i}.json")
            if os.path.exists(p):
                f1[j, i] = (json.load(open(p)).get("test_scores") or {}).get("f1") or 0.0
    return torch.from_numpy(f1.max(0)), f1


def load_transcoder(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)["transcoder"]
    n_feats, n_in = sd["input_to_features.weight"].shape
    tc = Transcoder(n_in, sd["features_to_outputs.weight"].shape[0], n_feats)
    tc.load_state_dict(sd)
    return tc.eval()


def check_fingerprints(feature_store, paths):
    meta = pickle.load(open(os.path.join(feature_store, "metadata.pkl"), "rb"))
    expected = meta.get("checkpoint_fingerprints", {})
    for key, path in paths.items():
        want = expected.get(GATES[key][1] if key in GATES else key)
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if want and want != got:
            print(f"WARNING: {path} does not match the checkpoint the features were extracted with ({key})")


def f1_bands(f1):
    f1 = np.asarray(f1)
    return {"n": int(len(f1)), "median_f1": float(np.median(f1)) if len(f1) else None,
            ">=0.8": int((f1 >= 0.8).sum()), "0.5-0.8": int(((f1 >= 0.5) & (f1 < 0.8)).sum()),
            "0.2-0.5": int(((f1 >= 0.2) & (f1 < 0.5)).sum()), "<0.2": int((f1 < 0.2).sum())}


def evaluate_source(rnn, groups, refs, X, Y, gate, source, f1, means, alive, importance, make_fn,
                    taus, n_random, core_frac, seed):
    rows, rng = [], np.random.default_rng(seed)
    width = len(f1)

    def add(selection, keep, tau=None, rep=None):
        fn = make_fn(keep)
        row = {"gate": gate, "source": source, "selection": selection, "tau": tau, "rep": rep,
               "n_kept": int(keep.sum()), "n_total": width, "fve": fve(X, Y, fn)}
        row.update(splice(rnn, groups, refs, gate, fn))
        rows.append(row)
        print(f"  {source:>12} {selection:>8}{'' if tau is None else f' {tau}':>5}{'' if rep is None else f' #{rep}':>4}"
              f"  kept {row['n_kept']:4d}/{width}  fve {row['fve']:.3f}  kl {row['splice_kl']:.4f}"
              f"  top1 {row['top1_agreement']:.3f}  acc {row['task_acc']:.3f}", flush=True)
        return row

    full = add("full", torch.ones(width, dtype=torch.bool))
    add("none", torch.zeros(width, dtype=torch.bool))
    pool = torch.nonzero(alive).flatten().numpy()
    for tau in taus:
        keep = f1 >= tau
        add("f1", keep, tau)
        for rep in range(n_random):
            rand = torch.zeros(width, dtype=torch.bool)
            rand[rng.choice(pool, size=min(int(keep.sum()), len(pool)), replace=False)] = True
            add("random", rand, tau, rep)
    core = core_set(X, Y, importance, make_fn, core_frac * full["fve"])
    add("core", core)
    summary = {"gate": gate, "source": source, "core_frac": core_frac, "full_fve": full["fve"],
               "core": f1_bands(f1[core].numpy()), "alive": f1_bands(f1[alive].numpy()),
               "all": f1_bands(f1.numpy()), "core_importance_share": float(importance[core].sum() / importance.sum())}
    print(f"  {source:>12} core set: {summary['core']}")
    return rows, summary


def drop_tests(rnn, groups, refs, X, Y, gate, ids, f1, make_fn, taus):
    """Is a specific set of components what the F1 selection is missing?"""
    rows, width = [], len(f1)
    s = torch.zeros(width, dtype=torch.bool)
    s[ids] = True
    tests = [("all but set", ~s), ("only set", s)]
    tests += [(f"f1>={tau} + set", (f1 >= tau) | s) for tau in taus]
    tests += [(f"all but {i}", ~torch.zeros(width, dtype=torch.bool).index_fill_(0, torch.tensor([i]), True)) for i in ids]
    for name, keep in tests:
        fn = make_fn(keep)
        row = {"gate": gate, "set": list(ids), "test": name, "n_kept": int(keep.sum()), "fve": fve(X, Y, fn)}
        row.update(splice(rnn, groups, refs, gate, fn))
        rows.append(row)
        print(f"  drop test {name:>16}  kept {row['n_kept']:4d}  fve {row['fve']:.3f}  top1 {row['top1_agreement']:.3f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--traces", nargs="+", required=True, help="trace files, in the unit stores' order")
    ap.add_argument("--transcoders", nargs="*", default=[], metavar="GATE=CKPT", help="reset / update / candidate")
    ap.add_argument("--feature_results", help="autointerp results for the transcoder features")
    ap.add_argument("--feature_store", help="feature store, to check transcoder fingerprints")
    ap.add_argument("--units", nargs="*", default=[], metavar="NAME=RESULTS", help="autointerp results for GRU units")
    ap.add_argument("--unit_store", help="a unit store, to check the trace order matches")
    ap.add_argument("--gates", nargs="*", default=list(GATES), choices=list(GATES))
    ap.add_argument("--max_sequences", type=int, default=20000)
    ap.add_argument("--taus", nargs="*", type=float, default=[0.8, 0.5, 0.2])
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--core_frac", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop_sets", nargs="*", default=[], metavar="GATE=ID,ID,...",
                    help="feature sets to test as what the F1 selection misses, e.g. candidate=162,214")
    ap.add_argument("--compare_set", metavar="GATE=ID,ID,...",
                    help="instead of the main run: feature F1 selections with and without this set, against the "
                         "--units selections, with paired-bootstrap significance (compare_selections), written to --out")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    transcoders = dict(s.split("=", 1) for s in args.transcoders)
    units = dict(s.split("=", 1) for s in args.units)
    if args.feature_store:
        check_fingerprints(args.feature_store, {"rnn": args.rnn, **transcoders})
    if args.unit_store:
        order = pickle.load(open(os.path.join(args.unit_store, "metadata.pkl"), "rb"))["traces"]
        if [os.path.basename(p) for p in order] != [os.path.basename(p) for p in args.traces]:
            raise SystemExit(f"--traces must be in the unit store's order: {order}")

    rnn = build_task_rnn("copy", input_size=31, hidden_size=128)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    rnn.eval()
    groups = load_eval(args.traces, args.max_sequences, args.seed)
    refs = [torch.cat([run(rnn, g["inputs"][i:i + 4096]) for i in range(0, len(g["inputs"]), 4096)]) for g in groups]
    for g, ref in zip(groups, refs):       # the traces must come from this RNN
        true = g["inputs"][:, :ref.shape[1], :-1].argmax(-1)
        print(f"  length {ref.shape[1]}: original copy accuracy {(ref.argmax(-1) == true).float().mean():.3f}")

    if args.compare_set:
        gate, ids = args.compare_set.split("=", 1)
        X, Y = teacher_forced(groups, gate)
        tc = load_transcoder(transcoders[gate])
        means, _, _ = components(X, Y, tc)
        f1, _ = load_f1(args.feature_results, [GATES[gate][1]], tc.n_feats)
        unit_means, _, _ = components(X, Y)
        unit_sources = {name: (load_f1(results, GATES[gate][2], Y.shape[1])[0],
                               lambda keep: Ablated(keep, unit_means)) for name, results in units.items()}
        res = compare_selections(rnn, groups, refs, gate, [int(i) for i in ids.split(",")], f1,
                                 lambda keep: Ablated(keep, means, tc), unit_sources, args.taus)
        json.dump({"args": vars(args), **res}, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
        return

    drop_sets = {g: [int(i) for i in ids.split(",")] for g, ids in (s.split("=", 1) for s in args.drop_sets)}
    rows, summaries, drops = [], [], []
    for gate in args.gates:
        X, Y = teacher_forced(groups, gate)
        print(f"{gate}: {len(X)} steps", flush=True)
        if gate in transcoders and args.feature_results:
            tc = load_transcoder(transcoders[gate])
            means, alive, importance = components(X, Y, tc)
            f1, _ = load_f1(args.feature_results, [GATES[gate][1]], tc.n_feats)
            make_fn = lambda keep, tc=tc, means=means: Ablated(keep, means, tc)
            r, s = evaluate_source(rnn, groups, refs, X, Y, gate, "features", f1, means, alive, importance,
                                   make_fn, args.taus, args.n_random, args.core_frac, args.seed)
            rows += r; summaries.append(s)
            if gate in drop_sets:
                drops += drop_tests(rnn, groups, refs, X, Y, gate, drop_sets[gate], f1, make_fn, args.taus)
        for name, results in units.items():
            means, alive, importance = components(X, Y)
            f1, _ = load_f1(results, GATES[gate][2], Y.shape[1])
            make_fn = lambda keep, means=means: Ablated(keep, means)
            r, s = evaluate_source(rnn, groups, refs, X, Y, gate, f"units {name}", f1, means, alive, importance,
                                   make_fn, args.taus, args.n_random, args.core_frac, args.seed)
            rows += r; summaries.append(s)
        json.dump({"args": vars(args), "rows": rows, "core": summaries, "drops": drops}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
