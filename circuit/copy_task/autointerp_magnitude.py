"""Magnitude interpretation of dense features: the agent-facing tool.

A firing predicate cannot explain a feature that is on almost everywhere; what
it encodes is in how strongly it fires. Here an explanation is a short list of
numeric terms (numpy expressions over the same per-step variables and helpers
as circuit/copy_task/autointerp.py, plus table() and n_in()). Each term's weight is fitted
on the dev split (even sequence ids), on top of a timing baseline (the mean
activation at each (L, t)), and the explanation is scored on the test split
(odd ids):

    r2_beyond_timing = 1 - SSE / SSE(timing only)

i.e. the share of what timing leaves unexplained that the terms capture. The
profile reports two references from circuit/copy_task/magnitude_scores.py: the full
additive token tables (a ceiling for additive explanations) and an automatic
token-set fit.

Commands:
    python -m circuit.copy_task.autointerp_magnitude profile STORE GATE FEATURE
    python -m circuit.copy_task.autointerp_magnitude score   STORE GATE FEATURE "TERM ;; TERM ;; ..." [--split test]
    python -m circuit.copy_task.autointerp_magnitude misses  STORE GATE FEATURE "TERMS"
    python -m circuit.copy_task.autointerp_magnitude record  STORE GATE FEATURE RESULT_JSON --out DIR
    python -m circuit.copy_task.autointerp_magnitude report  DIR

Terms are separated by ";;". Example (feature 487):
    "isin(rel_tok(0), S) ;; isin(rel_tok(-1), S) ;; isin(rel_tok(-2), S)"
with S written out, e.g. [tok('e'), tok('b'), tok('g')].
"""
import argparse
import json
import os
import sys

import numpy as np

from circuit.copy_task.autointerp import (DIGITS, HELPERS_DOC, LETTERS, PHASES, TOKEN_IDX, TOKENS, VARIABLE_DOCS,
                                VOWELS, Interp)

MAX_TERMS = 8
N_SEQUENCES = 40000
SCORES = "features/copy_circuits/magnitude_scores_{gate}.json"
EXTRA_HELPERS = ("table(col, {'a': 1.0, 'b': -0.5, ...}) -> per-row value looked up by token (0 for tokens not listed); "
                 "n_in(S, ds) -> how many of the inputs at relative steps ds (e.g. [0, -1, -2]) are in S; "
                 "w_in(S, {0: 1.0, -1: 0.8, -2: 0.5}) -> weighted count over relative steps. "
                 "Booleans become 0/1. Each term gets its own fitted weight, so write the shape, not the scale.")


class Mag:
    def __init__(self, store, gate, feature, seed=0):
        self.ip, self.gate, self.feature = Interp(store), gate, feature
        rng = np.random.default_rng(seed)
        n_seq = len(self.ip.cols["row_start"]) - 1
        self.rows = {}
        for split, parity in (("dev", 0), ("test", 1)):
            seqs = np.sort(rng.choice(np.arange(parity, n_seq, 2), size=min(N_SEQUENCES, n_seq // 2), replace=False))
            self.rows[split] = np.concatenate([np.arange(self.ip.cols["row_start"][s], self.ip.cols["row_start"][s + 1])
                                               for s in seqs])
        self.act = self.ip.activations(gate, feature)
        c = self.ip.cols
        self.tcode = (c["L"].astype(np.int64) - 3) * 18 + c["t"]
        dev = self.rows["dev"]
        sums = np.bincount(self.tcode[dev], weights=self.act[dev], minlength=7 * 18)
        cnt = np.bincount(self.tcode[dev], minlength=7 * 18)
        self.timing = (sums / np.maximum(cnt, 1))[self.tcode]           # per-row timing baseline

    def namespace(self):
        ip = self.ip
        ns = {k: v for k, v in ip.cols.items() if k not in ("row_start",)}

        def table(col, mapping):
            lut = np.zeros(len(TOKENS) + 1)
            for k, v in mapping.items():
                lut[TOKEN_IDX[k] if isinstance(k, str) else int(k)] = v
            col = np.asarray(col).astype(np.int64)
            return np.where(col >= 0, lut[np.clip(col, 0, len(TOKENS))], 0.0)

        def n_in(S, ds):
            return sum(np.isin(ip.rel_tok(d), S).astype(np.float32) for d in ds)

        def w_in(S, weights):
            return sum(w * np.isin(ip.rel_tok(d), S).astype(np.float32) for d, w in weights.items())

        ns.update(np=np, tok=lambda s: TOKEN_IDX[s], isin=lambda col, vals: np.isin(col, vals),
                  LETTERS=LETTERS, DIGITS=DIGITS, VOWELS=VOWELS, orig=ip.orig, rel_tok=ip.rel_tok,
                  out_at=ip.out_at, seen=ip.seen, n_seen=ip.n_seen, in_orig=ip.in_orig,
                  table=table, n_in=n_in, w_in=w_in)
        return ns

    def terms(self, spec):
        exprs = [e.strip() for e in spec.split(";;") if e.strip()]
        if not exprs:
            raise ValueError("no terms")
        if len(exprs) > MAX_TERMS:
            raise ValueError(f"at most {MAX_TERMS} terms (got {len(exprs)}): explanations must stay readable")
        ns = self.namespace()
        cols = []
        for e in exprs:
            v = np.broadcast_to(np.asarray(eval(e, {"__builtins__": {}}, ns), dtype=np.float64), (self.ip.n_rows,))
            cols.append(v)
        return exprs, np.stack(cols, 1)

    def fit(self, spec):
        exprs, X = self.terms(spec)
        dev = self.rows["dev"]
        r = self.act[dev] - self.timing[dev]
        Xd = np.column_stack([X[dev], np.ones(len(dev))])
        w = np.linalg.lstsq(Xd, r, rcond=None)[0]
        return exprs, X, w

    def predict(self, X, w):
        return self.timing + X @ w[:-1] + w[-1]

    def evaluate(self, spec, split="dev"):
        exprs, X, w = self.fit(spec)
        rows = self.rows[split]
        y, p, base = self.act[rows], self.predict(X, w)[rows], self.timing[rows]
        sse, sse_t, sst = ((y - p) ** 2).sum(), ((y - base) ** 2).sum(), ((y - y.mean()) ** 2).sum()
        return {"gate": self.gate, "feature": self.feature, "terms": exprs, "weights": [float(v) for v in w[:-1]],
                "split": split, "r2_beyond_timing": float(1 - sse / sse_t), "r2": float(1 - sse / sst),
                "r2_timing_only": float(1 - sse_t / sst)}


def _render(seq, values):
    return " ".join(f"{tk}[{v}]" for tk, v in zip(seq, values))


def profile(m: Mag, n_examples=6):
    ip, act, dev = m.ip, m.act, m.rows["dev"]
    a = act[dev]
    c = {k: v[dev] for k, v in ip.cols.items() if k not in ("row_start",)}
    lines = [f"# {m.gate}/{m.feature}: magnitude profile (dev split, {len(np.unique(c['seq']))} sequences)",
             f"active on {(a > 0).mean():.1%} of steps; mean activation {a.mean():.3f}, mean when active "
             f"{a[a > 0].mean() if (a > 0).any() else 0:.3f}, max {a.max():.3f}"]
    fr = f"features/copy_gru_v1_candiff_autointerp/{m.gate}_{m.feature}.json"
    if os.path.exists(fr):
        r = json.load(open(fr))
        lines.append(f"firing-based label (from the earlier autointerp pass, F1 {(r.get('test_scores') or {}).get('f1')}): "
                     f"[{r.get('final_verdict')}] {r.get('label')}")
    lines.append("\n## timing: mean activation by phase, and by step t for L = 4 and L = 7")
    for ph in range(3):
        sel = c["phase"] == ph
        lines.append(f"  {PHASES[ph]}: {a[sel].mean():.3f}")
    for L in (4, 7):
        sel = c["L"] == L
        means = [a[sel & (c["t"] == t)].mean() for t in range(2 * L)]
        lines.append(f"  L={L}: " + " ".join(f"t{t}:{v:.2f}" for t, v in enumerate(means)))
    path = SCORES.format(gate=m.gate)
    if os.path.exists(path):
        rec = next((r for r in json.load(open(path))["features"] if r["feature"] == m.feature), None)
        if rec:
            lines.append("\n## reference scores (test R^2 beyond timing) from additive token tables")
            for k, v in rec["models"].items():
                lines.append(f"  {k}: {v['r2_beyond_timing'] if v['r2_beyond_timing'] is not None else 0:.2f}")
            lines.append("\n## fitted token tables (activation relative to the table average; strongest tables first)")
            for t in rec["tables"]:
                lines.append(f"  {t['cue']} @ {t['phase']} (spread {t['spread']:.2f}): raised by {t['raised_by']}  lowered by {t['lowered_by']}")
            ts = rec.get("token_sets")
            if ts:
                lines.append(f"\n## automatic token-set fit: S+ {ts['S+']}  S- {ts['S-']}; weights per (cue, phase): {ts['weights']}")
            lines.append("cues: input = rel_tok(0), prev = rel_tok(-1), prevK = rel_tok(-K), out = out_tok = out_at(0), "
                         "next_out = out_at(1), out+K = out_at(K), first = orig(0); each table is phase-specific.")
    # examples: residual (activation - timing) per step, for sequences with large positive / negative residuals
    resid = act - m.timing
    seq_rows = ip.cols["row_start"]
    lines.append("\n## examples: token[activation - timing] at each step (positive = above what timing predicts)")
    dev_seqs = np.unique(ip.cols["seq"][dev])
    score = np.array([resid[seq_rows[s]:seq_rows[s + 1]].std() for s in dev_seqs[:4000]])
    for s in dev_seqs[:4000][np.argsort(-score)[:n_examples]]:
        lo, hi = seq_rows[s], seq_rows[s + 1]
        lines.append("  " + _render(ip.sequences[s], [f"{v:+.1f}" for v in resid[lo:hi]]))
    return "\n".join(lines)


def misses(m: Mag, spec, n=8):
    exprs, X, w = m.fit(spec)
    pred = m.predict(X, w)
    ip, dev = m.ip, m.rows["dev"]
    err = m.act - pred
    seq_rows = ip.cols["row_start"]
    dev_seqs = np.unique(ip.cols["seq"][dev])[:6000]
    sse = np.array([(err[seq_rows[s]:seq_rows[s + 1]] ** 2).sum() for s in dev_seqs])
    lines = [f"weights: " + ", ".join(f"{e} -> {v:+.2f}" for e, v in zip(exprs, w[:-1])),
             "worst-fit dev sequences: token[actual/predicted]"]
    for s in dev_seqs[np.argsort(-sse)[:n]]:
        lo, hi = seq_rows[s], seq_rows[s + 1]
        lines.append("  " + _render(ip.sequences[s], [f"{a:.1f}/{p:.1f}" for a, p in zip(m.act[lo:hi], pred[lo:hi])]))
    # systematic patterns: mean error by phase and by current input token
    c = {k: v[dev] for k, v in ip.cols.items() if k in ("phase", "in_tok", "out_tok")}
    e = err[dev]
    lines.append("mean error (actual - predicted) by phase: " + ", ".join(f"{PHASES[p]} {e[c['phase'] == p].mean():+.2f}" for p in range(3)))
    by_tok = sorted(((e[c["in_tok"] == k].mean(), TOKENS[k]) for k in range(31) if (c["in_tok"] == k).sum() > 100))
    lines.append("most under-predicted current inputs: " + ", ".join(f"{t} {v:+.2f}" for v, t in by_tok[::-1][:5]))
    lines.append("most over-predicted current inputs: " + ", ".join(f"{t} {v:+.2f}" for v, t in by_tok[:5]))
    return "\n".join(lines)


def report(out_dir):
    rows = []
    for name in sorted(os.listdir(out_dir)):
        if name.endswith(".json"):
            r = json.load(open(os.path.join(out_dir, name)))
            rows.append(r)
    lines = [f"{len(rows)} magnitude explanations"]
    verdicts = {}
    for r in rows:
        verdicts[r.get("final_verdict") or r.get("verdict")] = verdicts.get(r.get("final_verdict") or r.get("verdict"), 0) + 1
    lines.append(f"verdicts: {verdicts}")
    vals = [r["test_scores"]["r2_beyond_timing"] for r in rows if r.get("test_scores")]
    if vals:
        lines.append(f"test R^2 beyond timing: median {np.median(vals):.2f}, >=0.6: {sum(v >= 0.6 for v in vals)}, "
                     f"0.3-0.6: {sum(0.3 <= v < 0.6 for v in vals)}, <0.3: {sum(v < 0.3 for v in vals)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("profile", "score", "misses", "record"):
        s = sub.add_parser(name)
        s.add_argument("store"); s.add_argument("gate"); s.add_argument("feature", type=int)
        if name in ("score", "misses"):
            s.add_argument("terms")
        if name == "score":
            s.add_argument("--split", default="dev", choices=["dev", "test"])
        if name == "record":
            s.add_argument("result_json"); s.add_argument("--out", required=True)
    r = sub.add_parser("report"); r.add_argument("dir")
    sub.add_parser("variables")
    args = ap.parse_args()

    if args.cmd == "variables":
        print("\n".join(f"{k}: {v}" for k, v in VARIABLE_DOCS.items()))
        print(HELPERS_DOC)
        print(EXTRA_HELPERS)
        return
    if args.cmd == "report":
        print(report(args.dir))
        return
    m = Mag(args.store, args.gate, args.feature)
    if args.cmd == "profile":
        print(profile(m))
    elif args.cmd == "score":
        print(json.dumps(m.evaluate(args.terms, args.split), indent=1))
    elif args.cmd == "misses":
        print(misses(m, args.terms))
    elif args.cmd == "record":
        rec = json.loads(args.result_json)
        for key in ("label", "terms", "verdict", "explanation"):
            if key not in rec:
                sys.exit(f"result JSON needs '{key}'")
        test = m.evaluate(rec["terms"], "test")
        dev = m.evaluate(rec["terms"], "dev")
        rec.update({"gate": args.gate, "feature": args.feature, "test_scores": test, "dev_scores": dev,
                    "final_verdict": rec["verdict"], "final_label": rec["label"]})
        os.makedirs(args.out, exist_ok=True)
        json.dump(rec, open(os.path.join(args.out, f"{args.gate}_{args.feature}.json"), "w"), indent=1)
        print(json.dumps({"test_r2_beyond_timing": test["r2_beyond_timing"], "test_r2": test["r2"],
                          "weights": test["weights"]}, indent=1))


if __name__ == "__main__":
    main()
