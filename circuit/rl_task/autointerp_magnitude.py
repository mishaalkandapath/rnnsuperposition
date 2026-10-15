"""Magnitude interpretation for the RL agent: the agent-facing tool.

RL version of circuit/copy_task/autointerp_magnitude.py. An explanation is a
short list of numeric terms (numpy expressions over the per-step variables and
helpers of circuit/rl_task/autointerp.py, plus table()). Each term's weight is
fitted on the dev split (even episodes) on top of a timing baseline (the mean
activation at each (phase, episode-trial decile)), and the explanation is scored
on the test split (odd episodes):

    r2_beyond_timing = 1 - SSE / SSE(timing only)

Targets are transcoder features (gates reset / update / hidden, from the store)
or GRU units' raw values (gates unit_reset / unit_update / unit_candidate: r_t,
z_t, n_t at every step, from circuit/rl_task/neuron_baseline.py --raw).

The profile shows the timing profile, residual tables (activation minus timing)
by the previous trial's pattern and other cues, and reference scores: additive
one-hot tables per cue group (phase-specific), and all groups together (the
ceiling for additive explanations).

An explanation is recorded as a short label plus a SHORT FORM (--short; at most
3 simple terms, no hand-set constants or lookup tables; see check_short), which
is what gets scored. The full fit (up to 8 terms, any shape) is how hypotheses
are found and is kept as supporting evidence only: a fitted formula is not an
interpretation on its own.

Commands:
    python -m circuit.rl_task.autointerp_magnitude variables
    python -m circuit.rl_task.autointerp_magnitude profile STORE GATE ID
    python -m circuit.rl_task.autointerp_magnitude score   STORE GATE ID "TERM ;; TERM ;; ..." [--short] [--split test]
    python -m circuit.rl_task.autointerp_magnitude misses  STORE GATE ID "TERMS" [--short]
    python -m circuit.rl_task.autointerp_magnitude record  STORE GATE ID RESULT_JSON --out DIR
    python -m circuit.rl_task.autointerp_magnitude report  DIR
    python -m circuit.rl_task.autointerp_magnitude apply_skeptic STORE DIR
All commands take --observed_only (reject terms that read the model's own value / policy).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

from circuit.rl_task.autointerp import HELPERS_DOC, PHASES, VARIABLE_DOCS, Interp

MAX_TERMS = 8
N_WINDOWS = 30000
UNIT_VALUES = "rl_features/rl_unit_values"
UNIT_GATES = {"unit_reset": "reset", "unit_update": "update", "unit_candidate": "candidate"}
FIRING_RESULTS = {"reset": "rl_features/rl_autointerp", "update": "rl_features/rl_autointerp",
                  "hidden": "rl_features/rl_autointerp"}
PATTERNS = ["?", "a1 S1 r", "a1 S1 u", "a1 S2 r", "a1 S2 u", "a2 S2 r", "a2 S2 u", "a2 S1 r", "a2 S1 u"]
EXTRA_HELPERS = ("belief_lo = log(belief / (1 - belief)), the ideal observer's log-odds that S1 is good "
                 "(graded; positive = S1, negative = S2). "
                 "table(col, {value: weight, ...}) -> per-row weight looked up by the column's value (0 if not listed), "
                 "e.g. table(prev_action, {1: 1.0, 2: -1.0}); last_is('a2 S1 r') -> 1 where the previous trial was that "
                 "pattern (choice, state reached, r rewarded / u not); ep_dec = episode trial // 10. Booleans become 0/1. "
                 "Each term gets its own fitted weight (plus the timing baseline and an intercept), so write shapes, "
                 "not scales; multiply by (phase == D1) etc. to make a term phase-specific.")


def _pattern(action, state, reward):
    out = np.zeros(len(action), dtype=np.int16)
    for k, name in enumerate(PATTERNS[1:], 1):
        a, s, r = name.split()
        out[(action == int(a[1])) & (state == int(s[1])) & (reward == (1 if r == "r" else 0))] = k
    return out


MODEL_OUTPUT = re.compile(r"\b(value|choice|model_act)\b")
SHORT_MAX_TERMS = 3
FLOAT_CONSTANT = re.compile(r"(?<![\w.])\d*\.\d+")


def check_short(exprs):
    """The recorded explanation: a few simple named conditions whose weights the fit sets.

    Fitted terms (up to MAX_TERMS, any shape) are a tool for finding hypotheses; the explanation that
    is scored must be what a short verbal label says: at most SHORT_MAX_TERMS terms, no hand-set
    non-integer constants, no lookup tables, at most one previous-trial pattern and three factors per term.
    """
    if len(exprs) > SHORT_MAX_TERMS:
        raise ValueError(f"short form: at most {SHORT_MAX_TERMS} terms (got {len(exprs)})")
    for e in exprs:
        if FLOAT_CONSTANT.search(e):
            raise ValueError(f"short form: no hand-set non-integer constants (weights are fitted): {e}")
        if "table(" in e:
            raise ValueError(f"short form: no lookup tables: {e}")
        if e.count("last_is") > 1:
            raise ValueError(f"short form: at most one previous-trial pattern per term: {e}")
        if e.count("*") + e.count("&") > 2:
            raise ValueError(f"short form: at most three factors per term: {e}")


class Mag:
    def __init__(self, store, gate, feature, seed=0, observed_only=False):
        self.ip, self.gate, self.feature = Interp(store), gate, feature
        self.observed_only = observed_only      # reject terms that read the model's own outputs
        c = self.ip.cols
        rng = np.random.default_rng(seed)
        n_win = len(c["row_start"]) - 1
        test_win = (np.asarray(c["episode"])[np.asarray(c["row_start"][:-1])] % 2).astype(bool)
        self.rows = {}
        for split, mask in (("dev", ~test_win), ("test", test_win)):
            pool = np.nonzero(mask)[0]
            wins = np.sort(rng.choice(pool, size=min(N_WINDOWS, len(pool)), replace=False))
            self.rows[split] = np.concatenate([np.arange(c["row_start"][w], c["row_start"][w + 1]) for w in wins])
        if gate in UNIT_GATES:
            self.act = np.asarray(np.load(os.path.join(UNIT_VALUES, f"{UNIT_GATES[gate]}.npy"),
                                          mmap_mode="r")[:, feature], dtype=np.float64)
        else:
            self.act = self.ip.activations(gate, feature).astype(np.float64)
        self.ep_dec = np.clip(np.asarray(c["ep_trial"]) // 10, 0, 9)
        self.last = _pattern(np.asarray(c["prev_action"]), np.asarray(c["prev_state"]), np.asarray(c["prev_reward"]))
        self.tcode = np.asarray(c["phase"]).astype(np.int64) * 10 + self.ep_dec
        dev = self.rows["dev"]
        sums = np.bincount(self.tcode[dev], weights=self.act[dev], minlength=30)
        cnt = np.bincount(self.tcode[dev], minlength=30)
        self.timing = (sums / np.maximum(cnt, 1))[self.tcode]

    def namespace(self):
        ns = self.ip.namespace()
        ns.update(ep_dec=self.ep_dec, last=self.last)

        def table(col, mapping):
            col = np.asarray(col)
            out = np.zeros(len(col))
            for k, v in mapping.items():
                out[col == k] = v
            return out

        ns.update(table=table, last_is=lambda name: (self.last == PATTERNS.index(name)).astype(np.float64))
        if "belief" in ns:
            b = np.clip(np.asarray(ns["belief"], dtype=np.float64), 1e-4, 1 - 1e-4)
            ns["belief_lo"] = np.log(b / (1 - b))
        return ns

    def terms(self, spec, short=False):
        exprs = [e.strip() for e in spec.split(";;") if e.strip()]
        if not exprs:
            raise ValueError("no terms")
        if short:
            check_short(exprs)
        if len(exprs) > MAX_TERMS:
            raise ValueError(f"at most {MAX_TERMS} terms (got {len(exprs)}): explanations must stay readable")
        if self.observed_only and any(MODEL_OUTPUT.search(e) for e in exprs):
            raise ValueError("observed-only mode: terms may not use the model's outputs (value, choice, model_act); "
                             "use belief / belief_lo for the agent's inferred state")
        ns = self.namespace()
        cols = [np.broadcast_to(np.asarray(eval(e, {"__builtins__": {}}, ns), dtype=np.float64), (self.ip.n_rows,))
                for e in exprs]
        return exprs, np.stack(cols, 1)

    def fit(self, spec, short=False):
        exprs, X = self.terms(spec, short)
        dev = self.rows["dev"]
        Xd = np.column_stack([X[dev], np.ones(len(dev))])
        w = np.linalg.lstsq(Xd, self.act[dev] - self.timing[dev], rcond=None)[0]
        return exprs, X, w

    def predict(self, X, w):
        return self.timing + X @ w[:-1] + w[-1]

    def evaluate(self, spec, split="dev", short=False):
        exprs, X, w = self.fit(spec, short)
        rows = self.rows[split]
        y, p, base = self.act[rows], self.predict(X, w)[rows], self.timing[rows]
        sse, sse_t, sst = ((y - p) ** 2).sum(), ((y - base) ** 2).sum(), ((y - y.mean()) ** 2).sum()
        return {"gate": self.gate, "feature": self.feature, "terms": exprs, "weights": [float(v) for v in w[:-1]],
                "split": split, "r2_beyond_timing": float(1 - sse / sse_t), "r2": float(1 - sse / sst),
                "r2_timing_only": float(1 - sse_t / sst)}

    # reference models: phase-specific additive one-hot tables per cue group
    def cue_groups(self):
        c = self.ip.cols
        tv = self.ip.tv
        before = _pattern(tv("action", -2), tv("state", -2), tv("reward", -2))
        q = lambda x: np.searchsorted(np.quantile(np.asarray(x)[self.rows["dev"]], np.linspace(0.1, 0.9, 9)), x)
        groups = {}
        if "belief" in c:
            groups["belief decile"] = q(c["belief"])
        return {**groups, "last trial": self.last, "trial before": before,
                "this choice x state": np.asarray(c["action"]) * 3 + np.asarray(c["state"]),
                "this reward (future)": np.asarray(c["reward"]) + 2, "next choice (future)": np.asarray(c["next_action"]) + 1,
                "good state (hidden)": np.asarray(c["good"]) + 1,
                "trials since reversal (hidden)": np.clip(np.asarray(c["since_rev"]), -1, 20) + 1,
                "policy decile (model)": q(c["choice"]), "value decile (model)": q(c["value"])}

    def references(self):
        from scipy import sparse
        from sklearn.linear_model import Ridge
        groups = self.cue_groups()
        phase = np.asarray(self.ip.cols["phase"]).astype(np.int64)
        dev, test = self.rows["dev"], self.rows["test"]
        r_dev, r_test = self.act[dev] - self.timing[dev], self.act[test] - self.timing[test]
        sse_t = (r_test ** 2).sum()

        def design(names):
            """One-hot (phase x cue value) blocks; values unseen on dev get no column."""
            cols = []
            for n in names:
                code = phase * 64 + groups[n].astype(np.int64)
                uniq = np.unique(code[dev])
                pos = np.clip(np.searchsorted(uniq, code), 0, len(uniq) - 1)
                cols.append((np.where(uniq[pos] == code, pos, -1), len(uniq)))

            def build(rows):
                blocks = []
                for col, k in cols:
                    v = col[rows]
                    keep = v >= 0
                    blocks.append(sparse.csr_matrix((np.ones(keep.sum()), (np.nonzero(keep)[0], v[keep])),
                                                    shape=(len(rows), k)))
                return sparse.hstack(blocks).tocsr()
            return build(dev), build(test)

        out = {}
        for name in list(groups) + ["all observed", "all"]:
            names = ([n for n in groups if "future" not in n and "hidden" not in n and "model" not in n]
                     if name == "all observed" else list(groups) if name == "all" else [name])
            Xd, Xt = design(names)
            m = Ridge(alpha=1.0).fit(Xd, r_dev)
            out[name] = float(1 - ((r_test - m.predict(Xt)) ** 2).sum() / sse_t) if sse_t > 0 else 0.0
        return out


def _render(ip, s, values):
    """One window: [rel] r<prev reward> a<choice> S<state><c/R>, each cell with a value."""
    c = ip.cols
    lo, hi = int(c["row_start"][s]), int(c["row_start"][s + 1])
    rew = lambda v: {1: "1", 0: "0", -1: "-"}.get(int(v), "?")
    parts = []
    for r0 in range(lo, hi, 3):
        cells = [f"r{rew(c['prev_reward'][r0])}", f"a{int(c['action'][r0])}",
                 f"S{int(c['state'][r0])}{'c' if c['common'][r0] == 1 else 'R' if c['common'][r0] == 0 else '!'}"]
        parts.append(f"[{int(c['rel'][r0]):+d} t{int(c['ep_trial'][r0])}] " +
                     " ".join(f"{cell}[{values[r0 + k - lo]}]" for k, cell in enumerate(cells)))
    return " | ".join(parts)


def profile(m: Mag, n_examples=6):
    ip, act, dev = m.ip, m.act, m.rows["dev"]
    a = act[dev]
    c = {k: np.asarray(v)[dev] for k, v in ip.cols.items() if k != "row_start"}
    kind = "GRU unit raw value" if m.gate in UNIT_GATES else "transcoder feature activation"
    lines = [f"# {m.gate}/{m.feature}: magnitude profile ({kind}; dev split, {len(np.unique(c['seq']))} windows)",
             f"nonzero on {(a != 0).mean():.1%} of steps; mean {a.mean():.3f}, std {a.std():.3f}, min {a.min():.3f}, "
             f"max {a.max():.3f}"]
    fr = FIRING_RESULTS.get(m.gate)
    if fr and os.path.exists(os.path.join(fr, f"{m.gate}_{m.feature}.json")):
        r = json.load(open(os.path.join(fr, f"{m.gate}_{m.feature}.json")))
        lines.append(f"firing-based label (earlier pass, test F1 {(r.get('test_scores') or {}).get('f1')}): "
                     f"[{r.get('final_verdict')}] {r.get('final_label') or r.get('label')}")
        if r.get("scorecard"):
            lines.append(f"firing scorecard ({r['scorecard']['format']}, test F1 {r['scorecard']['test_f1']:.2f}): "
                         f"{r['scorecard']['summary']}")
    lines.append("\n## timing baseline: mean by phase x episode decile (trials 0-9, 10-19, ...)")
    for ph in range(3):
        sel = c["phase"] == ph
        means = [a[sel & (m.ep_dec[dev] == d)].mean() for d in range(10)]
        lines.append(f"  {PHASES[ph]}: " + " ".join(f"{v:+.4g}" for v in means))
    lines.append(f"timing alone explains R^2 {m.evaluate('0 * phase', 'test')['r2_timing_only']:.2f} of the variance (test)")
    resid = act - m.timing
    rd = resid[dev]
    lines.append("\n## residual (activation - timing) by phase x previous trial (choice, state, r/u); n >= 200")
    last = m.last[dev]
    for ph in range(3):
        cells = []
        for k, name in enumerate(PATTERNS):
            sel = (c["phase"] == ph) & (last == k)
            if sel.sum() >= 200:
                cells.append(f"{name} {rd[sel].mean():+.3g}")
        lines.append(f"  {PHASES[ph]}: " + ", ".join(cells))
    groups = m.cue_groups()
    shown = ("belief decile", "this choice x state", "good state (hidden)") if m.observed_only else \
        ("this choice x state", "policy decile (model)", "value decile (model)", "good state (hidden)")
    for name in (n for n in shown if n in groups):
        g = groups[name][dev]
        lines.append(f"\n## residual by phase x {name}")
        for ph in range(3):
            vals = [(int(v), rd[(c["phase"] == ph) & (g == v)].mean()) for v in np.unique(g)
                    if ((c["phase"] == ph) & (g == v)).sum() >= 200]
            lines.append(f"  {PHASES[ph]}: " + ", ".join(f"{v}: {x:+.3g}" for v, x in vals))
    lines.append("  (this choice x state code = 3 * action + state; belief decile 9 = most sure S1 is good; policy decile "
                 "9 = most a1; value decile 9 = highest; good state code = good + 1)")
    lines.append("\n## reference scores (test R^2 beyond timing), phase-specific additive tables")
    refs = m.references()
    lines.append("  " + ", ".join(f"{k}: {v:.2f}" for k, v in refs.items()))
    lines.append("  'all' is the ceiling for additive explanations over these cues; 'all observed' excludes future, "
                 "hidden and model-output cues.")
    lines.append("\n## examples: cell[activation - timing] (windows with the largest residual spread)")
    seq_rows = ip.cols["row_start"]
    dev_seqs = np.unique(ip.cols["seq"][dev])[:4000]
    spread = np.array([resid[seq_rows[s]:seq_rows[s + 1]].std() for s in dev_seqs])
    for s in dev_seqs[np.argsort(-spread)[:n_examples]]:
        lo, hi = seq_rows[s], seq_rows[s + 1]
        lines.append("  " + _render(ip, s, [f"{v:+.2g}" for v in resid[lo:hi]]))
    return "\n".join(lines)


def misses(m: Mag, spec, n=8, short=False):
    exprs, X, w = m.fit(spec, short)
    pred = m.predict(X, w)
    ip, dev = m.ip, m.rows["dev"]
    err = m.act - pred
    seq_rows = ip.cols["row_start"]
    dev_seqs = np.unique(ip.cols["seq"][dev])[:6000]
    sse = np.array([(err[seq_rows[s]:seq_rows[s + 1]] ** 2).sum() for s in dev_seqs])
    lines = ["weights: " + ", ".join(f"{e} -> {v:+.3f}" for e, v in zip(exprs, w[:-1])),
             "worst-fit dev windows: cell[actual/predicted]"]
    for s in dev_seqs[np.argsort(-sse)[:n]]:
        lo, hi = seq_rows[s], seq_rows[s + 1]
        lines.append("  " + _render(ip, s, [f"{a:.4g}/{p:.4g}" for a, p in zip(m.act[lo:hi], pred[lo:hi])]))
    e = err[dev]
    ph, last = np.asarray(ip.cols["phase"])[dev], m.last[dev]
    lines.append("mean error (actual - predicted) by phase: " + ", ".join(f"{PHASES[p]} {e[ph == p].mean():+.3f}" for p in range(3)))
    cells = sorted(((e[(ph == p) & (last == k)].mean(), f"{PHASES[p]} after {PATTERNS[k]}") for p in range(3)
                    for k in range(9) if ((ph == p) & (last == k)).sum() >= 200))
    lines.append("most under-predicted (phase, previous trial): " + ", ".join(f"{t} {v:+.3f}" for v, t in cells[::-1][:5]))
    lines.append("most over-predicted (phase, previous trial): " + ", ".join(f"{t} {v:+.3f}" for v, t in cells[:5]))
    return "\n".join(lines)


def report(out_dir):
    rows = [json.load(open(os.path.join(out_dir, n))) for n in sorted(os.listdir(out_dir)) if n.endswith(".json")]
    lines = [f"{len(rows)} magnitude explanations"]
    for gate in sorted({r["gate"] for r in rows}):
        g = [r for r in rows if r["gate"] == gate]
        verdicts = {}
        for r in g:
            v = r.get("final_verdict") or r.get("verdict")
            verdicts[v] = verdicts.get(v, 0) + 1
        vals = [r["test_scores"]["r2_beyond_timing"] for r in g if r.get("test_scores")]
        lines.append(f"{gate}: {len(g)}, verdicts {verdicts}, test R^2 beyond timing median {np.median(vals):.2f}"
                     if vals else f"{gate}: {len(g)}, verdicts {verdicts}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("profile", "score", "misses", "record"):
        s = sub.add_parser(name)
        s.add_argument("store"); s.add_argument("gate"); s.add_argument("feature", type=int)
        s.add_argument("--observed_only", action="store_true", help="reject terms that read the model's outputs")
        if name in ("score", "misses"):
            s.add_argument("terms")
            s.add_argument("--short", action="store_true", help="check the short-form rules (the recorded explanation)")
        if name == "score":
            s.add_argument("--split", default="dev", choices=["dev", "test"])
        if name == "record":
            s.add_argument("result_json"); s.add_argument("--out", required=True)
    r = sub.add_parser("report"); r.add_argument("dir")
    ak = sub.add_parser("apply_skeptic", help="adopt the skeptic's better short form in a results directory (re-scored)")
    ak.add_argument("store"); ak.add_argument("dir")
    sub.add_parser("variables")
    args = ap.parse_args()

    if args.cmd == "variables":
        print("\n".join(f"{k:13s} {v}" for k, v in VARIABLE_DOCS.items()))
        print(HELPERS_DOC)
        print(EXTRA_HELPERS)
        return
    if args.cmd == "report":
        print(report(args.dir))
        return
    if args.cmd == "apply_skeptic":
        print(f"{args.dir}: {apply_skeptic(args.store, args.dir)}")
        return
    m = Mag(args.store, args.gate, args.feature, observed_only=args.observed_only)
    if args.cmd == "profile":
        print(profile(m))
    elif args.cmd == "score":
        print(json.dumps(m.evaluate(args.terms, args.split, args.short), indent=1))
    elif args.cmd == "misses":
        print(misses(m, args.terms, short=args.short))
    elif args.cmd == "record":
        rec = json.loads(args.result_json)
        short = "short_terms" in rec
        for key in ("label", "terms", "verdict", "explanation"):
            if key not in rec:
                sys.exit(f"result JSON needs '{key}'")
        if short:
            # the explanation is the short form; the full fit is supporting evidence
            test, dev = m.evaluate(rec["short_terms"], "test", True), m.evaluate(rec["short_terms"], "dev", True)
            rec["full_test_scores"] = m.evaluate(rec["terms"], "test")
        else:
            test, dev = m.evaluate(rec["terms"], "test"), m.evaluate(rec["terms"], "dev")
        rec.update({"gate": args.gate, "feature": args.feature, "test_scores": test, "dev_scores": dev,
                    "final_verdict": rec["verdict"], "final_label": rec["label"], "observed_only": args.observed_only})
        os.makedirs(args.out, exist_ok=True)
        json.dump(rec, open(os.path.join(args.out, f"{args.gate}_{args.feature}.json"), "w"), indent=1)
        out = {"test_r2_beyond_timing": test["r2_beyond_timing"], "test_r2": test["r2"], "weights": test["weights"]}
        if short:
            out["full_fit_test_r2_beyond_timing"] = rec["full_test_scores"]["r2_beyond_timing"]
        print(json.dumps(out, indent=1))


def apply_skeptic(store, out_dir):
    """Make the skeptic's better short form the recorded explanation (re-scored); keep the original."""
    counts = {"adopted": 0, "failed": 0, "none": 0}
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(out_dir, name)
        rec = json.load(open(path))
        better = (rec.get("skeptic") or {}).get("better_short_terms") or (rec.get("skeptic") or {}).get("better_terms")
        if not better or rec.get("terms_from") == "skeptic" or "short_terms" not in rec:
            counts["none"] += 1
            continue
        try:
            m = Mag(store, rec["gate"], rec["feature"], observed_only=rec.get("observed_only", False))
            test, dev = m.evaluate(better, "test", True), m.evaluate(better, "dev", True)
        except Exception as e:                  # breaks the short-form rules, or does not parse: keep the original
            rec["skeptic_terms_error"] = str(e)[:200]
            counts["failed"] += 1
        else:
            rec.update(original_short_terms=rec["short_terms"], original_test_scores=rec["test_scores"],
                       short_terms=better, test_scores=test, dev_scores=dev, terms_from="skeptic")
            counts["adopted"] += 1
        json.dump(rec, open(path, "w"), indent=1)
    return counts


if __name__ == "__main__":
    main()
