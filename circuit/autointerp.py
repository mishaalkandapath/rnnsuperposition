"""Activation-based feature interpretation for the copy-task transcoders.

Works on a feature store (circuit/feature_store.py). Every (sequence, step) of the
analysed sequences gets a row of derived variables; a feature's activations are a
dense vector over those rows. Explanations are predicates over the variables,
written as numpy expressions, and are scored exactly against all sequences.

Sequences are split by sequence id: even ids are the dev split (profiles and
iteration), odd ids the test split (reported scores), so explanations are judged
on sequences they were not fitted to.

Commands:
    python -m circuit.autointerp profile STORE GATE FEATURE
    python -m circuit.autointerp score   STORE GATE FEATURE "PREDICATE" [--split dev|test]
    python -m circuit.autointerp misses  STORE GATE FEATURE "PREDICATE" [--split dev|test]
    python -m circuit.autointerp record  STORE GATE FEATURE RESULT_JSON --out DIR
    python -m circuit.autointerp report  DIR
    python -m circuit.autointerp variables

Copy-task sequence layout: L original tokens, <DEL> at step L, then the model's
own outputs fed back, so a sequence has 2L steps. At step t >= L the model
outputs the original token at index t - L.
"""
import argparse
import json
import os
import pickle
import sys
from typing import Dict, Optional

import numpy as np

from circuit.feature_store import open_feature_store

TOKENS = [chr(c) for c in range(ord("a"), ord("z") + 1)] + ["0", "1", "2", "3", "<DEL>"]
TOKEN_IDX = {t: i for i, t in enumerate(TOKENS)}
DEL = TOKEN_IDX["<DEL>"]
LETTERS = list(range(26))
DIGITS = list(range(26, 30))
VOWELS = [TOKEN_IDX[c] for c in "aeiou"]
PHASES = {0: "original", 1: "delimiter", 2: "copy"}

VARIABLE_DOCS = {
    "t": "step index in the sequence (0 .. 2L-1)",
    "L": "number of original tokens (3..9); the sequence has 2L steps",
    "phase": "0 = original input (t < L), 1 = delimiter step (t == L), 2 = copy (t > L)",
    "in_tok": "token index input at this step (0-25 letters a-z, 26-29 digits 0-3, 30 <DEL>)",
    "prev_tok": "input token at the previous step (-1 at t = 0)",
    "out_tok": "token being output at this step (= original token recall_idx); -1 if t < L",
    "orig_idx": "t during the original phase, else -1",
    "recall_idx": "t - L for t >= L (which original token is being output), else -1",
    "to_go": "original phase: tokens still to come before <DEL> (L-1-t); delimiter/copy: outputs left after this one (2L-1-t)",
    "correct": "1 if every copy-phase input so far matches the original prefix (model copying correctly), else 0; 1 outside the copy phase",
    "in_count": "how many times in_tok occurs among the L original tokens",
    "out_count": "how many times out_tok occurs among the original tokens (0 if t < L)",
    "in_first": "1 if this is the first occurrence of in_tok in the sequence so far",
    "n_digits": "number of digit tokens among the original tokens",
    "first_tok": "the first original token of the sequence (orig(0))",
    "prev2_tok": "input token two steps back (-1 if t < 2)",
    "next_out": "token output at the NEXT step (orig(recall_idx + 1)); -1 if none (t < L - 1 or last step)",
}
HELPERS_DOC = ("Helpers: tok('a') -> token index; isin(col, [idx, ...]); LETTERS, DIGITS, VOWELS (index lists); "
               "orig(k) -> original token k of the row's sequence (-1 if k >= L); "
               "rel_tok(d) -> input token at step t+d (-1 outside the sequence); "
               "out_at(d) -> token output at step t+d (-1 if that step outputs nothing); "
               "seen([idx, ...]) -> True if any input BEFORE this step is in the list; "
               "n_seen([idx, ...]) -> how many inputs before this step are in the list; "
               "in_orig([idx, ...]) -> True if any original token (whole sequence) is in the list; "
               "numpy as np. Combine with & | ~ and parentheses, e.g. "
               "\"(phase == 2) & (recall_idx == 1) & isin(out_tok, DIGITS)\" or "
               "\"(phase == 0) & (rel_tok(-1) == tok('b')) & (orig(0) == tok('o'))\".")


# ----------------------------------------------------------------------------- step table

def build_step_table(sequences) -> Dict[str, np.ndarray]:
    """Derived variables for every (sequence, step), sequences in store order."""
    seq_len = np.array([len(s) for s in sequences], dtype=np.int64)
    row_start = np.zeros(len(sequences) + 1, dtype=np.int64)
    np.cumsum(seq_len, out=row_start[1:])
    n_rows = int(row_start[-1])
    cols = {name: np.empty(n_rows, dtype=np.int16) for name in
            ("t", "L", "phase", "in_tok", "prev_tok", "out_tok", "orig_idx", "recall_idx",
             "to_go", "correct", "in_count", "out_count", "in_first", "n_digits")}
    cols["seq"] = np.empty(n_rows, dtype=np.int32)
    for T in np.unique(seq_len):
        ids = np.nonzero(seq_len == T)[0]
        L = T // 2
        toks = np.array([[TOKEN_IDX[x] for x in sequences[i]] for i in ids], dtype=np.int16)  # (n, T)
        n = len(ids)
        t = np.broadcast_to(np.arange(T), (n, T))
        orig = toks[:, :L]
        out = np.full((n, T), -1, dtype=np.int16)
        out[:, L:] = orig                                                    # step L + k outputs orig[k]
        prev = np.full((n, T), -1, dtype=np.int16)
        prev[:, 1:] = toks[:, :-1]
        # copy inputs at steps L+1.. are the outputs of steps L.., i.e. should equal orig[0..L-2]
        ok = np.ones((n, T), dtype=bool)
        if L > 1:
            match = toks[:, L + 1:] == orig[:, :L - 1]                       # (n, L-1)
            ok[:, L + 1:] = np.cumprod(match, axis=1).astype(bool)
        in_count = (toks[:, :, None] == orig[:, None, :]).sum(-1)
        out_count = np.where(out >= 0, (out[:, :, None] == orig[:, None, :]).sum(-1), 0)
        first = np.ones((n, T), dtype=bool)
        for k in range(1, T):
            first[:, k] = ~(toks[:, :k] == toks[:, k:k + 1]).any(1)
        rows = (row_start[ids][:, None] + np.arange(T)).reshape(-1)
        put = lambda name, val: cols[name].__setitem__(rows, np.asarray(val).reshape(-1))
        put("seq", np.broadcast_to(ids[:, None], (n, T)))
        put("t", t); put("L", np.full((n, T), L)); put("in_tok", toks); put("prev_tok", prev)
        put("phase", np.where(t < L, 0, np.where(t == L, 1, 2)))
        put("out_tok", out)
        put("orig_idx", np.where(t < L, t, -1))
        put("recall_idx", np.where(t >= L, t - L, -1))
        put("to_go", np.where(t < L, L - 1 - t, T - 1 - t))
        put("correct", ok); put("in_count", in_count); put("out_count", out_count)
        put("in_first", first); put("n_digits", np.broadcast_to(np.isin(orig, DIGITS).sum(1)[:, None], (n, T)))
    cols["row_start"] = row_start
    return cols


class Interp:
    def __init__(self, store_path: str):
        self.store_path = store_path
        self._sequences = None          # loaded on first use (only examples need them)
        # The step table is cached one .npy per column and memory-mapped, so
        # concurrent processes share a single copy through the page cache.
        cache = os.path.join(store_path, "autointerp_steps")
        if not os.path.exists(os.path.join(cache, "DONE")):
            self._build_cache(cache)
        self.cols = {name[:-4]: np.load(os.path.join(cache, name), mmap_mode="r")
                     for name in os.listdir(cache) if name.endswith(".npy")}
        self._start = self.cols.pop("_start")
        self.n_rows = len(self.cols["t"])
        self.test = (self.cols["seq"] % 2).astype(bool)

    @property
    def sequences(self):
        if self._sequences is None:
            with open(os.path.join(self.store_path, "sequences.pkl"), "rb") as f:
                self._sequences = pickle.load(f)
        return self._sequences

    def _build_cache(self, cache: str):
        import shutil
        tmp = f"{cache}.tmp{os.getpid()}"
        os.makedirs(tmp, exist_ok=True)
        self.cols = build_step_table(self.sequences)
        self.n_rows = len(self.cols["t"])
        # History lookups, as row arithmetic: rows of a sequence are contiguous.
        self._start = self.cols["row_start"][self.cols["seq"]]
        self.cols["first_tok"] = self.orig(0)
        self.cols["prev2_tok"] = self.rel_tok(-2)
        self.cols["next_out"] = self.out_at(1)
        for name, col in list(self.cols.items()) + [("_start", self._start)]:
            np.save(os.path.join(tmp, f"{name}.npy"), np.ascontiguousarray(col))
        open(os.path.join(tmp, "DONE"), "w").close()
        if os.path.exists(cache):      # another process finished first
            shutil.rmtree(tmp)
        else:
            os.rename(tmp, cache)

    def _tok_at_step(self, step: np.ndarray) -> np.ndarray:
        """Input token at absolute step `step` of each row's sequence (-1 outside)."""
        T = 2 * self.cols["L"].astype(np.int64)
        valid = (step >= 0) & (step < T)
        rows = np.where(valid, self._start + np.clip(step, 0, None), 0)
        return np.where(valid, self.cols["in_tok"][rows], -1).astype(np.int16)

    def orig(self, k: int) -> np.ndarray:
        """Original token k of each row's sequence (-1 if k >= L)."""
        out = self._tok_at_step(np.full(self.n_rows, k, dtype=np.int64))
        return np.where(k < self.cols["L"], out, -1).astype(np.int16)

    def rel_tok(self, d: int) -> np.ndarray:
        """Input token at step t + d."""
        return self._tok_at_step(self.cols["t"].astype(np.int64) + d)

    def _count_before(self, mask: np.ndarray, upto_step: Optional[np.ndarray] = None) -> np.ndarray:
        """Per row: number of rows of the same sequence with mask true at steps < upto_step (default t)."""
        cs = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
        end = self._start + (self.cols["t"] if upto_step is None else upto_step)
        return cs[end] - cs[self._start]

    def n_seen(self, vals) -> np.ndarray:
        """How many inputs before this step are in vals (original and copy phase alike)."""
        return self._count_before(np.isin(self.cols["in_tok"], vals))

    def seen(self, vals) -> np.ndarray:
        """True if any input before this step is in vals."""
        return self.n_seen(vals) > 0

    def in_orig(self, vals) -> np.ndarray:
        """True if any of the L original tokens is in vals (whole original sequence, any step)."""
        mask = np.isin(self.cols["in_tok"], vals) & (self.cols["phase"] == 0)
        return self._count_before(mask, upto_step=self.cols["L"].astype(np.int64)) > 0

    def out_at(self, d: int) -> np.ndarray:
        """Token output at step t + d: original token (t + d - L) when t + d >= L."""
        k = self.cols["t"].astype(np.int64) + d - self.cols["L"]
        T = 2 * self.cols["L"].astype(np.int64)
        valid = (k >= 0) & (self.cols["t"] + d < T)
        out = self._tok_at_step(np.where(valid, k, -1))
        return np.where(valid, out, -1).astype(np.int16)

    def activations(self, gate: str, feature: int) -> np.ndarray:
        """Dense activation vector over all (sequence, step) rows."""
        act = np.zeros(self.n_rows, dtype=np.float32)
        path = os.path.join(self.store_path, gate, f"{feature}.npz")
        if not os.path.exists(path):
            return act                      # never fires
        d = np.load(path)
        lengths = np.diff(d["offsets"])
        rows = np.repeat(self.cols["row_start"][d["seq_ids"]], lengths) + d["positions"].astype(np.int64)
        act[rows] = d["magnitudes"]
        return act

    def predicate(self, expr: str) -> np.ndarray:
        ns = {k: v for k, v in self.cols.items() if k not in ("row_start",)}
        ns.update(np=np, tok=lambda s: TOKEN_IDX[s], isin=lambda col, vals: np.isin(col, vals),
                  LETTERS=LETTERS, DIGITS=DIGITS, VOWELS=VOWELS,
                  orig=self.orig, rel_tok=self.rel_tok, out_at=self.out_at,
                  seen=self.seen, n_seen=self.n_seen, in_orig=self.in_orig)
        out = eval(expr, {"__builtins__": {}}, ns)
        out = np.broadcast_to(np.asarray(out), (self.n_rows,))
        if out.dtype != bool:
            raise ValueError("predicate must evaluate to a boolean array")
        return out


# ----------------------------------------------------------------------------- profile

def _render(seq, positions_mags, max_mag):
    out = []
    for t, tokn in enumerate(seq):
        m = positions_mags.get(t)
        out.append(f"[{tokn}:{m / max_mag:.2f}]" if m else tokn)
    return " ".join(out)


def profile(ip: Interp, gate: str, feature: int, n_examples: int = 12) -> str:
    act = ip.activations(gate, feature)
    dev = ~ip.test
    a, c = act[dev], {k: v[dev] for k, v in ip.cols.items() if k not in ("row_start",)}
    active = a > 0
    lines = [f"# Feature {gate}/{feature} (dev split: even sequence ids)"]
    n_act, n_rows = int(active.sum()), len(a)
    n_seqs = len(np.unique(c["seq"][active])) if n_act else 0
    lines.append(f"active on {n_act} of {n_rows} steps ({n_act / n_rows:.4%}), in {n_seqs} of "
                 f"{len(np.unique(c['seq']))} sequences")
    if n_act == 0:
        lines.append("Feature never fires on the dev split.")
        return "\n".join(lines)
    mags = a[active]
    lines.append(f"magnitude: mean {mags.mean():.3f}, median {np.median(mags):.3f}, max {mags.max():.3f}")

    def table(name, labels=None, top=None):
        vals = c[name]
        keys = np.unique(vals)
        rows = []
        for k in keys:
            sel = vals == k
            n = int(sel.sum())
            k_act = int((active & sel).sum())
            rows.append((k, n, k_act, k_act / n if n else 0.0, float(a[sel].sum()) / float(a.sum())))
        if top:
            rows = sorted(rows, key=lambda r: -r[3])[:top]
        base = n_act / n_rows
        lines.append(f"\n## by {name}: value | steps | active | rate | lift | share of activation mass")
        for k, n, k_act, rate, mass in rows:
            label = labels(k) if labels else k
            lines.append(f"  {label}: {n} | {k_act} | {rate:.3f} | {rate / base:.1f}x | {mass:.3f}")

    tokname = lambda k: TOKENS[k] if k >= 0 else "-"
    table("phase", labels=lambda k: PHASES[int(k)])
    table("t"); table("L"); table("orig_idx"); table("recall_idx"); table("to_go")
    table("in_tok", labels=tokname, top=12)
    table("out_tok", labels=tokname, top=12)
    table("prev_tok", labels=tokname, top=8)
    table("prev2_tok", labels=tokname, top=8)
    table("next_out", labels=tokname, top=8)
    table("first_tok", labels=tokname, top=8)
    table("correct"); table("in_count"); table("in_first"); table("n_digits")

    # examples: top activations, then samples from activation intervals
    order = np.argsort(-a)
    seq_rows = ip.cols["row_start"]
    dev_index = np.nonzero(dev)[0]

    def example(i):
        row = dev_index[i]
        s = int(ip.cols["seq"][row])
        lo, hi = seq_rows[s], seq_rows[s + 1]
        pm = {t: float(act[lo + t]) for t in range(hi - lo) if act[lo + t] > 0}
        return f"  ({gate}/{feature} max {a[i]:.3f}) " + _render(ip.sequences[s], pm, mags.max())

    lines.append(f"\n## top activating steps (token:[activation / max]); one line per sequence")
    seen = set()
    for i in order:
        s = int(c["seq"][i])
        if s in seen or a[i] <= 0:
            continue
        seen.add(s)
        lines.append(example(i))
        if len(seen) >= n_examples:
            break
    rng = np.random.default_rng(0)
    active_idx = np.nonzero(active)[0]
    qs = np.quantile(mags, [0.0, 0.25, 0.5, 0.75])
    for lo_q, hi_q, name in ((qs[0], qs[1], "lowest quartile"), (qs[1], qs[2], "second quartile"),
                             (qs[2], qs[3], "third quartile")):
        pool = active_idx[(a[active_idx] >= lo_q) & (a[active_idx] < hi_q)]
        if len(pool):
            lines.append(f"\n## random examples from the {name} of activations")
            for i in rng.choice(pool, min(4, len(pool)), replace=False):
                lines.append(example(i))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- score

def score(ip: Interp, gate: str, feature: int, expr: str, split: str = "dev") -> Dict:
    act = ip.activations(gate, feature)
    pred = ip.predicate(expr)
    sel = ip.test if split == "test" else ~ip.test
    a, p = act[sel], pred[sel]
    active = a > 0
    n_act, n_pred, both = int(active.sum()), int(p.sum()), int((active & p).sum())
    precision = both / n_pred if n_pred else 0.0
    recall = both / n_act if n_act else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mass_recall = float(a[p].sum()) / float(a.sum()) if a.sum() > 0 else 0.0
    corr = float(np.corrcoef(a, p.astype(np.float32))[0, 1]) if n_pred and n_pred < len(p) and n_act else 0.0
    # specificity by activation interval: fraction of active steps the predicate covers, for
    # four equal-size groups of active steps ordered by magnitude (rank-based, so features
    # with few distinct magnitudes still get four groups)
    by_interval = []
    if n_act:
        idx = np.nonzero(active)[0]
        ranked = idx[np.argsort(a[idx], kind="stable")]
        by_interval = [float(p[chunk].mean()) for chunk in np.array_split(ranked, 4) if len(chunk)]
    return {"gate": gate, "feature": feature, "predicate": expr, "split": split,
            "precision": precision, "recall": recall, "f1": f1, "mass_recall": mass_recall,
            "correlation": corr, "n_active": n_act, "n_predicted": n_pred,
            "coverage_by_activation_quartile": by_interval}


def report(out_dir: str) -> str:
    """Summarise a results directory: verdict counts per gate, score distributions, CSV of all features."""
    import csv
    rows = []
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json"):
            continue
        rec = json.load(open(os.path.join(out_dir, name)))
        t = rec.get("test_scores") or {}
        rows.append({"gate": rec.get("gate"), "feature": rec.get("feature"),
                     "verdict": rec.get("final_verdict") or rec.get("verdict"),
                     "label": rec.get("final_label") or rec.get("label"),
                     "predicate": rec.get("predicate"), "test_precision": t.get("precision"),
                     "test_recall": t.get("recall"), "test_f1": t.get("f1"), "test_mass_recall": t.get("mass_recall"),
                     "skeptic_upheld": (rec.get("skeptic") or {}).get("upheld"),
                     # Fitted scorecards (circuit/scorecards.py), for features no short predicate explains.
                     "scorecard_format": (rec.get("scorecard") or {}).get("format"),
                     "scorecard_test_f1": (rec.get("scorecard") or {}).get("test_f1"),
                     "full_scorecard_test_f1": (rec.get("scorecard") or {}).get("full_test_f1"),
                     "scorecard_summary": (rec.get("scorecard") or {}).get("summary")})
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["gate"])
        w.writeheader(); w.writerows(rows)
    lines = [f"{len(rows)} features in {out_dir} (CSV: summary.csv)"]
    for gate in ("reset", "update", "hidden"):
        g = [r for r in rows if r["gate"] == gate]
        if not g:
            continue
        counts = {}
        for r in g:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        f1 = [r["test_f1"] for r in g if r["test_f1"] is not None]
        lines.append(f"{gate}: {len(g)} features; " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
                     + (f"; median test F1 {np.median(f1):.2f}" if f1 else ""))
    return "\n".join(lines)


def misses(ip: Interp, gate: str, feature: int, expr: str, split: str = "dev", n: int = 8) -> str:
    """Counterexamples: active steps the predicate misses, and predicted steps that are inactive."""
    act = ip.activations(gate, feature)
    pred = ip.predicate(expr)
    sel = ip.test if split == "test" else ~ip.test
    max_mag = float(act.max()) if act.max() > 0 else 1.0
    rs = ip.cols["row_start"]
    rng = np.random.default_rng(0)
    lines = [f"# Counterexamples for {gate}/{feature} ({split} split)", f"predicate: {expr}",
             "Notation: [tok:a] = active with activation a (relative to the max); *tok* = predicate true there."]
    for title, mask in (("ACTIVE but predicate FALSE (missed activations)", (act > 0) & ~pred & sel),
                        ("predicate TRUE but INACTIVE (false positives)", (act == 0) & pred & sel)):
        rows = np.nonzero(mask)[0]
        lines.append(f"\n## {title}: {len(rows)} steps")
        if title.startswith("ACTIVE") and len(rows):
            share = float(act[rows].sum()) / float(act[sel].sum())
            lines.append(f"   (these hold {share:.1%} of the activation mass)")
        for r in rng.choice(rows, min(n, len(rows)), replace=False) if len(rows) else []:
            s = int(ip.cols["seq"][r]); lo, hi = rs[s], rs[s + 1]
            toks = []
            for t, tk in enumerate(ip.sequences[s]):
                a, p = act[lo + t], pred[lo + t]
                cell = f"[{tk}:{a / max_mag:.2f}]" if a > 0 else tk
                toks.append(f"*{cell}*" if p else cell)
            lines.append(f"  step {int(ip.cols['t'][r])}: " + " ".join(toks))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="Activation-based feature interpretation (copy task)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile"); p.add_argument("store"); p.add_argument("gate"); p.add_argument("feature", type=int)
    s = sub.add_parser("score"); s.add_argument("store"); s.add_argument("gate"); s.add_argument("feature", type=int)
    s.add_argument("predicate"); s.add_argument("--split", choices=["dev", "test"], default="dev")
    r = sub.add_parser("record"); r.add_argument("store"); r.add_argument("gate"); r.add_argument("feature", type=int)
    r.add_argument("result_json", help='{"label": ..., "predicate": ... or null, "verdict": ..., "notes": ...}')
    r.add_argument("--out", required=True)
    m = sub.add_parser("misses"); m.add_argument("store"); m.add_argument("gate"); m.add_argument("feature", type=int)
    m.add_argument("predicate"); m.add_argument("--split", choices=["dev", "test"], default="dev")
    rp = sub.add_parser("report"); rp.add_argument("out_dir")
    sub.add_parser("variables")
    args = ap.parse_args()

    if args.cmd == "variables":
        for k, v in VARIABLE_DOCS.items():
            print(f"{k:12s} {v}")
        print(HELPERS_DOC)
        return
    if args.cmd == "report":
        print(report(args.out_dir))
        return
    ip = Interp(args.store)
    if args.cmd == "profile":
        print(profile(ip, args.gate, args.feature))
    elif args.cmd == "score":
        print(json.dumps(score(ip, args.gate, args.feature, args.predicate, args.split), indent=2))
    elif args.cmd == "misses":
        print(misses(ip, args.gate, args.feature, args.predicate, args.split))
    elif args.cmd == "record":
        result = json.loads(args.result_json)
        if result.get("predicate"):
            # Reported scores are always on the held-out test split.
            result["test_scores"] = score(ip, args.gate, args.feature, result["predicate"], "test")
            result["dev_scores"] = score(ip, args.gate, args.feature, result["predicate"], "dev")
        result.update(gate=args.gate, feature=args.feature)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"{args.gate}_{args.feature}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"recorded {path}")
        if "test_scores" in result:
            t = result["test_scores"]
            print(f"test: precision {t['precision']:.3f}, recall {t['recall']:.3f}, f1 {t['f1']:.3f}, "
                  f"mass recall {t['mass_recall']:.3f}")


if __name__ == "__main__":
    main()
