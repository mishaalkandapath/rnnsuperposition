"""Magnitude explanations for dense features: additive token tables, scored on held-out data.

Firing predicates (circuit/copy_task/autointerp.py) cannot describe a feature that is on
almost everywhere: what it encodes is in its magnitude. Here a magnitude
explanation is

    activation ~ timing[L, t] + sum over cues c of table_c[phase, token_c]

with cues such as the current input, the last few inputs, or the token being
output and the next one; tables are phase-specific (original / <DEL> / copy).
Models are fitted by ridge on the dev split (even sequence ids) and scored on
the test split (odd ids):

  r2               1 - SSE / SST
  r2_beyond_timing 1 - SSE / SSE(timing only): the share of what timing leaves
                   unexplained that the token tables capture (the headline)
  readable         the "all cues" model with only the top-k tokens per table,
                   refitted: what a short label keeps
  label            auto-generated from the full model's largest table effects
  token sets       the compact form: one set of tokens that raises the feature (S+)
                   and one that lowers it (S-), each with a weight per (cue, phase),
                   i.e. "S+ remembered over the last few inputs". Set size chosen on dev.

The design is one-hot over categorical columns, so X'X is a table of
co-occurrence counts and is shared by every feature; each feature only needs
X'y (sums of its activation per category).

    python -m circuit.copy_task.magnitude_scores --store features/<run>_store --gate hidden \\
        --min_rate 0.2 --feature_results features/<run>_autointerp --out magnitude_scores.json
"""
import argparse
import json
import os

import numpy as np

from circuit.copy_task.autointerp import Interp, PHASES, TOKENS

# cue -> token per row: inputs d steps back (rel_tok(-d)), outputs d steps ahead (out_at(d)),
# and the first original token. In the copy phase the outputs ahead are the tokens still in memory.
CUES = {"input": ("rel", 0), "prev": ("rel", -1), "prev2": ("rel", -2), "prev3": ("rel", -3),
        "prev4": ("rel", -4), "prev5": ("rel", -5), "prev6": ("rel", -6), "prev7": ("rel", -7),
        "out": ("out", 0), "next_out": ("out", 1), "out+2": ("out", 2), "out+3": ("out", 3),
        "first": ("orig", 0)}
INPUTS8 = ["input", "prev", "prev2", "prev3", "prev4", "prev5", "prev6", "prev7"]
MODELS = {"timing": [], "+input": ["input"], "+last 2 inputs": ["input", "prev"],
          "+last 4 inputs": INPUTS8[:4], "+last 8 inputs": INPUTS8,
          "+outputs (current, next)": ["out", "next_out"],
          "+outputs (current .. 3 ahead)": ["out", "next_out", "out+2", "out+3"],
          "+all": INPUTS8 + ["out", "next_out", "out+2", "out+3", "first"]}
NTOK = 32                                                                 # token -1 .. 30


def encode(ip, rows):
    """Categorical codes per column for the chosen rows: timing (L, t), and cue x phase."""
    c = ip.cols
    codes = {"timing": ((c["L"][rows].astype(np.int64) - 3) * 18 + c["t"][rows]), }
    phase = c["phase"][rows].astype(np.int64)
    for cue, (kind, d) in CUES.items():
        tok = {"rel": ip.rel_tok, "out": ip.out_at, "orig": ip.orig}[kind](d)[rows].astype(np.int64)
        codes[cue] = phase * NTOK + tok + 1
    sizes = {"timing": 7 * 18, **{cue: 3 * NTOK for cue in CUES}}
    return codes, sizes


def gram(codes, sizes, names):
    """X'X for the one-hot design of the named columns, and column offsets."""
    offs = np.concatenate([[0], np.cumsum([sizes[n] for n in names])])
    G = np.zeros((offs[-1], offs[-1]))
    for a, na in enumerate(names):
        for b, nb in enumerate(names[a:], a):
            h = np.zeros(sizes[na] * sizes[nb])
            np.add.at(h, codes[na] * sizes[nb] + codes[nb], 1.0)
            G[offs[a]:offs[a + 1], offs[b]:offs[b + 1]] = h.reshape(sizes[na], sizes[nb])
            G[offs[b]:offs[b + 1], offs[a]:offs[a + 1]] = h.reshape(sizes[na], sizes[nb]).T
    return G, offs


def xty(codes, sizes, names, y):
    return np.concatenate([np.bincount(codes[n], weights=y, minlength=sizes[n]) for n in names])


def predict(codes, names, offs, w):
    return sum(w[offs[k] + codes[n]] for k, n in enumerate(names))


def ridge(G, b, cols, lam):
    sub = G[np.ix_(cols, cols)]
    return np.linalg.solve(sub + lam * np.eye(len(cols)), b[cols])


def r2(y, p, base_sse=None):
    sse = float(((y - p) ** 2).sum())
    return 1 - sse / float(((y - y.mean()) ** 2).sum()), (1 - sse / base_sse if base_sse else None), sse


def describe_tables(w, offs, names, counts, top=4):
    """Readable label: per (cue, phase) table, the tokens that raise / lower the feature most."""
    parts = []
    for k, n in enumerate(names):
        if n == "timing":
            continue
        for ph in range(3):
            idx = np.arange(ph * NTOK, (ph + 1) * NTOK)
            seen = counts[offs[k] + idx] >= 50
            if seen.sum() < 2:
                continue
            vals = w[offs[k] + idx]
            vals = vals - vals[seen].mean()
            order = [j for j in np.argsort(vals) if seen[j]]
            up = [(TOKENS[j - 1] if j > 0 else "-", round(float(vals[j]), 2)) for j in order[::-1][:top] if vals[j] > 0.1]
            down = [(TOKENS[j - 1] if j > 0 else "-", round(float(vals[j]), 2)) for j in order[:top] if vals[j] < -0.1]
            if up or down:
                parts.append({"cue": n, "phase": PHASES[ph], "spread": float(vals[seen].max() - vals[seen].min()),
                              "raised_by": up, "lowered_by": down})
    parts.sort(key=lambda p: -p["spread"])
    return parts


def token_sets(w, offs, names, counts, k):
    """S+ / S-: the k tokens that raise / lower the feature most, summed over every
    (cue, phase) table weighted by the table's spread (so the dominant code wins)."""
    score = np.zeros(NTOK)
    for j, n in enumerate(names):
        if n == "timing":
            continue
        for ph in range(3):
            idx = offs[j] + np.arange(ph * NTOK, (ph + 1) * NTOK)
            seen = counts[idx] >= 50
            if seen.sum() < 2:
                continue
            vals = np.where(seen, w[idx] - w[idx][seen].mean(), 0.0)
            score += vals * (vals.max() - vals.min())
    score[0] = 0.0                                                       # "no token" is not a token
    order = np.argsort(score)
    return [int(t) for t in order[::-1][:k] if score[t] > 0], [int(t) for t in order[:k] if score[t] < 0]


def set_codes(codes, plus, minus):
    """Per cue: phase * 3 + (0 other, 1 in S+, 2 in S-)."""
    out = {"timing": codes["timing"]}
    lut = np.zeros(3 * NTOK, dtype=np.int64)
    for ph in range(3):
        lut[ph * NTOK + np.array(plus, dtype=np.int64)] = 1
        lut[ph * NTOK + np.array(minus, dtype=np.int64)] = 2
    for cue in CUES:
        out[cue] = (codes[cue] // NTOK) * 3 + lut[codes[cue]]
    return out


def short_label(parts, n=3):
    def toks(items):
        return " ".join(t for t, _ in items[:3])
    bits = [f"{p['cue']}@{p['phase'][:4]}: +{toks(p['raised_by'])}" + (f" -{toks(p['lowered_by'])}" if p["lowered_by"] else "")
            for p in parts[:n]]
    return "; ".join(bits)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--gate", default="hidden")
    ap.add_argument("--features", type=int, nargs="*", help="default: all with firing rate >= --min_rate")
    ap.add_argument("--min_rate", type=float, default=0.2)
    ap.add_argument("--feature_results", default=None, help="autointerp results (firing labels, for comparison)")
    ap.add_argument("--n_sequences", type=int, default=60000, help="sequences per split")
    ap.add_argument("--top_k", type=int, default=5, help="tokens kept per table in the readable model")
    ap.add_argument("--set_sizes", type=int, nargs="+", default=[2, 3, 4, 6, 8], help="sizes of S+ / S- tried")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ip = Interp(args.store)
    store = json.load(open(os.path.join(args.store, "store.json")))
    rates = {int(f): c[1] / ip.n_rows for f, c in store["counts"][args.gate].items()}
    feats = args.features or sorted(f for f, r in rates.items() if r >= args.min_rate)
    rng = np.random.default_rng(args.seed)
    n_seq = len(ip.cols["row_start"]) - 1
    split_rows = {}
    for name, parity in (("dev", 0), ("test", 1)):
        seqs = np.sort(rng.choice(np.arange(parity, n_seq, 2), size=min(args.n_sequences, n_seq // 2), replace=False))
        starts, ends = ip.cols["row_start"][seqs], ip.cols["row_start"][seqs + 1]
        split_rows[name] = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
    enc = {name: encode(ip, rows) for name, rows in split_rows.items()}
    names_all = ["timing"] + list(CUES)
    codes_dev, sizes = enc["dev"]
    codes_test, _ = enc["test"]
    G, offs = gram(codes_dev, sizes, names_all)
    counts = np.diag(G)
    col_of = {n: np.arange(offs[k], offs[k + 1]) for k, n in enumerate(names_all)}
    print(f"{len(feats)} features; dev rows {len(split_rows['dev'])}, test rows {len(split_rows['test'])}")

    labels = {}
    if args.feature_results:
        for f in feats:
            p = os.path.join(args.feature_results, f"{args.gate}_{f}.json")
            if os.path.exists(p):
                r = json.load(open(p))
                labels[f] = {"firing_label": r.get("label"), "firing_verdict": r.get("final_verdict"),
                             "firing_test_f1": (r.get("test_scores") or {}).get("f1")}

    results = []
    for f in feats:
        act = ip.activations(args.gate, f)
        y_dev, y_test = act[split_rows["dev"]].astype(np.float64), act[split_rows["test"]].astype(np.float64)
        b = xty(codes_dev, sizes, names_all, y_dev)
        rec = {"feature": f, "fire_rate": rates[f], **labels.get(f, {}), "models": {}}
        base_sse = None
        for model, cues in MODELS.items():
            names = ["timing"] + cues
            cols = np.concatenate([col_of[n] for n in names])
            w_full = np.zeros(offs[-1])
            w_full[cols] = ridge(G, b, cols, args.lam)
            p = predict(codes_test, names_all, offs, w_full)
            r2_total, r2_beyond, sse = r2(y_test, p, base_sse)
            if model == "timing":
                base_sse = sse
            rec["models"][model] = {"r2": r2_total, "r2_beyond_timing": r2_beyond}
            if model == "+all":
                w_all = w_full
        # readable: top-k tokens per (cue, phase) table of the full model, refitted with timing
        keep = list(col_of["timing"])
        for n in CUES:
            for ph in range(3):
                idx = offs[names_all.index(n)] + np.arange(ph * NTOK, (ph + 1) * NTOK)
                seen = idx[counts[idx] >= 50]
                if len(seen) < 2:
                    continue
                vals = w_all[seen] - w_all[seen].mean()
                keep += list(seen[np.argsort(-np.abs(vals))[:args.top_k]])
        keep = np.array(sorted(set(keep)))
        w_read = np.zeros(offs[-1])
        w_read[keep] = ridge(G, b, keep, args.lam)
        r2_total, r2_beyond, _ = r2(y_test, predict(codes_test, names_all, offs, w_read), base_sse)
        rec["models"][f"readable (top {args.top_k} per table)"] = {"r2": r2_total, "r2_beyond_timing": r2_beyond}
        parts = describe_tables(w_all, offs, names_all, counts)
        rec["tables"] = parts[:8]
        rec["magnitude_label"] = short_label(parts)
        # set model: timing + per (cue, phase) weights on "token in S+" and "token in S-"; k chosen on dev
        set_sizes = {"timing": sizes["timing"], **{cue: 9 for cue in CUES}}
        best = None
        for k in args.set_sizes:
            plus, minus = token_sets(w_all, offs, names_all, counts, k)
            sc_dev, sc_test = set_codes(codes_dev, plus, minus), set_codes(codes_test, plus, minus)
            Gs, offs_s = gram(sc_dev, set_sizes, names_all)
            ws = np.linalg.solve(Gs + args.lam * np.eye(len(Gs)), xty(sc_dev, set_sizes, names_all, y_dev))
            dev_r2 = r2(y_dev, predict(sc_dev, names_all, offs_s, ws))[0]
            if best is None or dev_r2 > best[0]:
                best = (dev_r2, k, plus, minus, ws, offs_s, sc_test)
        _, k, plus, minus, ws, offs_s, sc_test = best
        r2_total, r2_beyond, _ = r2(y_test, predict(sc_test, names_all, offs_s, ws), base_sse)
        rec["models"][f"token sets (S+/S-, k chosen on dev)"] = {"r2": r2_total, "r2_beyond_timing": r2_beyond}
        weights = []
        for j, n in enumerate(names_all[1:], 1):
            for ph in range(3):
                wp, wm = ws[offs_s[j] + ph * 3 + 1] - ws[offs_s[j] + ph * 3], ws[offs_s[j] + ph * 3 + 2] - ws[offs_s[j] + ph * 3]
                if abs(wp) > 0.05 or abs(wm) > 0.05:
                    weights.append({"cue": n, "phase": PHASES[ph], "S+": round(float(wp), 2), "S-": round(float(wm), 2)})
        weights.sort(key=lambda x: -(abs(x["S+"]) + abs(x["S-"])))
        rec["token_sets"] = {"S+": [TOKENS[t - 1] for t in plus], "S-": [TOKENS[t - 1] for t in minus], "k": k,
                             "weights": weights[:10]}
        top_w = ", ".join(f"{x['cue']}@{x['phase'][:4]} {x['S+']:+.1f}/{x['S-']:+.1f}" for x in weights[:4])
        rec["set_label"] = (f"S+ {{{' '.join(rec['token_sets']['S+'])}}} vs S- {{{' '.join(rec['token_sets']['S-'])}}}; "
                            f"weights (S+/S-) {top_w}")
        results.append(rec)
        m = rec["models"]
        print(f"{f:>3} rate {rates[f]:.2f} | R2 timing {m['timing']['r2']:.2f} | beyond timing: input {m['+input']['r2_beyond_timing']:.2f} "
              f"last2 {m['+last 2 inputs']['r2_beyond_timing']:.2f} last8 {m['+last 8 inputs']['r2_beyond_timing']:.2f} "
              f"out0-3 {m['+outputs (current .. 3 ahead)']['r2_beyond_timing']:.2f} all {m['+all']['r2_beyond_timing']:.2f} "
              f"readable {m[f'readable (top {args.top_k} per table)']['r2_beyond_timing']:.2f} "
              f"sets {m['token sets (S+/S-, k chosen on dev)']['r2_beyond_timing']:.2f} | {rec['set_label'][:110]}")

    summary = {model: {"median_r2_beyond_timing": float(np.median([r["models"][model]["r2_beyond_timing"] or 0 for r in results])),
                       "median_r2": float(np.median([r["models"][model]["r2"] for r in results]))}
               for model in results[0]["models"]}
    print(json.dumps(summary, indent=1))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "summary": summary, "features": results}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
