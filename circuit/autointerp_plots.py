"""Figures for the activation-based interpretations (circuit/autointerp.py results).

    python -m circuit.autointerp_plots specificity RESULTS_DIR [--out FILE]
    python -m circuit.autointerp_plots coverage STORE RESULTS_DIR [--out FILE]
    python -m circuit.autointerp_plots fidelity FIDELITY_JSON [--out DIR]
    python -m circuit.autointerp_plots probe PROBE_JSON [--out DIR]

specificity: does an explanation hold for weak activations as well as strong
ones? For every feature with a recorded predicate, its active test steps are
split into four equal-size groups by activation strength; the y value is the
share of each group the predicate covers (the analogue of Scaling
Monosemanticity's specificity-by-activation plot, scored exactly rather than by
a rater). One panel per verdict, one line per gate (median across features, with
the interquartile range shaded).
"""
import argparse
import json
import os

import numpy as np

GATES = [("reset", "Reset gate", "#2a78d6", "o"), ("update", "Update gate", "#eb6834", "s"),
         ("hidden", "Candidate", "#1baf7a", "^")]
VERDICTS = [("explained", "Explained"), ("partially_explained", "Partially explained"),
            ("no_explanation", "No explanation")]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(results_dir):
    rows = []
    for name in os.listdir(results_dir):
        if not name.endswith(".json"):
            continue
        rec = json.load(open(os.path.join(results_dir, name)))
        cov = (rec.get("test_scores") or {}).get("coverage_by_activation_quartile")
        if cov and len(cov) == 4 and all(c is not None for c in cov):
            rows.append((rec["gate"], rec.get("final_verdict") or rec.get("verdict"), cov))
    return rows


def specificity(results_dir, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(results_dir)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True, facecolor=SURFACE)
    x = np.arange(1, 5)
    for ax, (verdict, title) in zip(axes, VERDICTS):
        ax.set_facecolor(SURFACE)
        for gate, label, color, marker in GATES:
            cov = np.array([c for g, v, c in rows if g == gate and v == verdict])
            if len(cov) < 3:
                continue
            med = np.median(cov, axis=0)
            lo, hi = np.percentile(cov, 25, axis=0), np.percentile(cov, 75, axis=0)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, med, color=color, linewidth=2, marker=marker, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=f"{label} (n={len(cov)})")
            # identity is carried by the legend plus a distinct marker per gate
            # (end-of-line labels collide where lines converge at 1.0)
        ax.set_title(title, fontsize=12, color=INK, loc="left")
        ax.set_xticks(x, ["weakest\n25%", "2nd", "3rd", "strongest\n25%"], fontsize=9, color=INK_2)
        ax.set_xlim(0.7, 4.3)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2)
        ax.legend(loc="best", fontsize=8.5, frameon=False, labelcolor=INK_2)
    axes[0].set_ylabel("Share of active steps the explanation covers", fontsize=10, color=INK)
    fig.supxlabel("Activation strength (a feature's active test steps in four equal groups)", fontsize=10, color=INK)
    fig.suptitle("Do explanations hold for weak activations as well as strong ones?", fontsize=13, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")

    # numbers behind the figure
    for verdict, title in VERDICTS:
        for gate, label, _, _ in GATES:
            cov = np.array([c for g, v, c in rows if g == gate and v == verdict])
            if len(cov) >= 3:
                print(f"{title:20s} {label:12s} n={len(cov):4d} median coverage by quartile: "
                      + " ".join(f"{m:.2f}" for m in np.median(cov, axis=0)))


F1_BANDS = [("F1 >= 0.8", 0.8, 1.01), ("0.5 - 0.8", 0.5, 0.8), ("0.2 - 0.5", 0.2, 0.5), ("< 0.2 or none", -1.0, 0.2)]
F1_CUTS = [0.8, 0.5, 0.2, -1.0]


def coverage(store, results, gates=None, subset_store=None):
    """Timestep coverage of features grouped by how well they are explained.

    For each gate, features are grouped by the held-out F1 of their recorded
    predicate (features without one count as F1 0). For each group: the union
    share of timesteps on which at least one of its features is active, and its
    share of all activation events (feature, step). Also cumulative groups
    (F1 >= cut). With subset_store, only the sequences of that store count, so
    stores built on different sequence sets compare on the same steps.
    """
    import pickle
    from circuit.autointerp import Interp
    ip = Interp(store)
    row_start = ip.cols["row_start"]
    if subset_store:
        seq_id = {s: i for i, s in enumerate(ip.sequences)}
        ids = np.array([seq_id[s] for s in pickle.load(open(os.path.join(subset_store, "sequences.pkl"), "rb"))])
        mask = np.zeros(ip.n_rows, dtype=bool)
        for a, b in zip(row_start[ids], row_start[ids + 1]):
            mask[a:b] = True
    else:
        mask = np.ones(ip.n_rows, dtype=bool)
    n_steps = int(mask.sum())
    counts = json.load(open(os.path.join(store, "store.json")))["counts"]
    out = {}
    for gate in gates or list(counts):
        f1 = {}
        for feat in counts.get(gate, {}):
            path = os.path.join(results, f"{gate}_{feat}.json")
            rec = json.load(open(path)) if os.path.exists(path) else {}
            f1[feat] = (rec.get("test_scores") or {}).get("f1") or 0.0
        union_band = {name: np.zeros(ip.n_rows, dtype=bool) for name, _, _ in F1_BANDS}
        events_band = dict.fromkeys(union_band, 0)
        n_band = dict.fromkeys(union_band, 0)
        for feat, score in f1.items():
            d = np.load(os.path.join(store, gate, f"{feat}.npz"))
            rows = np.repeat(row_start[d["seq_ids"]], np.diff(d["offsets"])) + d["positions"].astype(np.int64)
            rows = rows[mask[rows]]
            name = next(n for n, lo, hi in F1_BANDS if lo <= score < hi)
            union_band[name][rows] = True
            events_band[name] += len(rows)
            n_band[name] += 1
        total_events = sum(events_band.values())
        bands = {name: {"n": n_band[name], "union": float(union_band[name][mask].mean()),
                        "events_share": events_band[name] / total_events if total_events else 0.0}
                 for name in union_band}
        cumulative, acc, acc_events, acc_n = {}, np.zeros(ip.n_rows, dtype=bool), 0, 0
        for (name, lo, _), cut in zip(F1_BANDS, F1_CUTS):
            acc |= union_band[name]
            acc_events += events_band[name]
            acc_n += n_band[name]
            label = f"F1 >= {cut}" if cut >= 0 else "all"
            cumulative[label] = {"n": acc_n, "union": float(acc[mask].mean()),
                                 "events_share": acc_events / total_events if total_events else 0.0}
        out[gate] = {"steps": n_steps, "bands": bands, "cumulative": cumulative}
    return out


def print_coverage(label, cov):
    print(f"\n== {label}")
    for gate, g in cov.items():
        print(f"  {gate} ({g['steps']} steps)")
        print("    band            n     steps covered (union)   share of activations")
        for name, b in g["bands"].items():
            print(f"    {name:14s} {b['n']:5d}   {b['union']:8.1%}               {b['events_share']:8.1%}")
        print("    cumulative")
        for name, b in g["cumulative"].items():
            print(f"    {name:14s} {b['n']:5d}   {b['union']:8.1%}               {b['events_share']:8.1%}")


# Ordered F1 bands: one-hue sequential ramp, darkest = best explained.
BAND_COLORS = {"F1 >= 0.8": "#184f95", "0.5 - 0.8": "#3987e5", "0.2 - 0.5": "#86b6ef", "< 0.2 or none": "#cde2fb"}
GATE_ROWS = [("reset", "neuron_reset", "Reset", ""), ("update", "neuron_update", "Update", ""),
             ("hidden", "neuron_cand_pos", "Candidate", " (high end)"), (None, "neuron_cand_neg", "Candidate", " (low end)")]
SOURCE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical slots 1-4


def coverage_plots(feature_cov, unit_covs, outdir):
    """Bar charts of coverage_by_f1 numbers.

    feature_cov: coverage() output for the transcoder features.
    unit_covs: {label: coverage() output} for GRU-unit baselines (e.g. {"units, matched": ...}).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)

    def style(ax):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2)

    # 1. share of activations by F1 band: 100% stacked horizontal bars
    rows = []
    for gate, unit_gate, label, suffix in GATE_ROWS:
        if gate:
            rows.append((f"{label}: transcoder features", feature_cov[gate]))
        for ulabel, ucov in unit_covs.items():
            if unit_gate in ucov:
                rows.append((f"{label}: GRU {ulabel}{suffix}", ucov[unit_gate]))
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 1.6), facecolor=SURFACE)
    style(ax)
    y = np.arange(len(rows))[::-1]
    for yi, (label, g) in zip(y, rows):
        left = 0.0
        for name, color in BAND_COLORS.items():
            share = g["bands"][name]["events_share"]
            ax.barh(yi, share, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
            if share >= 0.07:
                ax.text(left + share / 2, yi, f"{share:.0%}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if name in ("F1 >= 0.8", "0.5 - 0.8") else INK)
            left += share
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of all activations (feature x timestep) of that gate", fontsize=10, color=INK)
    ax.legend(handles=[matplotlib.patches.Patch(color=c, label=f"held-out {n}") for n, c in BAND_COLORS.items()],
              ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9, labelcolor=INK_2)
    ax.set_title("How much activity comes from well-explained features?", fontsize=12, color=INK, loc="left", pad=28)
    fig.tight_layout()
    path1 = os.path.join(outdir, "activation_share_by_f1.png")
    fig.savefig(path1, dpi=170, facecolor=SURFACE)
    plt.close(fig)

    # 2. cumulative union coverage: small multiples per gate, grouped bars per source
    cuts = list(next(iter(feature_cov.values()))["cumulative"])
    panels = [("reset", ["neuron_reset"], "Reset gate"), ("update", ["neuron_update"], "Update gate"),
              ("hidden", ["neuron_cand_pos", "neuron_cand_neg"], "Candidate")]
    k = len(unit_covs)   # the candidate panel has two unit rows per baseline, so it gets more width
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1 + k, 1 + k, 1 + 2 * k]})
    for ax, (gate, unit_gates, title) in zip(axes, panels):
        style(ax)
        # color follows the source (features, each unit baseline); candidate low-end units are hatched
        sources = [("transcoder features", feature_cov[gate], SOURCE_COLORS[0], None)]
        for k, (ulabel, ucov) in enumerate(unit_covs.items(), start=1):
            for ug in unit_gates:
                if ug in ucov:
                    hatch = "////" if ug == "neuron_cand_neg" else None
                    sources.append((f"GRU {ulabel}", ucov[ug], SOURCE_COLORS[k % len(SOURCE_COLORS)], hatch))
        width = 0.8 / len(sources)
        x = np.arange(len(cuts))
        for j, (slabel, g, color, hatch) in enumerate(sources):
            vals = [g["cumulative"][c]["union"] for c in cuts]
            bars = ax.bar(x + (j - (len(sources) - 1) / 2) * width, vals, width=width * 0.92, color=color,
                          hatch=hatch, edgecolor=SURFACE, linewidth=2, label=None if hatch else slabel)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", va="bottom",
                        fontsize=7, color=INK_2, rotation=90 if len(sources) > 3 else 0)
        ax.set_xticks(x, [c.replace("F1 >= ", "F1 ≥ ") for c in cuts], fontsize=9, color=INK_2)
        ax.set_ylim(0, 1.18)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=12, color=INK, loc="left")
    axes[0].set_ylabel("Timesteps with at least one active member", fontsize=10, color=INK)
    fig.supxlabel("Features / units included (by held-out F1 of their explanation)", fontsize=10, color=INK)
    fig.suptitle("How much of the data do the explained features touch?", fontsize=13, color=INK, x=0.01, ha="left")
    # one shared legend above the panels (bars reach 100%, so in-axes legends would cover them)
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    handles.append(matplotlib.patches.Patch(facecolor=INK_2, hatch="////", edgecolor=SURFACE))
    labels.append("candidate units, low end (hatched)")
    fig.legend(handles, labels, ncol=len(labels), loc="upper left", bbox_to_anchor=(0.01, 0.93),
               frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    path2 = os.path.join(outdir, "step_coverage_by_f1.png")
    fig.savefig(path2, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path1, path2


FIDELITY_SOURCES = ["features", "units matched", "units d01", "units d05"]
FIDELITY_LABELS = {"features": "transcoder features", "units matched": "GRU units, matched density",
                   "units d01": "GRU units, 1% density", "units d05": "GRU units, 5% density"}
FIDELITY_GATES = [("reset", "Reset gate"), ("update", "Update gate"), ("candidate", "Candidate")]


def fidelity_plots(path, outdir):
    """Figures for circuit/explained_fidelity.py output (top-1 agreement with the original GRU)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    data = json.load(open(path))
    rows, cores, drops = data["rows"], data["core"], data.get("drops", [])
    taus = data["args"]["taus"]

    def style(ax):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2)

    def pick(gate, source, selection, tau=None):
        return [r for r in rows if r["gate"] == gate and r["source"] == source
                and r["selection"] == selection and (tau is None or r["tau"] == tau)]

    # 1. explained selections vs random selections of the same size
    groups = [(f"F1 ≥ {t}", "f1", t) for t in taus] + [("core set", "core", None)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), sharey=True, facecolor=SURFACE)
    width = 0.8 / len(FIDELITY_SOURCES)
    for ax, (gate, title) in zip(axes, FIDELITY_GATES):
        style(ax)
        for j, source in enumerate(FIDELITY_SOURCES):
            for k, (_, selection, tau) in enumerate(groups):
                sel = pick(gate, source, selection, tau)
                if not sel:
                    continue
                xpos = k + (j - (len(FIDELITY_SOURCES) - 1) / 2) * width
                v = sel[0]["top1_agreement"]
                ax.bar(xpos, v, width=width * 0.9, color=SOURCE_COLORS[j], edgecolor=SURFACE, linewidth=1.5,
                       label=FIDELITY_LABELS[source] if k == 0 else None)
                # kept count inside the bar's base (above it when the bar is too short), clear of the dots
                inside = v >= 0.12
                ax.text(xpos, 0.015 if inside else v + 0.015, str(sel[0]["n_kept"]), ha="center", va="bottom",
                        fontsize=6.5, color="#ffffff" if inside else INK_2, rotation=90)
                if selection == "f1":
                    rand = [r["top1_agreement"] for r in pick(gate, source, "random", tau)]
                    ax.scatter([xpos] * len(rand), rand, s=14, facecolor=SURFACE, edgecolor=INK, linewidth=1,
                               zorder=3, label="random, same count" if (j == 0 and k == 0) else None)
        none = pick(gate, "features", "none")
        if none:
            ax.axhline(none[0]["top1_agreement"], color=INK_2, linewidth=1, linestyle=":",
                       label="everything mean-ablated")
        ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=9, color=INK_2)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=12, color=INK, loc="left")
    axes[0].set_ylabel("Top-1 agreement with the original GRU", fontsize=10, color=INK)
    fig.suptitle("Does the model still work with only the explained components kept?",
                 fontsize=13, color=INK, x=0.01, ha="left")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, ncol=len(labels), loc="upper left", bbox_to_anchor=(0.01, 0.93),
               frameon=False, fontsize=9, labelcolor=INK_2)
    fig.text(0.01, 0.01, "Bar labels: number of components kept; the rest are set to their mean. Splice runs the "
             "ablated gate inside the GRU (errors compound).", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.87))
    path1 = os.path.join(outdir, "fidelity_by_selection.png")
    fig.savefig(path1, dpi=170, facecolor=SURFACE)
    plt.close(fig)

    # 2. F1 make-up of the core sets (fewest components recovering most of the fve)
    bands = [(">=0.8", "F1 >= 0.8"), ("0.5-0.8", "0.5 - 0.8"), ("0.2-0.5", "0.2 - 0.5"), ("<0.2", "< 0.2 or none")]
    core_rows = [(f"{title.split(' gate')[0]}: {FIDELITY_LABELS[c['source']]}", c)
                 for gate, title in FIDELITY_GATES for source in FIDELITY_SOURCES
                 for c in cores if c["gate"] == gate and c["source"] == source]
    width_of = {(r["gate"], r["source"]): r["n_total"] for r in rows}
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(core_rows) + 1.8), facecolor=SURFACE)
    style(ax)
    y = np.arange(len(core_rows))[::-1]
    for yi, (label, c) in zip(y, core_rows):
        left, n = 0.0, c["core"]["n"]
        for key, name in bands:
            share = c["core"][key] / n if n else 0
            ax.barh(yi, share, left=left, height=0.62, color=BAND_COLORS[name], edgecolor=SURFACE, linewidth=2)
            if share >= 0.08:
                ax.text(left + share / 2, yi, f"{share:.0%}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if name in ("F1 >= 0.8", "0.5 - 0.8") else INK)
            left += share
        ax.text(1.01, yi, f"{n} of {width_of[(c['gate'], c['source'])]}", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, [r[0] for r in core_rows], fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of the core set (components needed for 90% of the full set's fve)", fontsize=10, color=INK)
    ax.legend(handles=[matplotlib.patches.Patch(color=BAND_COLORS[n], label=f"held-out {n}") for _, n in bands],
              ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9, labelcolor=INK_2)
    ax.set_title("Are the components that matter the explained ones?", fontsize=12, color=INK, loc="left", pad=28)
    fig.tight_layout()
    path2 = os.path.join(outdir, "core_sets_by_f1.png")
    fig.savefig(path2, dpi=170, facecolor=SURFACE)
    plt.close(fig)

    paths = [path1, path2]
    # 3. drop tests (e.g. the dense candidate features)
    for gate in sorted({d["gate"] for d in drops}):
        tests = [d for d in drops if d["gate"] == gate]
        full = pick(gate, "features", "full")
        fig, ax = plt.subplots(figsize=(9, 0.42 * len(tests) + 1.6), facecolor=SURFACE)
        style(ax)
        y = np.arange(len(tests))[::-1]
        ax.barh(y, [d["top1_agreement"] for d in tests], height=0.62, color=SOURCE_COLORS[0],
                edgecolor=SURFACE, linewidth=2)
        for yi, d in zip(y, tests):
            ax.text(d["top1_agreement"] + 0.01, yi, f"{d['top1_agreement']:.1%}  ({d['n_kept']} kept)",
                    va="center", fontsize=8.5, color=INK_2)
        if full:
            ax.axvline(full[0]["top1_agreement"], color=INK_2, linewidth=1, linestyle=":")
            ax.text(full[0]["top1_agreement"], len(tests) - 0.4, "full transcoder", ha="right", fontsize=8,
                    color=INK_2)
        names = [d["test"].replace("set", "the dense set").replace("f1>=", "F1 ≥ ") for d in tests]
        ax.set_yticks(y, names, fontsize=9, color=INK)
        ax.set_xlim(0, 1.18)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("Top-1 agreement with the original GRU", fontsize=10, color=INK)
        ax.set_title(f"{gate.capitalize()}: what the F1 selection leaves out "
                     f"(dense set = features {', '.join(map(str, tests[0]['set']))})",
                     fontsize=11, color=INK, loc="left")
        fig.tight_layout()
        p = os.path.join(outdir, f"{gate}_drop_tests.png")
        fig.savefig(p, dpi=170, facecolor=SURFACE)
        plt.close(fig)
        paths.append(p)
    return paths


DIVERGING = ["#184f95", "#3987e5", "#f0efec", "#e34948", "#9e2a2a"]   # blue <-> gray <-> red


def probe_plots(path, outdir, clip=1.5):
    """Token tables from circuit/feature_probe.py: activation minus its (L, t) mean, by token."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    os.makedirs(outdir, exist_ok=True)
    tables = json.load(open(path))["tables"]
    tokens = tables["tokens"]
    features = [k for k in tables if k != "tokens"]
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    cmap.set_bad(SURFACE)

    def heat(ax, m, xlabels, ylabels, annotate=0.5, fontsize=6.5):
        m = np.array([[np.nan if v is None else v for v in row] for row in m], dtype=float)
        im = ax.imshow(np.clip(m, -clip, clip), cmap=cmap, vmin=-clip, vmax=clip, aspect="auto")
        for (r, k), v in np.ndenumerate(m):
            if not np.isnan(v) and abs(v) >= annotate:
                ax.text(k, r, f"{v:.1f}", ha="center", va="center", fontsize=fontsize,
                        color="#ffffff" if abs(v) >= 0.9 * clip else INK)
        ax.set_xticks(range(len(xlabels)), xlabels, fontsize=7.5, color=INK_2)
        ax.set_yticks(range(len(ylabels)), ylabels, fontsize=8, color=INK)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0)
        return im

    # 1. one row per feature, one panel per cue
    cues = [("input", "Current input token (input phase)"), ("prev_input", "Previous input token (input phase)"),
            ("target", "Token being output (copy phase)")]
    fig, axes = plt.subplots(len(cues), 1, figsize=(12, 2.2 * len(cues) + 1.4), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.5})
    for ax, (key, title) in zip(axes, cues):
        im = heat(ax, [tables[f][key] for f in features], tokens, [f"feature {f}" for f in features])
        ax.set_title(title, fontsize=11, color=INK, loc="left")
    cb = fig.colorbar(im, ax=axes, shrink=0.6, extend="both", pad=0.01)
    cb.set_label("activation above / below what timing predicts", fontsize=9, color=INK_2)
    cb.outline.set_visible(False)
    fig.suptitle("Dense candidate features: which tokens raise or lower their activation",
                 fontsize=13, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Activation minus its mean at the same (length, step). Numbers shown where |value| >= 0.5; "
             f"colour clipped at ±{clip}.", fontsize=8, color=INK_2)
    path1 = os.path.join(outdir, "dense_feature_tokens.png")
    fig.savefig(path1, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # 2. the strongest pairwise table per feature
    paths = [path1]
    for f in features:
        t = tables[f]
        spread = lambda key: np.nanstd([v for v in t[key] if v is not None])
        if spread("input") + spread("prev_input") >= spread("target"):
            key, ylab, xlab = "prev_x_input", "previous input token", "current input token (input phase)"
        else:
            key, ylab, xlab = "target_x_next", "token being output", "next token to output (copy phase)"
        fig, ax = plt.subplots(figsize=(9.5, 8.2), facecolor=SURFACE)
        im = heat(ax, t[key], tokens, tokens, annotate=1.0, fontsize=5.5)
        ax.set_xlabel(xlab, fontsize=10, color=INK)
        ax.set_ylabel(ylab, fontsize=10, color=INK)
        ax.set_title(f"Feature {f}: activation above / below timing, by token pair", fontsize=12, color=INK, loc="left")
        cb = fig.colorbar(im, ax=ax, shrink=0.7, extend="both", pad=0.01)
        cb.outline.set_visible(False)
        p = os.path.join(outdir, f"dense_feature_{f}_pairs.png")
        fig.savefig(p, dpi=170, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("specificity"); s.add_argument("results"); s.add_argument("--out", default=None)
    c = sub.add_parser("coverage", help="timestep coverage of features grouped by explanation F1")
    c.add_argument("store"); c.add_argument("results"); c.add_argument("--gates", nargs="*")
    c.add_argument("--subset_store", help="only count sequences of this store (e.g. a unit store on a subsample)")
    c.add_argument("--out", help="write the numbers as JSON")
    f = sub.add_parser("fidelity", help="figures for circuit/explained_fidelity.py output")
    f.add_argument("fidelity_json"); f.add_argument("--out", help="output directory (default: next to the JSON)")
    pr = sub.add_parser("probe", help="token tables from circuit/feature_probe.py output")
    pr.add_argument("probe_json"); pr.add_argument("--out", help="output directory (default: next to the JSON)")
    args = ap.parse_args()
    if args.cmd == "probe":
        for p in probe_plots(args.probe_json, args.out or os.path.join(os.path.dirname(args.probe_json), "plots")):
            print(p)
        return
    if args.cmd == "fidelity":
        for p in fidelity_plots(args.fidelity_json, args.out or os.path.join(os.path.dirname(args.fidelity_json), "plots")):
            print(p)
        return
    if args.cmd == "coverage":
        cov = coverage(args.store, args.results, args.gates, args.subset_store)
        print_coverage(args.results, cov)
        if args.out:
            json.dump(cov, open(args.out, "w"), indent=1)
        return
    if args.cmd == "specificity":
        out = args.out or os.path.join(args.results, "plots", "specificity_by_activation.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        specificity(args.results, out)


if __name__ == "__main__":
    main()
