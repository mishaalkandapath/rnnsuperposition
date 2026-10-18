"""Figures for the activation-based interpretations (circuit/copy_task/autointerp.py results).

    python -m circuit.copy_task.autointerp_plots specificity RESULTS_DIR [--out FILE]
    python -m circuit.copy_task.autointerp_plots coverage STORE RESULTS_DIR [--out FILE]
    python -m circuit.copy_task.autointerp_plots coverage_plots FEATURES_COVERAGE_JSON [--units "LABEL=JSON" ...] [--out DIR]
    python -m circuit.copy_task.autointerp_plots fidelity FIDELITY_JSON [--out DIR]
    python -m circuit.copy_task.autointerp_plots probe PROBE_JSON [--out DIR]

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
import textwrap
from collections import defaultdict

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
    from circuit.copy_task.autointerp import Interp
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
    """Figures for circuit/copy_task/explained_fidelity.py output (top-1 agreement with the original GRU)."""
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
    """Token tables from circuit/copy_task/feature_probe.py: activation minus its (L, t) mean, by token."""
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


def circuit_plots(path, outdir):
    """Figures for circuit/copy_task/copy_circuits.py: real-GRU swap tests and graph attributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import PercentFormatter
    os.makedirs(outdir, exist_ok=True)
    diverging = LinearSegmentedColormap.from_list("div", DIVERGING[::-1])     # red = negative
    sequential = LinearSegmentedColormap.from_list("seq", ["#f0efec"] + list(BAND_COLORS.values())[::-1])
    for cmap in (diverging, sequential):
        cmap.set_bad(SURFACE)
    pct = lambda v: f"{v:.0%}"

    def heat(ax, m, cmap, vmin, vmax, fmt, annotate, notes=None):
        """notes: optional matrix of small second-line labels (None entries skipped)."""
        m = np.array(m, dtype=float)
        im = ax.imshow(np.clip(m, vmin, vmax), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        dark = lambda v: (v - vmin) / (vmax - vmin) > 0.8 or (vmin < 0 and (v - vmin) / (vmax - vmin) < 0.2)
        for (r, k), v in np.ndenumerate(m):
            if np.isnan(v):
                continue
            note = notes[r][k] if notes is not None else None
            if abs(v) >= annotate:
                ax.text(k, r - (0.12 if note else 0), fmt(v), ha="center", va="center", fontsize=8,
                        color="#ffffff" if dark(v) else INK)
            if note:
                ax.text(k, r + 0.25, note, ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if dark(v) else INK_2)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0)
        return im

    def grid(rows, xlabels, ylabels, title, foot, name, panel_w, panel_h):
        """rows: [(colourbar label, cmap, vmin, vmax, extend, [(panel title, matrix, fmt, annotate, notes)])].
        Row labels only on the first panel of each row; one colourbar per row."""
        n_rows, n_cols = len(rows), max(len(r[-1]) for r in rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.4 + n_cols * panel_w, 1.2 + n_rows * panel_h),
                                 facecolor=SURFACE, layout="constrained", squeeze=False)
        for r, (cb_label, cmap, vmin, vmax, extend, panels) in enumerate(rows):
            for c, ax in enumerate(axes[r]):
                if c >= len(panels):
                    ax.set_visible(False)
                    continue
                ptitle, m, fmt, annotate, notes = panels[c]
                im = heat(ax, m, cmap, vmin, vmax, fmt, annotate, notes)
                ax.set_title(ptitle, fontsize=10, color=INK, loc="left")
                ax.set_xticks(range(len(xlabels)), xlabels, fontsize=8, color=INK_2)
                ax.set_yticks(range(len(ylabels)), ylabels if c == 0 else [], fontsize=8, color=INK)
            cb = fig.colorbar(im, ax=list(axes[r]), shrink=0.9, extend=extend, aspect=15)
            if panels[0][2] is pct:
                cb.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            cb.set_label(cb_label, fontsize=8.5, color=INK_2)
            cb.ax.tick_params(labelsize=7.5, colors=INK_2)
            cb.outline.set_visible(False)
        fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
        fig.supxlabel(textwrap.fill(foot, int(fig.get_figwidth() * 15)), fontsize=7.5, color=INK_2, x=0.01, ha="left")
        p = os.path.join(outdir, name)
        fig.savefig(p, dpi=170, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        return p

    paths = []
    for Ld in json.load(open(path))["lengths"]:
        L, steps = Ld["L"], Ld["steps"]
        outs = [f"output {j}" for j in range(L)]
        swapped = [f"swap pos {i}" for i in range(L)]
        sw = Ld["swaps"]
        modes = [("full", "Feedback changed to match"), ("input_only", "Feedback left unchanged")]
        acc_notes = {mode: [[f"acc {a:.0%}" if a < 0.95 else None for a in row] for row in sw[mode]["accuracy"]]
                     for mode, _ in modes}
        swap_title = f"Length {L}: which outputs change when one input token is swapped (real GRU)"
        swap_foot = (f"Mean over {sw['n_sequences']} sequences. Rows: the input position swapped to another token; "
                     "columns: outputs. Accuracy against the swapped sequence shown where below 95%.")

        # 1. swap tests, raw logit change
        paths.append(grid([("change in logit of the original token", diverging, -25, 25, "both",
                            [(t, sw[m]["dlogit"], "{:.1f}".format, 0.5, acc_notes[m]) for m, t in modes])],
                          outs, swapped, swap_title, swap_foot, f"swap_L{L}.png", 0.9 + 0.75 * L, 0.5 + 0.55 * L))

        # 1b. swap tests, normalised: fraction of the logit lost, and P(original token)
        paths.append(grid([("change as a fraction of the original logit", diverging, -1, 1, "both",
                            [(t, sw[m]["dlogit_frac"], pct, 0.05, None) for m, t in modes]),
                           ("probability of the original token", sequential, 0, 1, "neither",
                            [(t, sw[m]["p_orig"], pct, 0.0, acc_notes[m]) for m, t in modes])],
                          outs, swapped, swap_title + ", normalised", swap_foot + " Top: logit change divided "
                          "by the unswapped logit, per sequence. Bottom: softmax probability of the original token.",
                          f"swap_norm_L{L}.png", 0.9 + 0.75 * L, 0.5 + 0.55 * L))

        # 2. identity attribution by step, for the three targets (raw, then as row shares)
        kinds = [("correct", "Correct token orig[j]"), ("popped", "Token just emitted orig[j-1]"),
                 ("next", "Token still to come orig[j+1]")]
        tables = {}
        for kind, _ in kinds:
            m = np.full((L, len(steps)), np.nan)
            for o in Ld["outputs"]:
                if o["kind"] == kind:
                    m[o["j"]] = o["sources_by_step"]["identity"]
            tables[kind] = m
        id_foot = (f"Mean over {Ld['n_sequences']} sequences with distinct tokens. Identity = input minus the mean "
                   "token, traced through all paths with gates frozen. Off-diagonal credit is not dependence "
                   "(see the swap test).")
        id_title = f"Length {L}: which step's input token each output logit is attributed to (attribution graphs)"
        panel_w, panel_h = 0.6 + 0.55 * len(steps), 0.5 + 0.5 * L
        paths.append(grid([("attribution to input token identity", diverging, -40, 40, "both",
                            [(t, tables[k], "{:.0f}".format, 2, None) for k, t in kinds])],
                          steps, outs, id_title, id_foot, f"identity_attribution_L{L}.png", panel_w, panel_h))
        shares = {k: m / np.nansum(np.abs(m), axis=1, keepdims=True) for k, m in tables.items()}
        paths.append(grid([("share of the row's identity attribution", diverging, -0.5, 0.5, "both",
                            [(t, shares[k], pct, 0.02, None) for k, t in kinds])],
                          steps, outs, id_title + ", as row shares", id_foot + " Each row divided by the sum of "
                          "its absolute values, so |shares| add to 100%; sign kept.",
                          f"identity_share_L{L}.png", panel_w, panel_h))

        # 3. contrastive graphs: swapped minus original, by step, one panel per swapped position
        if "contrast" in Ld:
            co = Ld["contrast"]
            cell = {(c["i"], c["j"]): c for c in co["cells"]}
            raw = [[cell[(i, j)]["total_by_step"] for j in range(L)] for i in range(L)]
            scale = [abs(cell[(i, i)]["dvalue"]) for i in range(L)]
            ylabels = [f"output {j}" for j in range(L)]
            co_title = f"Length {L}: contrastive graphs, change in each output's logit when one token is swapped, by step"
            co_foot = (f"Mean over {co['n_sequences']} sequences with distinct tokens. Each panel swaps one input "
                       "position (feedback changed to match); cells are swapped-run minus original-run attribution "
                       "of the original token's logit, by the step it comes from (all sources). Rows sum exactly to "
                       "the real logit change; credit shared by both runs cancels.")
            paths.append(grid([("change in logit attributed to the step", diverging, -15, 15, "both",
                                [(f"Swap position {i} (total {cell[(i, i)]['dvalue']:.1f} on output {i})",
                                  raw[i], "{:.1f}".format, 0.5, None) for i in range(L)])],
                              steps, ylabels, co_title, co_foot, f"contrast_L{L}.png", panel_w, panel_h))
            paths.append(grid([("share of the swap's effect on its own output", diverging, -0.5, 0.5, "both",
                                [(f"Swap position {i}", np.array(raw[i]) / scale[i], pct, 0.03, None)
                                 for i in range(L)])],
                              steps, ylabels, co_title + ", normalised", co_foot + " Normalised by the size of "
                              "the swap's effect on its own output, so the diagonal row sums to -100%.",
                              f"contrast_share_L{L}.png", panel_w, panel_h))
    return paths


def feature_circuit_plots(path, outdir, gates_path=None, circuits_path=None, min_flow=1.5):
    """Circuit diagrams from circuit/copy_task/feature_circuits.py: one per slot, steps left to right.

    Boxes: candidate features at each step (top carriers with their magnitude
    code: tokens in the slot that raise / lower them), with how many features
    carry 80% of the flow and the intervention results. Arrows: contrastive
    flow between steps (width ~ flow). Grey tags: the update / reset gate
    features with the largest share at that step (from gate_mechanism.py).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    os.makedirs(outdir, exist_ok=True)
    d = json.load(open(path))
    L, steps = d["L"], d["steps"]
    gates = None
    if gates_path:
        gates = next((g for g in json.load(open(gates_path))["lengths"] if g["L"] == L), None)
    pops = {}
    if circuits_path:
        Ld = next((c for c in json.load(open(circuits_path))["lengths"] if c["L"] == L), None)
        for o in (Ld or {}).get("outputs", []):
            if o["kind"] == "popped":                                    # output j's popped token is slot j - 1
                pops[o["j"] - 1] = o["sources_by_step"]["identity"][L + o["j"]]
    steer = {(s["slot"], s["where"]): s for s in d.get("steer", [])
             if s["features"] == "top 20 carriers" and s["control"] is None}
    own = {(a["slot"], a["step"]): a for a in d.get("own_ablations", []) if a["k"] == 5}
    stage_color = {"write": SOURCE_COLORS[0], "relay": SOURCE_COLORS[2], "read-out": SOURCE_COLORS[1]}
    W, H, DX = 2.25, 2.55, 2.85                                          # box size, column spacing

    def code(t):
        up = " ".join(tok for tok, v in t.get("raised_by", [])[:3] if v > 0)
        down = " ".join(tok for tok, v in t.get("lowered_by", [])[:2] if v < 0)
        return f"+{up}" + (f"  -{down}" if down else "")

    paths = []
    for fl in d["flows"]:
        i = fl["slot"]
        cols = [st for st in steps[i:L + i + 1]]
        x_of = {st: k * DX for k, st in enumerate(cols)}
        fig, ax = plt.subplots(figsize=(1.2 + DX * (len(cols) + 0.9), 7.4), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        y_box, y_in, y_out = 0.0, -2.35, 2.75
        for k, st in enumerate(cols):
            stage = "write" if k == 0 else "read-out" if k == len(cols) - 1 else "relay"
            nd = fl["nodes_by_step"].get(st, {"top": [], "total": 0, "k80_per_sequence": 0})
            x0 = x_of[st] - W / 2
            ax.add_patch(FancyBboxPatch((x0, y_box - H / 2), W, H, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc=SURFACE, ec=stage_color[stage], lw=1.8))
            ax.text(x_of[st], y_box + H / 2 - 0.2, f"{stage} @ {st}", ha="center", va="center",
                    fontsize=9, color=stage_color[stage], fontweight="bold")
            lines = [f"{t['feature']:>3}  {code(t)}" for t in nd["top"][:5]]
            ax.text(x0 + 0.1, y_box + H / 2 - 0.45, "\n".join(lines), ha="left", va="top", fontsize=7.2,
                    color=INK, family="monospace", linespacing=1.35)
            info = f"flow {nd['total']:+.1f} | 80% in {nd['k80_per_sequence']:.0f} feats"
            s, a = steer.get((i, st)), own.get((i, st))
            test = []
            if s:
                test.append(f"steer 20: {s['flip_to_new']:.0%} flip")
            if a:
                test.append(f"ablate 5: acc {a['accuracy'][i]:.0%}")
            ax.text(x_of[st], y_box - H / 2 + 0.42, info, ha="center", va="center", fontsize=6.8, color=INK_2)
            ax.text(x_of[st], y_box - H / 2 + 0.17, " | ".join(test), ha="center", va="center", fontsize=6.8, color=INK_2)
            if gates:                                                    # gate tags under the box
                gsteps = {g["step"]: g for g in gates["slots"][i]["steps"]}
                g = gsteps.get(st, {})
                tags = []
                for name, gate, verb in (("erase_by", "upd", "erase"), ("read_by", "rst", "read")):
                    top = (g.get(name) or {}).get("top") or []
                    if top:
                        tags.append(f"{gate} {top[0]['feature']} ({verb} {top[0]['share']:+.2f})")
                ax.text(x_of[st], y_box - H / 2 - 0.25, "\n".join(tags), ha="center", va="top", fontsize=6.6,
                        color=INK_2, style="italic")

        def arrow(p, q, flow, rad=0.0, label=True):
            ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=0.6 + abs(flow) / 3.5,
                                         color=INK_2 if flow > 0 else "#e34948", alpha=0.85,
                                         connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
            if label:
                box = dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none")
                if p[0] == q[0]:                                         # vertical: label beside the arrow
                    ax.text(p[0] + 0.18, (p[1] + q[1]) / 2, f"{flow:+.1f}", ha="left", va="center", fontsize=7.5,
                            color=INK, bbox=box)
                else:                                                    # straight: above; arc: at its apex
                    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + (abs(rad) * (q[0] - p[0]) * 0.5 if rad else 0.18)
                    ax.text(mx, my, f"{flow:+.1f}", ha="center", va="center" if rad else "bottom", fontsize=7.5,
                            color=INK, bbox=box, zorder=5)

        for e in fl["step_edges"]:
            if abs(e["flow"]) < min_flow or e["src"] not in x_of:
                continue
            if e["dst"] == "target":
                continue
            if e["dst"] not in x_of:
                continue
            gap = cols.index(e["dst"]) - cols.index(e["src"])
            if gap == 1:
                arrow((x_of[e["src"]] + W / 2, y_box + 0.35), (x_of[e["dst"]] - W / 2, y_box + 0.35), e["flow"])
            elif gap > 1:                                                # skip via the carry: arc above
                arrow((x_of[e["src"]] + 0.4, y_box + H / 2), (x_of[e["dst"]] - 0.4, y_box + H / 2), e["flow"], rad=-0.25)
        # input -> write, read-out -> logit
        src_in = next((s["flow"] for s in fl["sources"] if s["source"] == "input" and s["step"] == cols[0]), 0)
        ax.text(x_of[cols[0]], y_in, f"input token\n(slot {i})", ha="center", va="center", fontsize=8.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", fc="#f0efec", ec=INK_2, lw=0.8))
        arrow((x_of[cols[0]], y_in + 0.4), (x_of[cols[0]], y_box - H / 2 - 0.02), src_in)
        to_target = sum(e["flow"] for e in fl["step_edges"] if e["dst"] == "target")
        ax.text(x_of[cols[-1]], y_out, f"logit of the token\nat output {i}", ha="center", va="center", fontsize=8.5,
                color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="#f0efec", ec=INK_2, lw=0.8))
        arrow((x_of[cols[-1]], y_box + H / 2 + 0.02), (x_of[cols[-1]], y_out - 0.4), to_target)
        if i in pops and i + 1 < L:                                      # the pop one step later
            xp = x_of[cols[-1]] + DX * 0.95
            ax.text(xp, y_box, f"pop @ {steps[L + i + 1]}\nfed-back copy\npushes this token's\nlogit {pops[i]:+.0f}",
                    ha="center", va="center", fontsize=7.8, color="#9e2a2a",
                    bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec="#e34948", lw=1.0, ls="--"))
        ax.set_xlim(-W / 2 - 0.4, x_of[cols[-1]] + DX * 1.55)
        ax.set_ylim(y_in - 0.9, y_out + 0.8)
        ax.axis("off")
        ax.set_title(f"Length {L}, slot {i}: feature circuit that stores the token and outputs it at step {steps[L + i]}",
                     fontsize=12, color=INK, loc="left")
        fig.text(0.01, 0.01, textwrap.fill(
            f"Contrastive flow (original minus token-swapped run, DeepLIFT rescale on the frozen-gate graph), mean over "
            f"{d['args']['n_flow']} sequences; total change in the logit {fl['dtarget']:+.1f}. Feature lines: id, then the tokens "
            "in this slot that raise (+) / lower (-) its activation at that step. Arrows: flow between steps "
            f"(|flow| >= {min_flow}; arcs skip steps through the update-gate carry). steer 20: top-20 carriers' magnitudes "
            "moved as if the slot held another token, share of outputs that switch to it. ablate 5: each sequence's own top-5 "
            "carriers mean-ablated, accuracy of this output. Grey: gate feature with the largest share of erasing (update) / "
            "reading (reset) the trace.", 190), fontsize=7, color=INK_2, va="bottom")
        p = os.path.join(outdir, f"feature_circuit_L{L}_slot{i}.png")
        fig.savefig(p, dpi=170, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def token_circuit_plots(path, outdir, min_flow=1.0):
    """One diagram per example from circuit/copy_task/token_circuits.py: a real sequence and slot,
    the features carrying that token at each step (with how much each moved and a short
    label), flow between steps, this sequence's gate features, interventions and the pop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    os.makedirs(outdir, exist_ok=True)
    d = json.load(open(path))
    L = d["L"]
    stage_color = {"write": SOURCE_COLORS[0], "relay": SOURCE_COLORS[2], "read-out": SOURCE_COLORS[1]}
    W, H, DX = 3.0, 4.4, 3.55
    def wrap(s, n):
        lines = textwrap.wrap(s, n)
        return lines if len(lines) <= 2 else [lines[0], lines[1].rstrip(" .") + " ..."]
    paths = []
    for ex in d["examples"]:
        i, tokn = ex["slot"], ex["token"]
        cols = [s["step"] for s in ex["steps"]]
        x_of = {st: k * DX for k, st in enumerate(cols)}
        fig, ax = plt.subplots(figsize=(1.5 + DX * (len(cols) + 1.0), 9.2), facecolor=SURFACE)
        y_box, y_in, y_out = 0.0, -4.1, 3.7

        def arrow(p, q, flow, rad=0.0):
            ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=0.6 + abs(flow) / 3.5,
                                         color=INK_2 if flow > 0 else "#e34948", alpha=0.85,
                                         connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
            box = dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none")
            if p[0] == q[0]:
                ax.text(p[0] + 0.18, (p[1] + q[1]) / 2, f"{flow:+.1f}", ha="left", va="center", fontsize=8, color=INK, bbox=box)
            else:
                mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + (abs(rad) * (q[0] - p[0]) * 0.5 if rad else 0.2)
                ax.text(mx, my, f"{flow:+.1f}", ha="center", va="center" if rad else "bottom", fontsize=8,
                        color=INK, bbox=box, zorder=5)

        for k, s in enumerate(ex["steps"]):
            st = s["step"]
            stage = "write" if k == 0 else "read-out" if k == len(cols) - 1 else "relay"
            x0 = x_of[st] - W / 2
            ax.add_patch(FancyBboxPatch((x0, y_box - H / 2), W, H, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc=SURFACE, ec=stage_color[stage], lw=1.8))
            ax.text(x_of[st], y_box + H / 2 - 0.22, f"{stage} @ {st}  (flow {s['flow_total']:+.1f})", ha="center",
                    va="center", fontsize=9, color=stage_color[stage], fontweight="bold")
            yy = y_box + H / 2 - 0.55
            for c in s["top"]:
                if c["flow"] <= 0:
                    continue
                ax.text(x0 + 0.1, yy, f"{c['feature']} ({c['kind']}): {tokn} moves it {c['da']:+.2f}",
                        ha="left", va="top", fontsize=7.4, color=INK, fontweight="bold")
                yy -= 0.24
                for line in wrap(c["label"], 46):
                    ax.text(x0 + 0.18, yy, line, ha="left", va="top", fontsize=6.8, color=INK_2)
                    yy -= 0.21
                yy -= 0.08
            ax.text(x_of[st], y_box - H / 2 + 0.36,
                    f"remove top 5: P({tokn}) = {s['ablate5_p_correct']:.2f}\n"
                    f"steer top 20 to '{ex['swapped_to']}': outputs '{s['steer20_pred']}'",
                    ha="center", va="center", fontsize=7, color=INK_2)
            tags = []
            for name, gate, verb in (("erase_by", "upd", "erase"), ("read_by", "rst", "read")):
                g = s["gates"].get(name)
                if g:
                    tags.append(f"{gate} {g['feature']} ({verb} {g['share']:+.2f}): {short_text(g['label'], 34)}")
            ax.text(x_of[st], y_box - H / 2 - 0.2, "\n".join(tags), ha="center", va="top", fontsize=6.6,
                    color=INK_2, style="italic")
        for e in ex["step_edges"]:
            if abs(e["flow"]) < min_flow or e["src"] not in x_of or e["dst"] not in x_of:
                continue
            gap = cols.index(e["dst"]) - cols.index(e["src"])
            if gap == 1:
                arrow((x_of[e["src"]] + W / 2, y_box + 0.5), (x_of[e["dst"]] - W / 2, y_box + 0.5), e["flow"])
            elif gap > 1:
                arrow((x_of[e["src"]] + 0.5, y_box + H / 2), (x_of[e["dst"]] - 0.5, y_box + H / 2), e["flow"], rad=-0.22)
        ax.text(x_of[cols[0]], y_in, f"input '{tokn}'\n(slot {i})", ha="center", va="center", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", fc="#f0efec", ec=INK_2, lw=0.8))
        arrow((x_of[cols[0]], y_in + 0.45), (x_of[cols[0]], y_box - H / 2 - 0.02), ex["input_flow"].get(cols[0], 0))
        to_target = sum(e["flow"] for e in ex["step_edges"] if e["dst"] == "target")
        ax.text(x_of[cols[-1]], y_out, f"logit of '{tokn}'\nat output {i}", ha="center", va="center", fontsize=9,
                color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="#f0efec", ec=INK_2, lw=0.8))
        arrow((x_of[cols[-1]], y_box + H / 2 + 0.02), (x_of[cols[-1]], y_out - 0.45), to_target)
        if ex["pop"]:
            p = ex["pop"]
            ax.text(x_of[cols[-1]] + DX * 0.95, y_box,
                    textwrap.fill(f"pop at the next step: '{tokn}' is fed back. Its logit at output {i + 1} is "
                                  f"{p['logit_true_feedback']:+.1f}. With a different token fed back (nothing to pop) "
                                  f"it would be {p['logit_mismatched_feedback']:+.1f} and the model outputs "
                                  f"'{p['pred_mismatched']}'.", 26),
                    ha="center", va="center", fontsize=7.8, color="#9e2a2a",
                    bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec="#e34948", lw=1.0, ls="--"))
        ax.set_xlim(-W / 2 - 0.4, x_of[cols[-1]] + DX * 1.6)
        ax.set_ylim(y_in - 0.8, y_out + 0.8)
        ax.axis("off")
        ax.set_title(f"Sequence  {' '.join(ex['sequence'])}   (L = {L}): how '{tokn}' in slot {i} is stored and output",
                     fontsize=12, color=INK, loc="left")
        fig.text(0.01, 0.005, textwrap.fill(
            f"Contrastive flow for this sequence against the same sequence with '{tokn}' replaced by '{ex['swapped_to']}' "
            f"(feedback changed to match); total change in the logit {ex['dtarget']:+.1f}. Each step lists the candidate "
            f"features carrying the most flow, how much '{tokn}' moved each one (activation, original minus swapped), "
            "and its label (magnitude label for dense features, firing label otherwise). Arrows: flow between steps; "
            "arcs skip steps through the update-gate carry. Grey italics: this sequence's update / reset gate features "
            "with the largest share of erasing / reading the token's trace. Interventions are on this sequence only.", 210),
            fontsize=7, color=INK_2, va="bottom")
        p = os.path.join(outdir, f"token_circuit_L{L}_slot{i}_{tokn}.png")
        fig.savefig(p, dpi=170, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def rotation_wheel_plots(path, outdir, n_labels=3):
    """Angle wheels from circuit/copy_task/rotation_edges.py --phases: one polar panel per number of steps
    since a token was written. Arrows: carriers at the angle they write the token (length = weight),
    keepers vs flippers; black: the token's trace. Pooled over slots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    d = json.load(open(path))
    L = d["L"]
    colors = {"keeper": SOURCE_COLORS[0], "flipper": SOURCE_COLORS[1]}
    ks = sorted({int(k) for f in d["features"] for k in f["angle_by_steps_since_write"]})
    sign = d.get("sign_test", {}).get("by_group", {})
    fig, axes = plt.subplots(1, len(ks), figsize=(3.8 * len(ks), 4.6), subplot_kw={"projection": "polar"},
                             facecolor=SURFACE, layout="constrained")
    fig.get_layout_engine().set(wspace=0.12)
    for ax, k in zip(axes, ks):
        ax.set_facecolor(SURFACE)
        items = []
        for f in d["features"]:
            v = f["angle_by_steps_since_write"].get(str(k))
            if v and v["weight"] > 0:
                items.append((v["weight"], f["feature"], np.deg2rad(v["angle"]), f["group"]))
        wmax = max(w for w, *_ in items) if items else 1
        for w, feat, ang, grp in sorted(items):
            ax.annotate("", xy=(ang, w / wmax), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=colors[grp], lw=0.8 + 1.6 * w / wmax, alpha=0.75))
        top = sorted(items, reverse=True)[:n_labels]
        ax.text(np.deg2rad(-60), 1.45, "largest: " + ", ".join(str(f) for _, f, _, _ in top), ha="center", va="center",
                fontsize=7.5, color=INK_2)
        tr = [np.deg2rad(s["trace_angle"]) for s in d["steps"] if s["steps_since_write"] == k]
        if tr:
            m = np.arctan2(np.mean(np.sin(tr)), np.mean(np.cos(tr)))
            ax.annotate("", xy=(m, 1.0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.6))
        ax.set_ylim(0, 1.25)
        ax.set_yticks([])
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]), ["0°", "90°", "180°", ""], fontsize=7.5, color=INK_2)
        ax.grid(color=GRID, lw=0.6)
        ax.spines["polar"].set_color(GRID)
        title = "write step" if k == 0 else f"{k} step{'s' if k > 1 else ''} after the write"
        if k == L:
            title += "\n(output step)"
        sub = ""
        if str(k) in sign.get("flipper", {}):
            sub = f"\nflippers: input vs memory r = {sign['flipper'][str(k)]['median']:+.2f}"
        ax.set_title(title + sub, fontsize=9.5, color=INK, pad=14)
    handles = [plt.Line2D([], [], color=colors["keeper"], lw=2, label="keepers (write near the original orientation)"),
               plt.Line2D([], [], color=colors["flipper"], lw=2, label="flippers (reverse their token code)"),
               plt.Line2D([], [], color=INK, lw=2.6, label="the token's trace")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Length {L}: how the candidate features hold a stored token as its code turns ~180° (pooled over slots)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, -0.16, textwrap.fill(
        "Frame per token: 0° = the token's code as written, 90°/180° towards its final read direction. Arrows: each "
        "candidate feature's write of the token (original minus token-swapped run) at that step, length = mean weight in "
        "this plane; colour = group by its mean angle. Black: the trace in h. r: median correlation, over flippers, between "
        "a feature's input code (encoder weights for the token) and its memory code (how the stored token moves it).", 190),
        fontsize=7.5, color=INK_2)
    p = os.path.join(outdir, f"rotation_wheel_L{L}.png")
    fig.savefig(p, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return [p]


def set_controls_plot(path, out):
    """Figure for explained_fidelity.py --set_controls: each F1 selection with and without the fixed set, against
    random sets of the same size (dots), with 95% bootstrap intervals and empirical p-values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.load(open(path))
    sel = d["selections"]
    taus = sorted({r["tau"] for r in sel}, reverse=True)
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    w = 0.36
    colors = {False: SOURCE_COLORS[0], True: SOURCE_COLORS[2]}
    jitter = np.random.default_rng(0)
    for i, tau in enumerate(taus):
        for j, with_set in enumerate((False, True)):
            r = next(x for x in sel if x["tau"] == tau and x["with_set"] == with_set)
            x = i + (j - 0.5) * w
            ax.bar(x, 100 * r["top1"], width=w * 0.9, color=colors[with_set])
            lo, hi = 100 * r["ci"][0], 100 * r["ci"][1]
            ax.errorbar(x, 100 * r["top1"], yerr=[[100 * r["top1"] - lo], [hi - 100 * r["top1"]]], fmt="none",
                        ecolor=INK, lw=1, capsize=3)
            key = "random_with_set" if with_set else "random_any"
            pkey = "p_vs_random_with_set" if with_set else "p_vs_random_any"
            vals = 100 * np.array(r[key])
            ax.scatter(x + jitter.uniform(-w * 0.25, w * 0.25, len(vals)), vals, s=12, facecolor=SURFACE,
                       edgecolor=INK, lw=0.8, zorder=3)
            n_below = int((np.array(r[key]) < r["top1"]).sum())
            ax.text(x, 103, f"{r['n_kept']} kept\n{n_below}/{len(vals)} random below", ha="center", va="bottom",
                    fontsize=7, color=INK_2)
    ax.axhline(100 * d["full"]["top1"], color=INK_2, lw=0.8, ls=":")
    ax.text(len(taus) - 0.5, 100 * d["full"]["top1"] - 1, "full transcoder", ha="right", va="top", fontsize=7.5, color=INK_2)
    ax.set_xticks(range(len(taus)), [f"F1 ≥ {t}" for t in taus], fontsize=9)
    ax.set_ylim(0, 118)
    ax.set_ylabel("top-1 agreement with the original GRU (%)", fontsize=9, color=INK_2)
    handles = [matplotlib.patches.Patch(color=colors[False], label="features with F1 ≥ threshold"),
               matplotlib.patches.Patch(color=colors[True], label=f"… plus the dense set {', '.join(map(str, d['set']))}"),
               plt.Line2D([], [], marker="o", ls="none", markerfacecolor=SURFACE, markeredgecolor=INK,
                          label="random sets of the same size (for the + set bars: always including the set)")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=1)
    fig.suptitle("Candidate gate: keeping the explained features, with and without the 5 dense ones, against chance",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.text(0.01, -0.2, textwrap.fill(
        "Candidate transcoder spliced into the GRU; features not kept are set to their mean. 20,000 held-out sequences, "
        "lengths 3–9. Error bars: 95% bootstrap interval over sequences. Dots: 20 random sets of the same size, drawn "
        "from all features that fire; for the '+ set' bars they always include the dense set. 'Random below' = random "
        "sets scoring lower than the bar (p for 'the selection beats chance' = (1 + random at or above) / 21). "
        "explained_fidelity.py --set_controls.", 170), fontsize=7.5, color=INK_2)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def conditional_plots(d, outdir, n_labels=8):
    """Figures for circuit/copy_task/conditional_analyses.py: effects averaged over all sequences vs over only the
    sequences a feature takes part in. File names end in _conditional so no earlier figure is replaced."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    J = lambda name: json.load(open(os.path.join(d, name)))
    paths = []
    dense_c, sparse_c = SOURCE_COLORS[0], SOURCE_COLORS[1]

    def style(ax, title, xlabel, ylabel):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK_2)
        ax.set_ylabel(ylabel, fontsize=8.5, color=INK_2)

    def save(fig, name, title, foot):
        fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
        fig.supxlabel(textwrap.fill(foot, int(fig.get_figwidth() * 15)), fontsize=7.5, color=INK_2, x=0.01, ha="left")
        p = os.path.join(outdir, name)
        fig.savefig(p, dpi=170, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    def scatter(ax, x, y, rate, labels_by):
        x, y, rate = map(np.asarray, (x, y, rate))
        dense = rate >= 0.9
        ax.scatter(x[dense], y[dense], s=16, color=dense_c, alpha=0.75, lw=0, label="dense (active in ≥ 90% of sequences)")
        ax.scatter(x[~dense], y[~dense], s=16, color=sparse_c, alpha=0.75, lw=0, label="sparse")
        hi = max(float(x.max()), float(y.max())) * 1.05
        ax.plot([0, hi], [0, hi], color=INK_2, lw=0.8, ls="--")
        for k in np.argsort(-labels_by)[:n_labels]:
            ax.annotate(str(k), (x[k], y[k]), xytext=(3, 3), textcoords="offset points", fontsize=7, color=INK)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    if os.path.exists(os.path.join(d, "conditional_importance.json")):
        imp = J("conditional_importance.json")
        Ls = [k for k in imp if k.startswith("L")]
        fig, axes = plt.subplots(2, len(Ls), figsize=(5.4 * len(Ls), 9), facecolor=SURFACE, layout="constrained",
                                 squeeze=False)
        for j, L in enumerate(Ls):
            r = imp[L]
            o = 100 * np.array(r["overall_drop"])
            a = 100 * np.array(r["active_drop"])
            w = 100 * np.array(r["drop_by_output_token"]).max(1)
            scatter(axes[0, j], o, a, r["seq_rate"], a)
            style(axes[0, j], f"length {L[1:]}: on the sequences where it is active", "accuracy drop over all sequences (%)",
                  "accuracy drop on sequences where it is active (%)")
            scatter(axes[1, j], o, w, r["seq_rate"], w)
            style(axes[1, j], f"length {L[1:]}: on its most affected output token", "accuracy drop over all sequences (%)",
                  "accuracy drop on outputs of its worst token (%)")
        save(fig, "feature_importance_conditional.png",
             "Single-feature ablation, measured where the feature matters instead of averaged over every sequence",
             "Real GRU, each candidate feature mean-ablated alone (per-step mean, transcoder error kept), 2000 sequences per "
             "length. x: accuracy drop averaged over all sequences (the earlier ranking). Top: the drop on only the sequences "
             "where the feature is active at some step (a dense feature is active everywhere, so it stays on the dashed y = x "
             "line). Bottom: the drop on the outputs of the one token it hurts most (treats dense and sparse features alike). "
             "Labelled: the largest conditional drops. conditional_analyses.py --sections importance.")

    if os.path.exists(os.path.join(d, "conditional_pop.json")):
        pop = J("conditional_pop.json")["features"]
        by_tok = defaultdict(list)
        for f, r in pop.items():
            off = (r.get("L7") or {}).get("off") or (r.get("L4") or {}).get("off")
            on = (r.get("L7") or {}).get("on") or (r.get("L4") or {}).get("on")
            if off:
                by_tok[r["token"]].append((off["argmax_is_X"], (on or {}).get("dlogit_X", 0.0), int(f)))
        toks = sorted(by_tok, key=lambda t: -max(v[0] for v in by_tok[t]))
        fig, axes = plt.subplots(1, 2, figsize=(12, 0.26 * len(toks) + 1.6), facecolor=SURFACE, layout="constrained",
                                 sharey=True)
        ys = np.arange(len(toks))
        for ax, idx, title, xlabel in ((axes[0], 0, "Off where X is fed back: X is output again",
                                        "share of next outputs that repeat X (%)"),
                                       (axes[1], 1, "On before X's turn: X's logit at its own output",
                                        "logit change of X")):
            for y, t in zip(ys, toks):
                vals = sorted(by_tok[t], key=lambda v: -v[0])
                best = vals[0]
                ax.barh(y, 100 * best[idx] if idx == 0 else best[idx], color=sparse_c, height=0.6)
                for v in vals[1:]:
                    ax.plot(100 * v[idx] if idx == 0 else v[idx], y, "o", ms=3.5, color=INK_2)
                if idx == 0:
                    ax.text(100 * best[0] + 1.5, y, f"{best[2]}" + (f"  (+{len(vals) - 1} inert)" if len(vals) > 1 else ""),
                            va="center", fontsize=7, color=INK)
            style(ax, title, xlabel, None)
            ax.grid(axis="y", visible=False)
        axes[0].set_yticks(ys, [f"fed-back {t}" for t in toks], fontsize=7.5)
        axes[0].invert_yaxis()
        save(fig, "pop_features_by_token_conditional.png",
             "Token-specific pop features: one clamp test per 'copy input X' feature, measured only where X is involved",
             "Real GRU, length 7 (length 4 where a feature has no L=7 test), 1000 sequences. Every candidate feature whose label "
             "is 'copy-phase input X' (any verdict). Bar: the feature of that token with the largest effect (its id is printed); "
             "grey dots: the other features with the same label. Left: feature set to 0 at the step X is fed back. Right: feature "
             "set to its mean active value at a copy step before X is output. conditional_analyses.py --sections pop.")

    if os.path.exists(os.path.join(d, "conditional_flows.json")):
        fl = J("conditional_flows.json")
        F = fl["features"]
        ids = np.array([f["feature"] for f in F])
        rate = np.array([f["seq_rate"] for f in F])
        a_all = np.array([f["mean_abs_flow_all"] for f in F])
        a_part = np.array([f["mean_abs_flow_when_taking_part"] for f in F])
        fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor=SURFACE, layout="constrained")
        dense = rate >= 0.9
        ax.scatter(a_all[dense], a_part[dense], s=16, color=dense_c, alpha=0.75, lw=0, label="dense (active in ≥ 90% of sequences)")
        ax.scatter(a_all[~dense], a_part[~dense], s=16, color=sparse_c, alpha=0.75, lw=0, label="sparse")
        hi = max(a_all.max(), a_part.max()) * 1.05
        ax.plot([0, hi], [0, hi], color=INK_2, lw=0.8, ls="--")
        for k in np.argsort(-a_part)[:n_labels]:
            ax.annotate(str(ids[k]), (a_all[k], a_part[k]), xytext=(3, 3), textcoords="offset points", fontsize=7, color=INK)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        style(ax, f"length {fl['L']}", "mean |flow| over all (sequence, slot) pairs",
              "mean |flow| over the pairs it takes part in")
        save(fig, f"carrier_flow_conditional_L{fl['L']}.png",
             "How much each candidate feature carries a token: averaged over all pairs vs only where it takes part",
             f"Exact contrastive flow (DeepLIFT rescale) of the token in a slot, original vs token-swapped run, summed over steps "
             f"i … L+i; {fl['n']} sequences × {fl['L']} slots. A feature 'takes part' in a pair if its |flow| > {fl['min_flow']}. "
             "The earlier circuit diagrams and supernodes used the x-axis average. conditional_analyses.py --sections flows.")

        # wheel: write angle vs conditional write weight
        ok = [f for f in F if f["write_angle"] is not None and f["write_weight_when_taking_part"] > 0 and f["share_of_pairs"] > 0.01]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"projection": "polar"}, facecolor=SURFACE,
                                 layout="constrained")
        for ax, key, title in ((axes[0], "write_weight_all", "weight averaged over all pairs (as before)"),
                               (axes[1], "write_weight_when_taking_part", "weight over the pairs it takes part in")):
            ax.set_facecolor(SURFACE)
            w = np.array([f[key] for f in ok])
            wmax = w.max()
            for f, ww in sorted(zip(ok, w), key=lambda z: z[1]):
                c = dense_c if f["seq_rate"] >= 0.9 else sparse_c
                ax.annotate("", xy=(np.deg2rad(f["write_angle"]), ww / wmax), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=0.6 + 1.4 * ww / wmax, alpha=0.7))
            top = [ok[k]["feature"] for k in np.argsort(-w)[:5]]
            ax.text(np.deg2rad(-60), 1.45, "largest: " + ", ".join(map(str, top)), ha="center", fontsize=7.5, color=INK_2)
            ax.set_ylim(0, 1.25)
            ax.set_yticks([])
            ax.set_xticks(np.deg2rad([0, 90, 180, 270]), ["0°", "90°", "180°", ""], fontsize=7.5, color=INK_2)
            ax.grid(color=GRID, lw=0.6)
            ax.spines["polar"].set_color(GRID)
            ax.set_title(title, fontsize=9.5, color=INK, pad=14)
        handles = [plt.Line2D([], [], color=dense_c, lw=2, label="dense"), plt.Line2D([], [], color=sparse_c, lw=2, label="sparse")]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.05))
        save(fig, f"rotation_wheel_conditional_L{fl['L']}.png",
             "Write angles of the candidate features, weighted over all pairs vs only the pairs each takes part in",
             "Each arrow: one feature at the circular mean angle it writes the token (0° = the token's code as written, "
             "180° = towards its read direction; pooled over steps i … L+i and slots), length = its in-plane write. "
             "Features taking part in > 1% of pairs. conditional_analyses.py --sections flows.")
    return paths


def circuit_summary_plots(d, outdir):
    """Evidence figures for docs/interp_circuits.md, from the JSON outputs in features/copy_circuits/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    J = lambda name: json.load(open(os.path.join(d, name)))
    paths = []

    def style(ax, title=None, xlabel=None, ylabel=None):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        if title:
            ax.set_title(title, fontsize=10, color=INK, loc="left")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=8.5, color=INK_2)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8.5, color=INK_2)

    def save(fig, name, title, foot):
        fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
        fig.supxlabel(textwrap.fill(foot, int(fig.get_figwidth() * 15)), fontsize=7.5, color=INK_2, x=0.01, ha="left")
        p = os.path.join(outdir, name)
        fig.savefig(p, dpi=170, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    ex = J("circuit_extras.json")
    ct = J("circuit_tests_L4.json")

    # 1. the trace turns: cosine with the written code, and visibility to the output layer
    gates = next(g for g in J("gates.json")["lengths"] if g["L"] == 4)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE, layout="constrained")
    for i, color in zip(range(4), SOURCE_COLORS):
        cs = ex["trace_cosine_L4"][str(i)]
        axes[0].plot(range(len(cs)), list(cs.values())[:len(cs)], marker="o", color=color, lw=2, label=f"slot {i}")
        steps = gates["slots"][i]["steps"]
        axes[1].plot(range(len(steps)), [s["readout"] for s in steps], marker="o", color=color, lw=2, label=f"slot {i}")
    for ax in axes:
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.legend(frameon=False, fontsize=8)
    style(axes[0], "Cosine of the token's trace with its code at the write step", "steps since the write", "cosine")
    style(axes[1], "How visible the trace is to the output layer", "steps since the write",
          "logit(token) - logit(swapped token)")
    save(fig, "trace_turning_L4.png", "Length 4: a stored token's code turns ~180° and lines up with the output at its output step",
         "Real GRU, 3000 sequences. Trace = hidden state minus the same run with the token in that slot swapped (feedback matched). "
         "Every slot is output L steps after it is written (4 steps at L = 4). Left: circuit_extras.py "
         "trace_cosine; right: gate_mechanism.py readout.")

    # 2. the pop: kept part and candidate write of the head token
    ps = ex["pop_split_L4"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor=SURFACE, layout="constrained")
    cats = [("kept from memory", lambda r: r["matched"]["kept"]), ("candidate write, memory only", lambda r: r["matched"]["written_memory_only"]),
            ("write, matched token fed back", lambda r: r["matched"]["written"]), ("write, other token fed back", lambda r: r["mismatched"]["written"])]
    xs = np.arange(len(ps))
    for j, ((name, fn), color) in enumerate(zip(cats, SOURCE_COLORS)):
        vals = [fn(r) for r in ps.values()]
        ax.bar(xs + (j - 1.5) * 0.2, vals, 0.18, color=color, label=name)
        for xx, v in zip(xs + (j - 1.5) * 0.2, vals):
            ax.text(xx, v + (0.6 if v >= 0 else -2.2), f"{v:+.0f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(xs, [f"pop at {k}" for k in ps])
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_ylim(-4, 48)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
    style(ax, None, None, "logit of the head token after the step")
    save(fig, "pop_cancellation_L4.png", "Length 4: the fed-back token cancels its own stored copy in the candidate's write",
         "Real GRU, 3000 sequences (circuit_extras.py pop_split). At each pop step the update gate replaces ~70% of memory either way; "
         "the head survives only if the candidate writes it again. Memory alone would (+30); the matched input term cancels it (~0); "
         "another token does not (+30, so the head is output again).")

    # 3. feeding back a token early
    ef = ex["early_feedback_L4"]["copy step L+2"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), facecolor=SURFACE, layout="constrained")
    names = list(ef)
    xs = np.arange(len(names))
    for j, (key, color) in enumerate(zip(["output k = orig[k]", "output k = orig[k-1]"], SOURCE_COLORS)):
        axes[0].bar(xs + (j - 0.5) * 0.35, [ef[n][key] for n in names], 0.33, color=color,
                    label={"output k = orig[k]": "outputs the right token", "output k = orig[k-1]": "outputs the previous token again"}[key])
    axes[0].set_xticks(xs, [textwrap.fill(n, 14) for n in names], fontsize=7.5)
    axes[0].set_ylim(0, 1.3)
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    style(axes[0], "What is output at step k", None, "share of sequences")
    for j, (tok, color) in enumerate(zip(["orig[k-1]", "orig[k]", "orig[k+1]"], SOURCE_COLORS)):
        axes[1].bar(xs + (j - 1) * 0.26, [ef[n]["write"][tok]["with input"] for n in names], 0.24, color=color, label=f"write for {tok}")
    axes[1].axhline(0, color=INK_2, lw=0.8)
    axes[1].set_xticks(xs, [textwrap.fill(n, 14) for n in names], fontsize=7.5)
    axes[1].legend(frameon=False, fontsize=8)
    style(axes[1], "Candidate's write for each queued token at step k", None, "logit contribution")
    save(fig, "early_feedback_L4.png", "Length 4: any fed-back token cancels its own write, wherever it is in the queue",
         "Real GRU, 3000 sequences, copy step L+2 (circuit_extras.py early_feedback). Feeding back the token due now or next turns "
         "its write negative (-14, -26); the head is not popped, so it is output again.")

    # 4. queue tests: freezing h for one copy step
    tm = ct["timing"]
    rows = [k for k in tm if k.startswith("freeze")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE, layout="constrained")
    for ax, key, title in ((axes[0], "accuracy", "Accuracy per output"), (axes[1], "repeat_previous", "Share outputting the previous token")):
        m = np.array([tm[r][key] for r in rows])
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (a, b), v in np.ndenumerate(m):
            ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="#ffffff" if v > 0.6 else INK)
        ax.set_yticks(range(len(rows)), [r.replace("freeze h at step ", "freeze ") for r in rows], fontsize=7.5)
        ax.set_xticks(range(m.shape[1]), [f"output {j}" for j in range(m.shape[1])], fontsize=8)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    save(fig, "queue_freeze_L4.png", "Length 4: skipping one copy step delays the rest of the output by one (free-running)",
         "Real GRU, 3000 sequences (circuit_tests.py timing). Freeze = z = 0 for one step (nothing written or erased). Free-running: "
         "the model's own outputs are fed back; every later output is the previous token, i.e. delayed, not skipped.")

    # 5. dense carriers are needed
    dd = ex["dense_drop"]
    keys = ["none", "the 5", "the 15 dense circuit carriers", "the 5 + the 15", "all other dense"] + \
           [k for k in dd if k.startswith("random")] + sorted([k for k in dd if k.startswith("only")], key=lambda k: dd[k]["accuracy"])
    fig, ax = plt.subplots(figsize=(8, 7.5), facecolor=SURFACE, layout="constrained")
    vals = [dd[k]["accuracy"] for k in keys]
    colors = [SOURCE_COLORS[0] if k.startswith(("none", "the", "all")) else SOURCE_COLORS[2] if k.startswith("random") else SOURCE_COLORS[1]
              for k in keys]
    ax.barh(range(len(keys)), vals, color=colors)
    for y, v in enumerate(vals):
        ax.text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(keys)), keys, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    style(ax, None, "copy accuracy (lengths 3-9)")
    save(fig, "dense_carriers_needed.png", "The dense circuit carriers outside the earlier '5' are needed",
         "Candidate transcoder spliced into the GRU, features mean-ablated, 10000 held-out sequences of lengths 3-9 "
         "(circuit_extras.py dense_drop). Blue: sets; green: random controls; orange: one carrier at a time.")

    # 6. reset features: L=4 vs L=7
    rc = {L: J(f"reset_combos_L{L}.json")["reset_combos"] for L in (4, 7)}
    keys = [k for k in rc[4] if k.endswith(" off") and "+" not in k] + \
           ["input phase (76, 147, 312, 271)", "<DEL> (31)", "copy phase (434, 322, 477)", "input + <DEL>", "<DEL> + copy", "all 8"] + \
           [k for k in rc[4] if k.startswith("random")]
    fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor=SURFACE, layout="constrained")
    ys = np.arange(len(keys))
    for j, L in enumerate((4, 7)):
        vals = [rc[L][k]["accuracy_all"] for k in keys]
        ax.barh(ys + (j - 0.5) * 0.38, vals, 0.36, color=SOURCE_COLORS[j], label=f"L = {L}")
        for y, v in zip(ys + (j - 0.5) * 0.38, vals):
            ax.text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=6.8, color=INK)
    ax.set_yticks(ys, keys, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    style(ax, None, "copy accuracy")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    save(fig, "reset_features_by_length.png", "Reset-gate features matter under heavier memory load (L = 7), little at L = 4",
         "Real GRU, 3000 sequences per length (circuit_tests.py reset_combos). Reset features turned off (zeroed) at every step; "
         "random: 8 other active reset features.")

    # 7. gate time-shifts
    ts = ct["timeshift"]
    keys = list(ts)
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=SURFACE, layout="constrained")
    xs = np.arange(len(keys))
    for j in range(4):
        ax.bar(xs + (j - 1.5) * 0.2, [ts[k]["accuracy"][j] for k in keys], 0.18, color=SOURCE_COLORS[j], label=f"output {j}")
    ax.set_xticks(xs, [textwrap.fill(k, 16) for k in keys], fontsize=6.8)
    ax.set_ylim(0, 1.18)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    style(ax, None, None, "accuracy")
    save(fig, "gate_timeshift_L4.png", "Length 4: giving a gate feature its value from another step (the clock is wrong)",
         "Real GRU, 3000 sequences (circuit_tests.py timeshift). 717 left on in the copy phase and 434 turned on during input break "
         "copying; 650 off early hurts late outputs.")

    # 8. reversal tests
    rv = J("reversal_tests_L4.json")["tests"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), facecolor=SURFACE, layout="constrained", sharey=True)
    for i, ax in enumerate(axes):
        for key, color, label in (("none", INK, "no intervention"), ("A", SOURCE_COLORS[0], "A: keepers removed at +1"),
                                  ("B", SOURCE_COLORS[1], "B: flippers removed at the write step")):
            c = rv[key][str(i)]["flipper_corr"]
            ax.plot([int(k) for k in c], list(c.values()), marker="o", color=color, lw=2, label=label)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_xticks([1, 2, 3])
        style(ax, f"slot {i}", "steps since the write", "flippers: input vs memory r" if i == 0 else None)
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "reversal_tests_L4.png", "Length 4: what makes the flippers reverse (causal tests)",
         "Real GRU, 3000 sequences (rotation_edges.py --reversal_tests). r < 0: a flipper's memory code is opposite to its input "
         "code. Removing the flippers' input drive (B) makes them reverse a step early; removing 10 keepers at +1 (A) weakens the "
         "reversal. Random controls (not drawn) change nothing.")

    # 10. minimal circuit: sufficiency and complement against circuit size
    if os.path.exists(os.path.join(d, "circuit_sufficiency_L4.json")):
        sf = J("circuit_sufficiency_L4.json")["slots"]
        ks = [int(k.split()[-1]) for k in sf["0"] if k.startswith("circuit top")]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), facecolor=SURFACE, layout="constrained", sharey=True)
        series = [("circuit", lambda r, k: r[f"circuit top {k}"], SOURCE_COLORS[0]),
                  ("random features outside the circuit", lambda r, k: r[f"random {k} (outside the circuit)"], SOURCE_COLORS[1])]
        panels = [
            (axes[0, 0], "sufficiency_free", False, "Sufficiency, free: swapped run, circuit set to original values,\nthe rest recomputed",
             "output i = original token"),
            (axes[0, 1], "sufficiency_clamped", False, "Sufficiency, clamped: circuit at original values,\nevery other feature clamped to the swapped run", None),
            (axes[1, 0], "necessity_free", True, "Necessity, free: original run, circuit set to swapped values,\nthe rest recomputed",
             "output i is no longer the original token"),
            (axes[1, 1], "necessity_clamped", True, "Necessity, clamped: circuit at swapped values,\nevery other feature clamped to the original run", None)]
        for ax, mode, flip, title, ylabel in panels:
            key = f"{mode}_output_is_v"
            val = (lambda x: 1 - x) if flip else (lambda x: x)
            for name, get, color in series:
                for r in sf.values():
                    ax.plot(ks, [val(get(r, k)[key]) for k in ks], color=color, lw=0.8, alpha=0.35)
                ax.plot(ks, [np.mean([val(get(r, k)[key]) for r in sf.values()]) for k in ks], marker="o", color=color, lw=2.2,
                        label=name)
            ax.set_xscale("log")
            ax.set_xticks(ks, [str(k) for k in ks])
            ax.set_ylim(-0.03, 1.05)
            style(ax, None, "features per step (top k by |contrastive flow|)" if flip else None, ylabel)
            ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        axes[0, 0].legend(frameon=False, fontsize=8, loc="center right")
        cover = {k: np.mean([r[f"circuit top {k}"]["share_of_abs_flow"] for r in sf.values()]) for k in ks}
        save(fig, "minimal_circuit_L4.png", "Length 4: the circuit is both sufficient and necessary to carry token i to output L + i",
             "Real GRU, 300 sequences per slot (circuit_sufficiency.py). Circuit = each sequence's top-k candidate features per step "
             "(steps i .. L+i) by |contrastive flow| between the original run and the run with token i swapped. Patching sets a "
             "feature to its value in the other run. Free: unpatched features are recomputed; clamped: they are held at the other "
             "run's values. Thin lines: slots 0-3; thick: mean. Share of |flow| in the top k: "
             + ", ".join(f"k={k} {v:.0%}" for k, v in cover.items()) + ".")

    # 9. carry test
    cr = ct["carry"]
    keys = sorted({k.rsplit(":", 1)[0] for k in cr})
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor=SURFACE, layout="constrained")
    xs = np.arange(len(keys))
    for j, (cond, color) in enumerate(zip(["steer", "z=1", "z=1 + steer"], SOURCE_COLORS)):
        ax.bar(xs + (j - 1) * 0.26, [cr[f"{k}: {cond}"]["flip_to_new" if cond != "z=1" else "acc_own"] for k in keys], 0.24,
               color=color, label={"steer": "steer only: share switching to the new token",
                                   "z=1": "z = 1 only: accuracy of that output (rewrite alone)",
                                   "z=1 + steer": "z = 1 + steer: share switching"}[cond])
    ax.set_xticks(xs, [textwrap.fill(k, 10) for k in keys], fontsize=7)
    ax.set_ylim(0, 1.2)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    style(ax, None, None, "share")
    save(fig, "carry_second_copy_L4.png", "Length 4: the update gate's carry holds a second copy of the token",
         "Real GRU, 3000 sequences (circuit_tests.py carry). Steering the slot's carriers by the token tables at one relay step "
         "often fails; dropping the carried copy at that step (z = 1) lets it work. The rewrite alone mostly keeps the token.")
    return paths


def short_text(text, n):
    text = (text or "?").split(";")[0].strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("specificity"); s.add_argument("results"); s.add_argument("--out", default=None)
    c = sub.add_parser("coverage", help="timestep coverage of features grouped by explanation F1")
    c.add_argument("store"); c.add_argument("results"); c.add_argument("--gates", nargs="*")
    c.add_argument("--subset_store", help="only count sequences of this store (e.g. a unit store on a subsample)")
    c.add_argument("--out", help="write the numbers as JSON")
    cp = sub.add_parser("coverage_plots", help="activation-share and step-coverage figures from saved coverage JSONs")
    cp.add_argument("features_json", help="coverage --out JSON of the transcoder features")
    cp.add_argument("--units", nargs="*", default=[], metavar="LABEL=JSON",
                    help='unit baselines, e.g. "units, matched density=features/<run>/coverage_by_f1.json"')
    cp.add_argument("--out", help="output directory (default: next to the features JSON)")
    f = sub.add_parser("fidelity", help="figures for circuit/copy_task/explained_fidelity.py output")
    f.add_argument("fidelity_json"); f.add_argument("--out", help="output directory (default: next to the JSON)")
    pr = sub.add_parser("probe", help="token tables from circuit/copy_task/feature_probe.py output")
    pr.add_argument("probe_json"); pr.add_argument("--out", help="output directory (default: next to the JSON)")
    ci = sub.add_parser("circuits", help="figures for circuit/copy_task/copy_circuits.py output")
    ci.add_argument("circuits_json"); ci.add_argument("--out", help="output directory (default: next to the JSON)")
    fc = sub.add_parser("feature_circuit", help="circuit diagrams from circuit/copy_task/feature_circuits.py output")
    fc.add_argument("feature_circuits_json"); fc.add_argument("--gates", help="gate_mechanism.py JSON (gate tags)")
    fc.add_argument("--circuits", help="copy_circuits.py JSON (pop strength)")
    fc.add_argument("--out", help="output directory (default: next to the JSON)")
    tk = sub.add_parser("token_circuit", help="token-specific circuit diagrams from circuit/copy_task/token_circuits.py output")
    tk.add_argument("token_circuits_json"); tk.add_argument("--out", help="output directory (default: next to the JSON)")
    rw = sub.add_parser("rotation_wheel", help="angle wheels from circuit/copy_task/rotation_edges.py --phases output")
    rw.add_argument("rotation_json"); rw.add_argument("--out", help="output directory (default: next to the JSON)")
    cs = sub.add_parser("circuit_summary", help="evidence figures for docs/interp_circuits.md")
    cs.add_argument("dir", help="features/copy_circuits"); cs.add_argument("--out", help="output directory (default: DIR/plots)")
    cd = sub.add_parser("conditional", help="figures for circuit/copy_task/conditional_analyses.py output")
    cd.add_argument("dir", help="features/copy_circuits"); cd.add_argument("--out", help="output directory (default: DIR/plots)")
    sc = sub.add_parser("set_controls", help="figure for explained_fidelity.py --set_controls output")
    sc.add_argument("json"); sc.add_argument("--out", required=True, help="output PNG")
    args = ap.parse_args()
    if args.cmd == "set_controls":
        print(set_controls_plot(args.json, args.out))
        return
    if args.cmd == "coverage_plots":
        units = {k: json.load(open(v)) for k, v in (u.rsplit("=", 1) for u in args.units)}
        for p in coverage_plots(json.load(open(args.features_json)), units,
                                args.out or os.path.dirname(args.features_json)):
            print(p)
        return
    if args.cmd == "conditional":
        for p in conditional_plots(args.dir, args.out or os.path.join(args.dir, "plots")):
            print(p)
        return
    if args.cmd == "circuit_summary":
        for p in circuit_summary_plots(args.dir, args.out or os.path.join(args.dir, "plots")):
            print(p)
        return
    if args.cmd == "rotation_wheel":
        for p in rotation_wheel_plots(args.rotation_json, args.out or os.path.join(os.path.dirname(args.rotation_json), "plots")):
            print(p)
        return
    if args.cmd == "token_circuit":
        for p in token_circuit_plots(args.token_circuits_json,
                                     args.out or os.path.join(os.path.dirname(args.token_circuits_json), "plots")):
            print(p)
        return
    if args.cmd == "feature_circuit":
        for p in feature_circuit_plots(args.feature_circuits_json,
                                       args.out or os.path.join(os.path.dirname(args.feature_circuits_json), "plots"),
                                       args.gates, args.circuits):
            print(p)
        return
    if args.cmd == "circuits":
        for p in circuit_plots(args.circuits_json, args.out or os.path.join(os.path.dirname(args.circuits_json), "plots")):
            print(p)
        return
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
