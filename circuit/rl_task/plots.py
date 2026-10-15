"""Figures for the RL feature analyses.

    python -m circuit.rl_task.plots all [--out docs/figures/rl]
    python -m circuit.rl_task.plots interp | fidelity | sets | substitutes | probe | scorecards | time

  interp       held-out F1 against firing rate, transcoder features vs thresholded GRU
               units (neuron baseline, if its results exist), and the share explained
  fidelity     switch / GO agreement and value error with only F1-selected, explained,
               random and core features kept (rl_fidelity/fidelity.json)
  sets         behaviour-ranked top-k sets, behaviour cores and backward elimination, for
               the choice and the value (reset_sets, value_vs_switch, backward / split selection)
  substitutes  can the choice-only / value-only features be swapped for their nearest
               substitutes? (substitutes.json)
  probe        conditional ablation, magnitude regression and residual tables of the
               probed features (rl_probe/*_probe.json)
  scorecards   agent predicate F1 vs fitted scorecard F1 (scorecard fields in the results)
  time         activation mass by episode trial for features in the choice / value cores
  breakdown    how the features that matter are explained (refined), firing-rule coverage and discrete levels, vs units
  clock        the value head's late fall and the late-onset candidate features that mirror it
  tests        blind simulation, per-component substitution and story test: features (first / current) vs units

Inputs default to the paths used in the RL runs (rl_features/...). fidelity, sets, substitutes, probe, scorecards,
time, ranked and substitution belong to abandoned analyses: they write to <out>/archived and `all` skips them
(docs/figures/rl/archived/README.md); conditional writes its clamp figure there too.
"""
import argparse
import json
import os

import numpy as np

from circuit.copy_task.autointerp_plots import BAND_COLORS, GRID, INK, INK_2, SOURCE_COLORS, SURFACE

FEATURES = "rl_features/rl_autointerp"
STORE = "rl_features/rl_chosen/chosen_rl_v1_r128_store"
FIDELITY = "rl_features/rl_fidelity"
PROBE = "rl_features/rl_probe"
UNITS = [("1% density", "rl_features/rl_neuron_d01"), ("matched density", "rl_features/rl_neuron_matched"),
         ("5% density", "rl_features/rl_neuron_d05")]
GATES = [("reset", "reset", "Reset gate", ["neuron_reset"]), ("update", "update", "Update gate", ["neuron_update"]),
         ("candidate", "hidden", "Candidate", ["neuron_cand_pos", "neuron_cand_neg"])]
GATE_COLORS = {"reset": "#2a78d6", "update": "#eb6834", "candidate": "#1baf7a"}
METRICS = [("switch_agreement", "Switch agreement", True), ("go_agreement", "GO agreement", True),
           ("value_nmae", "Value error (nMAE)", False)]
N_ROWS = 2329047


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return matplotlib, plt


def style(ax, title=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=11, color=INK, loc="left")


def save(fig, out, name, foot=None):
    if foot:
        fig.text(0.01, 0.005, foot, fontsize=7.5, color=INK_2, va="bottom", wrap=True)
    path = os.path.join(out, name)
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    print(f"wrote {path}")
    return path


def results(results_dir, gates):
    """{(gate, feature): (verdict, test F1 or 0, record)} for result JSONs of the given gates."""
    out = {}
    if not os.path.isdir(results_dir):
        return out
    for name in os.listdir(results_dir):
        if not name.endswith(".json") or name == "summary.json":
            continue
        rec = json.load(open(os.path.join(results_dir, name)))
        if rec.get("gate") in gates:
            out[(rec["gate"], int(rec["feature"]))] = (rec.get("final_verdict") or rec.get("verdict"),
                                                       (rec.get("test_scores") or {}).get("f1") or 0.0, rec)
    return out


def counts(store):
    return json.load(open(os.path.join(store, "store.json")))["counts"]


def label_of(rec, n=60):
    s = rec.get("final_label") or rec.get("label") or ""
    return s if len(s) <= n else s[:n - 1] + "…"


# ----------------------------------------------------------------------------- interp

def interp(out):
    matplotlib, plt = _plt()
    feats = results(FEATURES, {"reset", "update", "hidden"})
    fc = counts(STORE)
    units = []
    for label, prefix in UNITS:
        r = results(f"{prefix}_autointerp", {"neuron_reset", "neuron_update", "neuron_cand_pos", "neuron_cand_neg"})
        if r:
            units.append((label, r, counts(f"{prefix}_store")))
    # matched-density update units = the 1% units (same density; not run twice)
    unit_marker = {"1% density": "s", "matched density": "D", "5% density": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, facecolor=SURFACE)
    for ax, (gate, rgate, title, ugates) in zip(axes, GATES):
        style(ax, title)
        rate = [fc[rgate].get(str(f), [0, 0])[1] / N_ROWS for (g, f) in feats if g == rgate]
        f1 = [v[1] for (g, f), v in feats.items() if g == rgate]
        keep = [i for i, r in enumerate(rate) if r > 0]
        ax.scatter(np.array(rate)[keep], np.array(f1)[keep], s=10, color=GATE_COLORS[gate], alpha=0.55,
                   linewidth=0, label=f"transcoder features (n={len(keep)})")
        for k, (ulabel, ures, uc) in enumerate(units, start=1):
            pts = [(uc[g][str(f)][1] / N_ROWS, v[1]) for (g, f), v in ures.items() if g in ugates and str(f) in uc.get(g, {})]
            if pts:
                x, y = zip(*pts)
                ax.scatter(x, y, s=22, marker=unit_marker[ulabel], facecolor="none", edgecolor=SOURCE_COLORS[k % 4],
                           linewidth=1, label=f"GRU units, {ulabel} (n={len(pts)})")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Firing rate (share of all steps)", fontsize=9, color=INK)
        ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="lower left")
    axes[0].set_ylabel("Held-out F1 of the recorded predicate", fontsize=10, color=INK)
    fig.suptitle("RL agent: how well does a firing predicate explain each feature / unit?", fontsize=13, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    paths = [save(fig, out, "interp_f1_vs_rate.png", "F1 0 = no predicate recorded. Test split = odd episodes. "
                  "Units: a GRU gate coordinate counted active on its top fraction of steps.")]
    plt.close(fig)

    # share explained by source, at each gate
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [4, 3, 7]})
    bands = [("F1 >= 0.8", 0.8, 1.01), ("0.5 - 0.8", 0.5, 0.8), ("0.2 - 0.5", 0.2, 0.5), ("< 0.2 or none", -1, 0.2)]
    for ax, (gate, rgate, title, ugates) in zip(axes, GATES):
        style(ax, title)
        rows = [("features", [v[1] for (g, _), v in feats.items() if g == rgate and fc[rgate].get(str(_), [0, 0])[1]])]
        for ulabel, ures, _ in units:
            for ug in ugates:
                vals = [v[1] for (g, _), v in ures.items() if g == ug]
                if vals:
                    end = {"neuron_cand_pos": "\nhigh end", "neuron_cand_neg": "\nlow end"}.get(ug, "")
                    rows.append((f"units {ulabel.split()[0]}{end}", vals))
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        for name, lo, hi in bands:
            share = np.array([np.mean([(lo <= v < hi) for v in vals]) for _, vals in rows])
            ax.bar(x, share, bottom=bottom, color=BAND_COLORS[name], edgecolor=SURFACE, linewidth=1.5,
                   label=f"held-out {name}")
            for xi, s, b in zip(x, share, bottom):
                if s >= 0.08:
                    ax.text(xi, b + s / 2, f"{s:.0%}", ha="center", va="center", fontsize=7.5,
                            color="#ffffff" if name in ("F1 >= 0.8", "0.5 - 0.8") else INK)
            bottom += share
        ax.set_xticks(x, [f"{r[0]}\nn={len(r[1])}" for r in rows], fontsize=7.5, color=INK_2, rotation=0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Share of features / units", fontsize=10, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper left", bbox_to_anchor=(0.01, 0.93), frameon=False, fontsize=9,
               labelcolor=INK_2)
    fig.suptitle("RL agent: share of features / units with a precise firing explanation", fontsize=13, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    paths.append(save(fig, out, "interp_f1_bands.png"))
    plt.close(fig)

    # verdicts of the transcoder features
    verdicts = ["explained", "partially_explained", "polysemantic", "no_explanation", "too_sparse", "never_fires"]
    vcolors = ["#184f95", "#3987e5", "#86b6ef", "#cde2fb", "#d9d8d4", "#efeeeb"]
    fig, ax = plt.subplots(figsize=(10, 2.6), facecolor=SURFACE)
    style(ax, "Final verdicts (after the skeptic), transcoder features")
    ax.grid(False)
    for yi, (gate, rgate, title, _) in enumerate(GATES[::-1]):
        vs = [v[0] for (g, _), v in feats.items() if g == rgate]
        left = 0
        for v, c in zip(verdicts, vcolors):
            n = sum(x == v for x in vs)
            if n:
                ax.barh(yi, n, left=left, color=c, edgecolor=SURFACE, linewidth=1.5,
                        label=v.replace("_", " ") if yi == 0 else None)
                if n >= 8:
                    ax.text(left + n / 2, yi, str(n), ha="center", va="center", fontsize=8,
                            color="#ffffff" if c in vcolors[:2] else INK)
                left += n
    ax.set_yticks(range(3), [g[2] for g in GATES[::-1]], fontsize=9, color=INK)
    ax.set_xlabel("Features", fontsize=9, color=INK)
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.12), frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    paths.append(save(fig, out, "interp_verdicts.png"))
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- fidelity

def fidelity(out):
    matplotlib, plt = _plt()
    data = json.load(open(os.path.join(FIDELITY, "fidelity.json")))
    rows = data["rows"]
    taus = data["args"]["taus"]
    pick = lambda gate, sel, tau=None: [r for r in rows if r["gate"] == gate and r["selection"] == sel
                                        and (tau is None or r["tau"] == tau)]
    groups = [("explained", "explained", None)] + [(f"F1 ≥ {t}", "f1", t) for t in taus] + [("fve core", "core", None)]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), facecolor=SURFACE)
    for col, (gate, _, title, _) in enumerate(GATES):
        for row, (key, mname, higher) in enumerate(METRICS):
            ax = axes[row, col]
            style(ax, title if row == 0 else None)
            for k, (glabel, sel, tau) in enumerate(groups):
                r = pick(gate, sel, tau)
                if not r:
                    continue
                v = r[0][key]
                ax.bar(k, v, width=0.62, color=GATE_COLORS[gate], edgecolor=SURFACE)
                ax.text(k, v, f"{v:.2f}\n({r[0]['n_kept']})", ha="center", va="bottom", fontsize=7, color=INK_2)
                if sel == "f1":
                    rand = [x[key] for x in pick(gate, "random", tau)]
                    ax.scatter([k] * len(rand), rand, s=16, facecolor=SURFACE, edgecolor=INK, linewidth=1, zorder=3,
                               label="random, same count" if k == 1 else None)
            for sel, ls, lab in (("full", "-", "all features"), ("none", ":", "all mean-ablated")):
                r = pick(gate, sel)
                if r:
                    ax.axhline(r[0][key], color=INK_2, linewidth=1, linestyle=ls, label=lab)
            ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8, color=INK_2)
            if col == 0:
                ax.set_ylabel(mname + ("" if higher else " (lower is better)"), fontsize=9, color=INK)
            if key == "value_nmae":
                ax.set_ylim(0, max(0.2, max(x["value_nmae"] for x in rows if x["gate"] == gate) * 1.25))
            else:
                lo = min(x[key] for x in rows if x["gate"] == gate)
                span = 1 - lo
                ax.set_ylim(max(0, lo - 0.1 * span), 1 + 0.15 * span)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.955), frameon=False, fontsize=9,
               labelcolor=INK_2)
    fig.suptitle("RL agent: does it still behave the same with only the explained features kept?", fontsize=13,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    p = save(fig, out, "fidelity_by_selection.png",
             "Bars: value (features kept). Features not kept are set to their mean; the gate runs inside the GRU "
             "(errors compound over the window). Held-out episodes, 20,008 windows.")
    plt.close(fig)
    paths = [p]

    # transcoder features vs GRU units (fidelity_units.json), same selections
    upath = os.path.join(FIDELITY, "fidelity_units.json")
    if os.path.exists(upath):
        rows = rows + json.load(open(upath))["rows"]
        sources = [("features", "transcoder features"), ("units d01", "GRU units, 1%"),
                   ("units matched", "GRU units, matched"), ("units d05", "GRU units, 5%")]
        groups = [("explained", "explained", None), ("F1 ≥ 0.8", "f1", 0.8), ("F1 ≥ 0.5", "f1", 0.5), ("core", "core", None)]
        pick = lambda gate, src, sel, tau=None: [r for r in rows if r["gate"] == gate and r.get("source", "features") == src
                                                 and r["selection"] == sel and (tau is None or r["tau"] == tau)]
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), facecolor=SURFACE)
        width = 0.8 / len(sources)
        for col, (gate, _, title, _) in enumerate(GATES):
            for row, (key, mname, higher) in enumerate([METRICS[0], METRICS[2]]):
                ax = axes[row, col]
                style(ax, title if row == 0 else None)
                for j, (src, slabel) in enumerate(sources):
                    s = "units d01" if (src == "units matched" and gate == "update") else src   # same density
                    for k, (_, sel, tau) in enumerate(groups):
                        r = pick(gate, s, sel, tau)
                        if not r:
                            continue
                        x = k + (j - (len(sources) - 1) / 2) * width
                        ax.bar(x, r[0][key], width=width * 0.9, color=SOURCE_COLORS[j], edgecolor=SURFACE,
                               label=slabel if k == 0 else None)
                        ax.text(x, r[0][key], f"{r[0]['n_kept']}", ha="center", va="bottom", fontsize=6, color=INK_2)
                        if sel == "f1":
                            rand = [q[key] for q in pick(gate, s, "random", tau)]
                            ax.scatter([x] * len(rand), rand, s=10, facecolor=SURFACE, edgecolor=INK, linewidth=0.8,
                                       zorder=3, label="random, same count" if (j == 0 and k == 1) else None)
                for sel, ls, lab in (("full", "-", "all components"), ("none", ":", "all mean-ablated")):
                    ax.axhline(pick(gate, "features", sel)[0][key], color=INK_2, linewidth=0.9, linestyle=ls, label=lab)
                ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8.5, color=INK_2)
                if key == "switch_agreement":
                    ax.set_ylim(0.3, 1.04)
                if col == 0:
                    ax.set_ylabel(mname + ("" if higher else " (lower is better)"), fontsize=9, color=INK)
        handles, labels = [], []
        for ax in axes.ravel():
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h); labels.append(l)
        fig.legend(handles, labels, ncol=7, loc="upper left", bbox_to_anchor=(0.01, 0.95), frameon=False, fontsize=8.5,
                   labelcolor=INK_2)
        fig.suptitle("Explained transcoder features vs explained GRU units: does keeping only them preserve behaviour?",
                     fontsize=13, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.03, 1, 0.92))
        paths.append(save(fig, out, "fidelity_features_vs_units.png",
                          "Bar labels: components kept (features of 128 / 256; units of 48). Units: the real gate with "
                          "the other units at their mean; a candidate unit counts as explained if either end is. "
                          "Matched update units = the 1% units (same density). Core: by fve importance, not explanation. "
                          "Solid line: the full transcoder (all units kept = the original GRU, exact)."))
        plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- sets

def _set_series():
    """{gate: {series name: [(n_kept, metrics)]}} and {gate: {core name: (n, metrics)}}, plus full / none."""
    fid = json.load(open(os.path.join(FIDELITY, "fidelity.json")))["rows"]
    series, cores, refs = {}, {}, {}
    for gate, *_ in GATES:
        refs[gate] = {sel: next(r for r in fid if r["gate"] == gate and r["selection"] == sel) for sel in ("full", "none")}
        series[gate], cores[gate] = {}, {}
    rs = json.load(open(os.path.join(FIDELITY, "reset_sets.json")))
    vs = json.load(open(os.path.join(FIDELITY, "value_vs_switch.json")))
    for gate, rows in (("reset", rs["eval"]), ("update", vs["update"]["eval"]), ("candidate", vs["candidate"]["eval"])):
        for r in rows:
            name = r["set"]
            if name.startswith("top "):
                _, k, which = name.split()
                series[gate].setdefault(f"top-k by {which} drop", []).append((r["n_kept"], r))
            elif name.startswith("core ("):
                cores[gate][name[6:-1]] = (r["n_kept"], r)
            elif name.startswith("joint backward"):
                series[gate].setdefault("backward, joint objective", []).append((r["n_kept"], r))
    bs = json.load(open(os.path.join(FIDELITY, "backward_selection.json")))
    for gate, rows in bs.items():
        series[gate]["backward, joint objective"] = [(r["n_kept"], r) for r in rows]
    sp = json.load(open(os.path.join(FIDELITY, "split_selection.json")))
    for gate, d in sp.items():
        for which in ("choice", "value"):
            series[gate][f"backward, {which} objective"] = [(r["n_kept"], r) for r in d["eval"] if r["set"] == which]
    for gate in series:
        for k in series[gate]:
            series[gate][k].sort(key=lambda x: x[0])
    return series, cores, refs


SERIES_STYLE = {"top-k by switch drop": ("#2a78d6", "o", "-"), "top-k by value drop": ("#eb6834", "o", "-"),
                "backward, joint objective": ("#0b0b0b", "s", "-"), "backward, choice objective": ("#2a78d6", "s", "--"),
                "backward, value objective": ("#eb6834", "s", "--")}
CORE_STYLE = {"switch": ("#2a78d6", "*"), "value": ("#eb6834", "*"), "fve": ("#8a8984", "X")}


def sets(out):
    matplotlib, plt = _plt()
    series, cores, refs = _set_series()
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), facecolor=SURFACE)
    for col, (gate, _, title, _) in enumerate(GATES):
        for row, (key, mname, higher) in enumerate(METRICS):
            ax = axes[row, col]
            style(ax, title if row == 0 else None)
            for name, pts in series[gate].items():
                color, marker, ls = SERIES_STYLE[name]
                ax.plot([p[0] for p in pts], [p[1][key] for p in pts], color=color, marker=marker, linestyle=ls,
                        markersize=4.5, linewidth=1.4, label=name)
            for name, (n, r) in cores[gate].items():
                color, marker = CORE_STYLE[name]
                ax.scatter([n], [r[key]], s=120 if marker == "*" else 60, marker=marker, color=color,
                           edgecolor=INK, linewidth=0.6, zorder=4, label=f"{name} core")
            for sel, ls in (("full", "-"), ("none", ":")):
                ax.axhline(refs[gate][sel][key], color=INK_2, linewidth=0.9, linestyle=ls,
                           label="all features" if sel == "full" else "all mean-ablated")
            ax.set_xscale("log")
            ax.set_xticks([5, 10, 20, 40, 80], ["5", "10", "20", "40", "80"])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if row == 2:
                ax.set_xlabel("Features kept", fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel(mname + ("" if higher else " (lower is better)"), fontsize=9, color=INK)
    handles, labels = [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, ncol=5, loc="upper left", bbox_to_anchor=(0.01, 0.96), frameon=False, fontsize=8.5,
               labelcolor=INK_2)
    fig.suptitle("RL agent: how few features reproduce the choice and the value?", fontsize=13, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    p1 = save(fig, out, "sets_behaviour.png",
              "Top-k: features ranked by the drop when removed alone. Cores: shortest prefix of that ranking recovering "
              "90% of the gap (switch / value); fve core: by teacher-forced variance explained. Backward: greedy "
              "elimination on a separate selection sample. All scored on held-out episodes.")
    plt.close(fig)

    # overlap of the choice and value sets
    sp = json.load(open(os.path.join(FIDELITY, "split_selection.json")))
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    style(ax, "Overlap of the backward choice and value sets")
    for gate, d in sp.items():
        ns = sorted(int(n) for n in d["sets"]["choice"])
        jac = [len(set(d["sets"]["choice"][str(n)]) & set(d["sets"]["value"][str(n)])) /
               len(set(d["sets"]["choice"][str(n)]) | set(d["sets"]["value"][str(n)])) for n in ns]
        ax.plot(ns, jac, marker="o", color=GATE_COLORS[gate], label=gate)
    for gate, (n, r) in [(g, cores[g]["switch"]) for g in cores if "switch" in cores[g]]:
        sw, va = set(_core_ids(gate, "switch")), set(_core_ids(gate, "value"))
        ax.scatter([len(va)], [len(sw & va) / len(sw | va)], marker="*", s=130, color=GATE_COLORS[gate],
                   edgecolor=INK, linewidth=0.6, zorder=4)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 40, 80], ["10", "20", "40", "80"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Set size", fontsize=9, color=INK)
    ax.set_ylabel("Jaccard(choice set, value set)", fontsize=9, color=INK)
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p2 = save(fig, out, "sets_overlap.png", "Lines: backward elimination per objective. Stars: switch core vs value "
              "core (single-drop rankings), at the value core's size.")
    plt.close(fig)
    return [p1, p2]


def _core_ids(gate, which):
    if gate == "reset":
        return json.load(open(os.path.join(FIDELITY, "reset_sets.json")))["cores"][which]
    return json.load(open(os.path.join(FIDELITY, "value_vs_switch.json")))[gate]["cores"][which]


# ----------------------------------------------------------------------------- substitutes

def substitutes(out):
    matplotlib, plt = _plt()
    d = json.load(open(os.path.join(FIDELITY, "substitutes.json")))
    kinds = ["original", "shared only", "nearest substitutes", "random substitutes"]
    colors = ["#184f95", "#86b6ef", "#eb6834", "#d9d8d4"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), facecolor=SURFACE)
    for col, gate in enumerate(["update", "candidate"]):
        for row, (which, key, mname) in enumerate((("choice", "switch_agreement", "Switch agreement (choice set)"),
                                                   ("value", "value_nmae", "Value error, nMAE (value set)"))):
            ax = axes[row, col]
            style(ax, f"{gate.capitalize()}: {which} set" if True else None)
            ns = sorted(d[gate], key=int)
            width = 0.8 / len(kinds)
            for j, kind in enumerate(kinds):
                vals = []
                for n in ns:
                    f = d[gate][n]["functional"]
                    if kind == "random substitutes":
                        vs = [v[key] for k, v in f.items() if k.startswith(f"{which}: random")]
                        vals.append((np.mean(vs), np.min(vs), np.max(vs)))
                    else:
                        v = f[f"{which}: {kind}"][key]
                        vals.append((v, v, v))
                x = np.arange(len(ns)) + (j - 1.5) * width
                ax.bar(x, [v[0] for v in vals], width=width * 0.92, color=colors[j], edgecolor=SURFACE,
                       label=kind if row == 0 and col == 0 else None)
                if kind == "random substitutes":
                    ax.errorbar(x, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                                fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
            ax.set_xticks(range(len(ns)), [f"n = {n}" for n in ns], fontsize=8.5, color=INK_2)
            if key == "switch_agreement":
                ax.set_ylim(0.4, 1.0)
            if col == 0:
                ax.set_ylabel(mname, fontsize=9, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper left", bbox_to_anchor=(0.01, 0.94), frameon=False, fontsize=9,
               labelcolor=INK_2)
    fig.suptitle("Are the choice-only and value-only features replaceable?", fontsize=13, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    p = save(fig, out, "substitutes.png", "Each backward-elimination set (choice or value objective, size n): as "
             "selected; with only the features it shares with the other set; with its own-only features replaced by "
             "the most similar feature outside it (contribution cosine); or by random features (mean, min-max of 3).")
    plt.close(fig)
    return [p]


# ----------------------------------------------------------------------------- probe

def probe(out):
    matplotlib, plt = _plt()
    from circuit.rl_task.feature_probe import CONDITIONS
    paths = []
    for gate, rgate in (("candidate", "hidden"), ("update", "update")):
        path = os.path.join(PROBE, f"{gate}_probe.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        recs = results(FEATURES, {rgate})
        probed = [str(f) for f in d["args"]["features"] + d["args"]["controls"]]
        controls = {str(f) for f in d["args"]["controls"]}
        name = lambda f: (f"{f}: {label_of(recs[(rgate, int(f))][2], 48)}" + (" [control]" if f in controls else "")
                          if f != "all" else "all probed (not controls)")
        cond_names = [c for c, _ in CONDITIONS]
        sets_ = d["behaviour"]["sets"]
        plain = d["behaviour"]["plain_transcoder"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 0.42 * len(sets_) + 3.6), facecolor=SURFACE)
        for ax, (key, title, cmap, vmin, vmax) in zip(axes, (
                ("switch_agreement", f"Switch agreement (plain transcoder {plain['switch_agreement']:.2f})", "Blues",
                 0.3, 1.0),
                ("value_nmae", f"Value error, nMAE (plain transcoder {plain['value_nmae']:.3f})", "Oranges_r", 0.0, 0.6))):
            m = np.array([[s["conditions"][c][key] for c in cond_names] for s in sets_])
            ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    v = m[i, j]
                    dark = (v - vmin) / (vmax - vmin) > 0.6 if cmap == "Blues" else (v - vmin) / (vmax - vmin) < 0.4
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="#ffffff" if dark else INK)
            ax.set_xticks(range(len(cond_names)), [c.replace("phase, ", "") for c in cond_names], rotation=40,
                          ha="right", fontsize=8, color=INK_2)
            ax.set_yticks(range(len(sets_)), [name(s["set"]) for s in sets_], fontsize=8, color=INK)
            ax.set_title(title, fontsize=10.5, color=INK, loc="left")
            ax.set_xlabel("Feature replaced by its mean given (phase and) …", fontsize=9, color=INK)
        axes[1].set_yticks([])
        fig.suptitle(f"{gate.capitalize()}: which information does the model need from these features?", fontsize=13,
                     color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        paths.append(save(fig, out, f"probe_{gate}_ablation.png",
                          "Conditional mean ablation on held-out episodes; 'last trial' = previous choice x state "
                          "reached x reward. A cue that restores behaviour is what the feature carries."))
        plt.close(fig)

        # residual tables and magnitude R^2
        t = d["tables"]
        pats, phases = t["patterns"], t["phases"]
        fig, axes = plt.subplots(1, 2, figsize=(17, 0.42 * len(probed) + 3.2), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [3, 1.3]})
        full = np.array([[[np.nan if v is None else v for v in ph] for ph in t[f]["phase_x_last"]] for f in probed])
        lim = np.nanmax(np.abs(full))
        shown = [k for k in range(len(phases)) if np.nanmax(np.abs(full[:, k])) >= 0.05 * lim]   # phases with signal
        m = np.concatenate([full[:, k] for k in shown], 1)
        axes[0].imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if not np.isnan(m[i, j]) and abs(m[i, j]) >= 0.05 * lim:
                    axes[0].text(j, i, f"{m[i, j]:+.1f}", ha="center", va="center", fontsize=6.3,
                                 color="#ffffff" if abs(m[i, j]) > 0.6 * lim else INK)
        axes[0].set_xticks(range(8 * len(shown)), [p for _ in shown for p in pats], rotation=90, fontsize=7,
                           color=INK_2)
        for j, k in enumerate(shown):
            axes[0].text(j * 8 + 3.5, -0.75, phases[k], ha="center", va="bottom", fontsize=9, color=INK)
            if j:
                axes[0].axvline(j * 8 - 0.5, color=INK, linewidth=0.8)
        axes[0].set_yticks(range(len(probed)), [name(f) for f in probed], fontsize=8, color=INK)
        axes[0].set_title("Activation minus its (phase, episode decile) mean, by the previous trial "
                          "(choice, state reached, r = rewarded / u = not); phases where none fires are omitted",
                          fontsize=10, color=INK, loc="left", pad=22)
        reg = d["regression"]
        cols = ["timing", "timing+last trial", "timing+trial before", "timing+policy bin (model)", "observed", "all"]
        width = 0.8 / len(cols)
        ax = axes[1]
        style(ax, "Held-out R² of the magnitude")
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        y = np.arange(len(probed))
        ramp = ["#d9d8d4", "#86b6ef", "#3987e5", "#eda100", "#184f95", "#0b0b0b"]
        for j, c in enumerate(cols):
            ax.barh(y + (j - (len(cols) - 1) / 2) * width, [reg[f][c] for f in probed], height=width * 0.9,
                    color=ramp[j], label=c.replace("timing+", "+ "))
        ax.set_yticks(y, [""] * len(probed))
        ax.set_ylim(len(probed) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7, frameon=False, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=3)
        fig.suptitle(f"{gate.capitalize()}: what the probed features' magnitudes encode", fontsize=13, color=INK,
                     x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        paths.append(save(fig, out, f"probe_{gate}_tables.png",
                          "R²: ridge on one-hot cue groups, windows split by parity. timing = phase x episode decile; "
                          "observed = timing + last trial + trial before + this choice + this state."))
        plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- scorecards

def scorecards(out):
    matplotlib, plt = _plt()
    feats = results(FEATURES, {"reset", "update", "hidden"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, facecolor=SURFACE)
    for ax, (gate, rgate, title, _) in zip(axes, GATES):
        style(ax, title)
        pts = [(v[1], v[2]["scorecard"]["test_f1"], v[2]["scorecard"]["full_test_f1"])
               for (g, _), v in feats.items() if g == rgate and "scorecard" in v[2]]
        if not pts:
            continue
        a, r, f = map(np.array, zip(*pts))
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
        ax.scatter(a, f, s=10, color=GRID, edgecolor=INK_2, linewidth=0.3, label="full scorecard")
        ax.scatter(a, r, s=14, color=GATE_COLORS[gate], alpha=0.8, linewidth=0, label="readable (top-40 or two-slot)")
        ax.text(0.02, 0.97, f"n = {len(a)}\nmedian F1: agent {np.median(a):.2f}, readable {np.median(r):.2f}, "
                f"full {np.median(f):.2f}", transform=ax.transAxes, va="top", fontsize=8, color=INK_2)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Agent predicate, held-out F1", fontsize=9, color=INK)
        ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="lower right")
    axes[0].set_ylabel("Scorecard, held-out F1", fontsize=10, color=INK)
    fig.suptitle("RL agent: features no short predicate explains, fitted with additive scorecards", fontsize=13,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = save(fig, out, "scorecards_vs_agent.png", "Partially explained / unexplained / polysemantic features. "
             "Scorecards: logistic over one-hot cues of the step (phase, trial history, hidden state, model output bins).")
    plt.close(fig)
    return [p]


# ----------------------------------------------------------------------------- time

def time_profiles(out):
    matplotlib, plt = _plt()
    from circuit.rl_task.autointerp import Interp
    ip = Interp(STORE)
    dec = np.clip(np.asarray(ip.cols["ep_trial"]) // 10, 0, 9)
    steps = np.bincount(dec, minlength=10)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True, facecolor=SURFACE)
    for ax, (gate, rgate, title, _) in zip(axes, GATES):
        style(ax, title)
        sw, va = set(_core_ids(gate, "switch")), set(_core_ids(gate, "value"))
        for name, ids, color in (("in both cores", sw & va, "#184f95"), ("switch core only", sw - va, "#86b6ef"),
                                 ("value core only", va - sw, "#eb6834")):
            if not ids:
                continue
            prof = []
            for f in ids:
                a = ip.activations(rgate, f)
                m = np.bincount(dec, weights=a, minlength=10) / steps
                prof.append(m / m.sum() if m.sum() else m)
            prof = np.array(prof)
            ax.plot(range(10), prof.mean(0), marker="o", color=color, label=f"{name} (n={len(ids)})")
            ax.fill_between(range(10), np.percentile(prof, 25, 0), np.percentile(prof, 75, 0), color=color,
                            alpha=0.12, linewidth=0)
        ax.axhline(0.1, color=INK_2, linewidth=0.8, linestyle=":")
        ax.set_xticks(range(10), [f"{10 * k}" for k in range(10)], fontsize=8)
        ax.set_xlabel("Episode trial (start of 10-trial bin)", fontsize=9, color=INK)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK_2)
    axes[0].set_ylabel("Share of the feature's activation\n(per step, normalised over bins)", fontsize=9, color=INK)
    fig.suptitle("When in the episode do the choice and value features fire?", fontsize=13, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = save(fig, out, "time_profiles.png", "Mean over features (IQR shaded) of activation per step in each bin of "
             "the 100-trial episode; dotted = flat. Cores from single-drop rankings (sets figure).")
    plt.close(fig)
    return [p]


MAGNITUDE = [("reset", "unit_reset", "Reset gate"), ("update", "unit_update", "Update gate"),
             ("hidden", "unit_candidate", "Candidate")]


def magnitude(out):
    """Magnitude explanations (circuit/rl_task/autointerp_magnitude.py): features vs GRU units' raw values."""
    matplotlib, plt = _plt()
    load = lambda d: [json.load(open(os.path.join(d, n))) for n in os.listdir(d) if n.endswith(".json")] \
        if os.path.isdir(d) else []
    F, U = load("rl_features/rl_magnitude"), load("rl_features/rl_magnitude_units")
    if not F and not U:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, facecolor=SURFACE)
    rng = np.random.default_rng(0)
    cols = [("timing\nalone", lambda r: r["test_scores"]["r2_timing_only"]),
            ("short form\n(the label)", lambda r: r["test_scores"]["r2_beyond_timing"]),
            ("free fit\n(evidence)", lambda r: (r.get("full_test_scores") or r["test_scores"])["r2_beyond_timing"])]
    step = len(cols) + 0.8
    for ax, (fg, ug, title) in zip(axes, MAGNITUDE):
        style(ax, title)
        for k, (rows, gate, label, color) in enumerate(((F, fg, "transcoder features", GATE_COLORS[
                {"hidden": "candidate"}.get(fg, fg)]), (U, ug, "GRU units (raw value)", INK_2))):
            x = [r for r in rows if r["gate"] == gate]
            if not x:
                continue
            ok = np.array([r["final_verdict"] == "explained" for r in x])
            for j, (_, get) in enumerate(cols):
                v = np.array([get(r) for r in x])
                pos = k * step + j
                jit = rng.uniform(-0.18, 0.18, len(v))
                ax.scatter(pos + jit[ok], v[ok], s=12, color=color, alpha=0.75, linewidth=0)
                ax.scatter(pos + jit[~ok], v[~ok], s=14, facecolor="none", edgecolor=color, linewidth=0.8)
                ax.plot([pos - 0.3, pos + 0.3], [np.median(v)] * 2, color=INK, linewidth=2)
                ax.text(pos, 1.04, f"{np.median(v):.2f}", ha="center", fontsize=8, color=INK)
            ax.text(k * step + 1, -0.33, f"{label}\nn={len(x)}, {ok.sum()} explained ({ok.mean():.0%})", ha="center",
                    va="top", fontsize=8, color=INK_2)
        ax.set_xticks([k * step + j for k in range(2) for j in range(len(cols))], [c[0] for _ in range(2) for c in cols],
                      fontsize=7.5, color=INK_2)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(-0.6, step + len(cols) - 0.4)
    axes[0].set_ylabel("Held-out R² (timing alone: of the activation;\notherwise beyond timing)", fontsize=9, color=INK)
    fig.suptitle("Magnitude explanations: does a one-sentence interpretation capture what the free fit finds?",
                 fontsize=13, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    p = save(fig, out, "magnitude_features_vs_units.png",
             "Short form: at most 3 simple terms saying what the one-sentence label says (no hand-set constants or tables), "
             "the recorded explanation. Free fit: up to 8 terms, used to find hypotheses. Observed variables only (belief "
             "allowed, the model's own value / policy not). Filled = explained after the skeptic, open = not.")
    plt.close(fig)
    return [p]


def ranked(out):
    """Budget-matched fidelity (circuit/rl_task/ranked_fidelity.py): keep / drop the top-k by explanation score."""
    matplotlib, plt = _plt()
    path = os.path.join(FIDELITY, "ranked_tiers.json")
    if not os.path.exists(path):
        path = os.path.join(FIDELITY, "ranked.json")
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    rows, summ = d["rows"], d["summaries"]
    colors = {"features": None, "units": INK_2}
    paths = []
    for mode, title in (("keep", "Keep only the top-k (rest mean-ablated)"), ("drop", "Remove the top-k (keep the rest)")):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), facecolor=SURFACE)
        for col, (gate, _, gtitle, _) in enumerate(GATES):
            for row, (m, mname) in enumerate((("switch_agreement", "Switch recovery"), ("value_nmae", "Value recovery"))):
                ax = axes[row, col]
                style(ax, gtitle if row == 0 else None)
                for source in ("features", "units"):
                    color = colors[source] or GATE_COLORS[gate]
                    sel = [r for r in rows if r["gate"] == gate and r["source"] == source and r["mode"] == mode]
                    if not sel:
                        continue
                    ks = sorted({r["k"] for r in sel})
                    frac = {r["k"]: r["fraction"] for r in sel}
                    get = lambda name: [np.mean([r["recovery"][m] for r in sel if r["k"] == k and
                                                 (r["order"] == name or (name == "random" and r["order"].startswith("random")))])
                                        for k in ks]
                    x = [frac[k] for k in ks]
                    lab = "transcoder features" if source == "features" else "GRU units"
                    ax.plot(x, get("explained"), color=color, marker="o", markersize=4, linewidth=1.8,
                            label=f"{lab}: by explanation")
                    if any(r["order"] == "explained (verdict first)" for r in sel):
                        ax.plot(x, get("explained (verdict first)"), color=color, marker="s", markersize=4,
                                linestyle="-.", linewidth=1.5, label=f"{lab}: explained by any, first")
                        n_exp = next(s["n_explained"] for s in summ if s["gate"] == gate and s["source"] == source)
                        ax.axvline(n_exp / sel[0]["k"] * sel[0]["fraction"], color=color, linewidth=0.8, alpha=0.6)
                    ax.plot(x, get("oracle"), color=color, linestyle="--", linewidth=1.1, label=f"{lab}: oracle")
                    ax.plot(x, get("random"), color=color, linestyle=":", linewidth=1.1, label=f"{lab}: random")
                ax.axhline(0, color=GRID, linewidth=0.8)
                ax.axhline(1, color=GRID, linewidth=0.8)
                ax.set_xscale("log")
                ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.4, 0.9])
                ax.set_ylim(-0.6 if mode == "keep" else -0.2, 1.15)
                if row == 1:
                    ax.set_xlabel("Share of the source's components (top-k)", fontsize=9, color=INK)
                if col == 0:
                    ax.set_ylabel(f"{mname}\n(0 = all ablated, 1 = full)" if mode == "keep" else
                                  f"{mname} after removal\n(1 = unharmed)", fontsize=9, color=INK)
                s = {x["source"]: x for x in summ if x["gate"] == gate}
                txt = "  ".join(f"{src}: {s[src]['position'].get(f'{mode} {m}'):+.2f}" for src in ("features", "units")
                                if src in s and s[src]["position"].get(f"{mode} {m}") is not None)
                ax.text(0.02, 0.97, f"position random→oracle  {txt}", transform=ax.transAxes, fontsize=7.5,
                        color=INK_2, va="top")
        handles, labels = [], []
        for ax in axes.ravel():
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h); labels.append(l)
        fig.legend(handles, labels, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.955), frameon=False,
                   fontsize=8.5, labelcolor=INK_2)
        fig.suptitle(f"Budget-matched fidelity. {title}: does the explanation order pick what matters?", fontsize=13,
                     color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.03, 1, 0.89))
        paths.append(save(fig, out, f"ranked_fidelity_{mode}.png",
                          "Order: by best held-out explanation score (predicate F1 or magnitude R² beyond timing); oracle: "
                          "by single-drop behaviour loss on a separate sample; random: mean of 3 draws. Features: of "
                          "128 / ~225 / ~247 firing; units: of 48. 'Explained by any, first': final verdict explained by a "
                          "firing or magnitude explanation ranks first (vertical line: size of that set). Position: "
                          "(explained - random) / (oracle - random), mean over budgets."))
        plt.close(fig)
    return paths


def substitution(out):
    """Explanation substitution (circuit/rl_task/explanation_substitution.py)."""
    matplotlib, plt = _plt()
    path = os.path.join(FIDELITY, "substitution.json")
    if not os.path.exists(path):
        return []
    res = json.load(open(path))["results"]
    conds = [("E -> timing", "timing baseline", "#d9d8d4"), ("E -> explanation", "explanation", "#184f95"),
             ("E -> explanation, no model-output terms", "explanation without\nmodel-output terms", "#86b6ef"),
             ("E -> observed-only explanation (with belief)", "observed-only re-explanation\n(with belief)", "#eb6834")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), facecolor=SURFACE)
    for col, (gate, _, title, _) in enumerate(GATES):
        for row, (m, mname) in enumerate((("switch_agreement", "Switch recovery"), ("value_nmae", "Value recovery"))):
            ax = axes[row, col]
            style(ax, title if row == 0 else None)
            for k, source in enumerate(("features", "units")):
                r = next((x for x in res if x["gate"] == gate and x["source"] == source), None)
                if r is None:
                    continue
                for j, (c, lab, color) in enumerate(conds):
                    if c not in r["conditions"]:
                        continue
                    v = r["conditions"][c]["recovery_vs_E_mean"][m]
                    x = k * 5 + j
                    ax.bar(x, v, width=0.85, color=color, edgecolor=SURFACE, label=lab if (k == 0 and col == 0 and row == 0) else None)
                    ax.text(x, v + (0.03 if v >= 0 else -0.03), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                            fontsize=7, color=INK_2)
                ax.text(k * 5 + 1.5, -1.42, f"{'features' if source == 'features' else 'GRU units'}\n"
                        f"{r['n_explained']}/{r['n_alive']} explained\n{r['n_using_model_outputs']} read model outputs",
                        ha="center", va="top", fontsize=7.5, color=INK_2)
            ax.axhline(0, color=INK_2, linewidth=0.8)
            ax.axhline(1, color=INK_2, linewidth=0.8, linestyle="--")
            ax.set_xticks([])
            ax.set_ylim(-1.35, 1.15)
            if col == 0:
                ax.set_ylabel(f"{mname}\n(0 = explained set mean-ablated, 1 = real)", fontsize=9, color=INK)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=4, loc="upper left", bbox_to_anchor=(0.01, 0.95),
               frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle("Replace each explained component by what its explanation predicts: does behaviour survive?",
                 fontsize=13, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.9), h_pad=5)
    p = save(fig, out, "explanation_substitution.png",
             "Held-out episodes. Explained = final verdict explained by a firing or magnitude explanation; all other "
             "components stay real. Timing baseline: mean at each phase x episode decile. Model-output terms (value, "
             "choice) held at their mean in the last bar.")
    plt.close(fig)
    return [p]


USEFUL_GO = 0.05          # worst-pattern GO-choice drop that counts as useful (conditional_tests importance)


def conditional(out):
    """Conditional usefulness and clamp tests (circuit/rl_task/conditional_tests.py)."""
    matplotlib, plt = _plt()
    path = os.path.join(FIDELITY, "conditional_tests.json")
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    paths = []
    gate_map = {"reset": "reset", "update": "update", "candidate": "candidate"}
    tiers = {}
    if "clamp" in d:
        for r in d["clamp"]:
            tiers[(r["gate"], r["source"], r["id"])] = r
    if "importance" in d:
        comps = d["importance"]["components"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, facecolor=SURFACE)
        for ax, (gate, _, title, _) in zip(axes, GATES):
            style(ax, title)
            for source, color, marker in (("features", GATE_COLORS[gate], "o"), ("units", INK_2, "s")):
                x = [r for r in comps if r["gate"] == gate and r["source"] == source]
                if not x:
                    continue
                ov = np.array([max(r["mean"]["overall"]["go"], 0) for r in x])
                wo = np.array([max(r["mean"]["worst_pattern_go"][1] or 0, 0) for r in x])
                n_use = int((wo >= USEFUL_GO).sum())
                ax.scatter(ov + 1e-4, wo + 1e-4, s=12 if source == "features" else 18, color=color, marker=marker,
                           alpha=0.6, linewidth=0, label=f"{source}: {n_use} of {len(x)} useful")
            ax.plot([1e-4, 1], [1e-4, 1], color=GRID, linewidth=1)
            ax.axhline(USEFUL_GO, color=INK_2, linestyle=":", linewidth=0.9)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(1e-4, 0.5); ax.set_ylim(1e-4, 1)
            ax.set_xlabel("GO-choice agreement lost, averaged over all windows", fontsize=9, color=INK)
            ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper left")
        axes[0].set_ylabel("GO-choice agreement lost after the\nprevious-trial pattern where it hurts most", fontsize=9,
                           color=INK)
        fig.suptitle("Usefulness measured where a component takes part, not averaged over every window", fontsize=13,
                     color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        paths.append(save(fig, out, "conditional_importance.png",
                          f"Each component mean-ablated alone in the real GRU (per phase x episode-decile mean), held-out "
                          f"episodes. Useful = loses at least {USEFUL_GO:.0%} of GO choices after some previous-trial "
                          f"pattern (dotted line). Points above the diagonal are diluted by averaging."))
        plt.close(fig)
    if "substitute" in d:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, facecolor=SURFACE)
        for ax, (gate, _, title, _) in zip(axes, GATES):
            style(ax, title)
            ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
            for source, color, marker in (("features", GATE_COLORS[gate], "o"), ("units", INK_2, "s")):
                x = [r for r in d["substitute"] if r["gate"] == gate and r["source"] == source]
                if not x:
                    continue
                a = [r["ablated_go"] for r in x]
                s = [r["substituted_go"] for r in x]
                ok = sum(r["pass"] for r in x)
                ax.scatter(a, s, s=26, color=color, marker=marker, alpha=0.8, linewidth=0,
                           label=f"{source}: {ok} of {len(x)} recover >= 50%")
                for r in x:
                    if not r["pass"]:
                        ax.annotate(str(r["id"]), (r["ablated_go"], r["substituted_go"]), fontsize=7, color=INK_2,
                                    xytext=(3, -8), textcoords="offset points")
            ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
            ax.set_xlabel("GO-choice agreement with it mean-ablated", fontsize=9, color=INK)
            ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="lower right")
        axes[0].set_ylabel("... with it replaced by its explanation", fontsize=9, color=INK)
        fig.suptitle("Does each useful component's explanation carry what the model uses from it, where it is used?",
                     fontsize=13, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        paths.append(save(fig, out, "component_substitution.png",
                          "One component at a time, in the real GRU, on the GO choices after its worst previous-trial "
                          "pattern (held-out episodes). Explanation = its short form (magnitude) or predicate x mean "
                          "level (firing). Labelled points recover less than half of what ablation loses."))
        plt.close(fig)
    if "diagnose" in d:
        # useful components: does clamping them push behaviour consistently, and in the direction the situation
        # they signal naturally moves it?
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), facecolor=SURFACE)
        for ax, (k, title) in zip(axes, (("logit", "Next GO choice (logit a1 - a2)"), ("value", "Value"))):
            style(ax, title)
            labels, x0 = [], 0
            for gate, _, gtitle, _ in GATES:
                for source in ("features", "units"):
                    rs = [r for r in d["diagnose"] if r["gate"] == gate and r["source"] == source]
                    if not rs:
                        continue
                    push = [r["on_single"][f"d_{k}"] - r["off_single"][f"d_{k}"] for r in rs]
                    cons = [np.sign(r["on_single"][f"d_{k}"]) == -np.sign(r["off_single"][f"d_{k}"]) for r in rs]
                    match = [np.sign(p) == np.sign(r["natural"][k]) for p, r in zip(push, rs)]
                    color = GATE_COLORS[gate] if source == "features" else INK_2
                    ax.bar(x0, np.mean(cons), width=0.8, color=color, alpha=0.3, edgecolor=SURFACE)
                    ax.bar(x0, np.mean([c and m for c, m in zip(cons, match)]), width=0.8, color=color, edgecolor=SURFACE)
                    labels.append((x0, f"{gtitle.split()[0]}\n{source}\n(n={len(rs)})"))
                    x0 += 1
                x0 += 0.6
            ax.set_xticks([l[0] for l in labels], [l[1] for l in labels], fontsize=7.5, color=INK_2)
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        axes[0].set_ylabel("Share of useful explained components", fontsize=9, color=INK)
        fig.suptitle("Clamping a useful component: a consistent push, but often against the situation's natural trend",
                     fontsize=12.5, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        os.makedirs(os.path.join(out, "archived"), exist_ok=True)
        paths.append(save(fig, os.path.join(out, "archived"), "clamp_effects.png",   # superseded (archived/README.md)
                          "Clamp ON where its explanation says low and OFF where it says high, one step, real GRU. Light: "
                          "on and off move the readout in opposite directions (a consistent push). Dark: and that push has "
                          "the sign of the natural high-minus-low difference. Several components signal one situation with "
                          "opposite weights, so the natural trend does not predict each one's push."))
        plt.close(fig)
    return paths


NEGLIGIBLE = {("update", 144), ("update", 131), ("candidate", 153), ("candidate", 192)}   # in the matter set on 1-2% of windows
KINDS = [("firing rule only", "#184f95"), ("firing rule + one modifier", "#2a78d6"),
         ("firing rule just under 0.8 + modifier", "#86b6ef"), ("discrete magnitude only", "#eb6834")]


def explanation_kinds():
    """{gate: {kind: [ids]}} for the features that matter (rl_fidelity/matter_sets.json without the negligible ones):
    the refined explanation's kind (rl_features/rl_refined) where one exists, else the firing record."""
    matter = json.load(open(os.path.join(FIDELITY, "matter_sets.json")))
    kind_of = {"firing+modifier": KINDS[1][0], "firing~+modifier": KINDS[2][0], "magnitude": KINDS[3][0]}
    out = {}
    for gate, rgate, _, _ in GATES:
        out[gate] = {k: [] for k, _ in KINDS + [("unexplained", None)]}
        for f in matter[f"{gate}/features"]:
            if (gate, f) in NEGLIGIBLE:
                continue
            ref = os.path.join("rl_features/rl_refined", f"{rgate}_{f}.json")
            if os.path.exists(ref):
                out[gate][kind_of[json.load(open(ref))["kind"]]].append(f)
            elif json.load(open(os.path.join(FEATURES, f"{rgate}_{f}.json"))).get("final_verdict") == "explained":
                out[gate][KINDS[0][0]].append(f)
            else:
                out[gate]["unexplained"].append(f)
    return out


def breakdown(out):
    """How the features that matter are explained, and how much of a component one short rule covers (features vs units)."""
    matplotlib, plt = _plt()
    kinds = explanation_kinds()
    cov = json.load(open(os.path.join(FIDELITY, "firing_coverage.json")))
    lv_u = {(r["gate"].replace("unit_", ""), r["id"]): r for r in json.load(open(os.path.join(FIDELITY, "discrete_levels_units.json")))}
    lv_f = {({"hidden": "candidate"}.get(r["gate"], r["gate"]), r["id"]): r
            for r in json.load(open(os.path.join(FIDELITY, "discrete_levels_features.json")))}
    matter = json.load(open(os.path.join(FIDELITY, "matter_sets.json")))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor=SURFACE)
    ax = axes[0]
    style(ax, "How the features that matter are explained")
    for i, (gate, _, title, _) in enumerate(GATES):
        n = sum(len(v) for v in kinds[gate].values())
        left = 0
        for k, color in KINDS:
            c = len(kinds[gate][k])
            ax.barh(i, c / n, left=left, color=color, label=k if i == 0 else None)
            if c:
                ax.text(left + c / n / 2, i, str(c), ha="center", va="center", fontsize=9, color="white")
            left += c / n
        ax.text(1.01, i, f"n={n}", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(3), [t for _, _, t, _ in GATES]); ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.set_xlim(0, 1.12)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax = axes[1]
    style(ax, "Share of the whole activity a firing rule covers")
    w = 0.25
    for i, (gate, _, _, _) in enumerate(GATES):
        f = [r["r2_onoff"] for r in cov[f"matter/{gate}/features"] if r["r2_onoff"] is not None]
        for j, (dn, color) in enumerate((("features", "#184f95"), ("units 5%", INK_2))):
            vals = f if dn == "features" else [r["r2_onoff"] for r in cov[f"matter/{gate}/units/5%"] if r["r2_onoff"] is not None]
            ax.bar(i + (j - 0.5) * w * 1.1, np.median(vals) if vals else 0, w, color=color,
                   label=("transcoder features" if j == 0 else "GRU units (5% density)") if i == 0 else None)
            ax.text(i + (j - 0.5) * w * 1.1, (np.median(vals) if vals else 0) + 0.01, f"n={len(vals)}", ha="center",
                    fontsize=7.5, color=INK_2)
    ax.set_xticks(range(3), [t for _, _, t, _ in GATES]); ax.set_ylabel("Median R² beyond timing (on/off rule)", fontsize=9, color=INK)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2)
    ax = axes[2]
    style(ax, "Discrete levels: where both need magnitude")
    mag_f = [lv_f.get(k) for g in kinds for k in [(g, i) for i in kinds[g][KINDS[3][0]]]]
    mag_u = [lv_u.get((g, i)) for g, *_ in GATES for i in matter[f"{g}/units"]]
    tim_u = [json.load(open(f"rl_features/rl_magnitude_units/unit_{g}_{i}.json"))["test_scores"]["r2_timing_only"]
             for g, *_ in GATES for i in matter[f"{g}/units"]]
    tim_f = [json.load(open(f"rl_features/rl_magnitude/{FEATURE_REC[g]}_{i}.json"))["test_scores"]["r2_timing_only"]
             for g in kinds for i in kinds[g][KINDS[3][0]]]
    cols = [("timing share\nof variance", tim_f, tim_u), ("short form", [r["short"] for r in mag_f if r], [r["short"] for r in mag_u if r]),
            ("4 levels", [r["L4"] for r in mag_f if r], [r["L4"] for r in mag_u if r])]
    for i, (lab, fv, uv) in enumerate(cols):
        for j, (vals, color, name) in enumerate(((fv, "#eb6834", "10 discrete-magnitude features"), (uv, INK_2, "35 units that matter"))):
            ax.bar(i + (j - 0.5) * w * 1.1, np.median(vals), w, color=color, label=name if i == 0 else None)
    ax.set_xticks(range(3), [c[0] for c in cols]); ax.set_ylabel("Median (R² beyond timing; timing: R² of timing alone)", fontsize=9, color=INK)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = save(fig, out, "explanation_breakdown.png",
             "Left: refined explanations (rl_features/rl_refined) where they exist, else the firing record. Middle: the on/off "
             "firing rule applied to the component's whole value (features: their rule; units: top-5% rule). Right: "
             "magnitude short forms and their 4-level tables (circuit/rl_task/discrete_levels.py), test split.")
    plt.close(fig)
    return [p]


FEATURE_REC = {"reset": "reset", "update": "update", "candidate": "hidden"}
CLOCK_SHARE = {60: 0.1, 70: 0.2, 80: 0.35, 85: 0.5, 90: 0.75, 95: 1.0}


def clock(out):
    """The value head's late fall, and the candidate features active from trial 50 on that mirror it."""
    matplotlib, plt = _plt()
    from circuit.rl_task.autointerp import Interp
    ip = Interp(STORE)
    t, te = np.asarray(ip.cols["ep_trial"]), ip.test
    val = np.asarray(ip.cols["value"]).astype(float)
    v = np.array([val[te & (t == k)].mean() for k in range(100)])
    ids = json.load(open(os.path.join(FIDELITY, "matched_sets.json")))["candidate"]["features"]["ids"]
    b5 = t // 5
    late = []
    for f in ids:
        a = ip.activations("hidden", f).astype(float)
        prof = np.array([a[te & (b5 == k)].mean() for k in range(20)])
        if prof.max() <= 0 or np.corrcoef(np.arange(20), prof)[0, 1] <= 0.6:
            continue
        k0 = int(np.argmax(prof > 0.1 * prof.max()))
        if k0 * 5 >= 50:
            late.append((f, k0 * 5, prof / prof.max()))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), facecolor=SURFACE)
    ax = axes[0]
    style(ax, "Value head output over the episode (held-out mean)")
    ax.plot(range(100), v, color=INK)
    ax.set_xlabel("Episode trial", fontsize=9, color=INK); ax.set_ylabel("Value", fontsize=9, color=INK)
    ax = axes[1]
    style(ax, f"Candidate features active from trial 50 on (n={len(late)}): one curve, staggered onsets")
    cmap = plt.get_cmap("viridis")
    for f, k0, prof in sorted(late, key=lambda x: x[1]):
        ax.plot(np.arange(20) * 5 + 2.5, prof, color=cmap((k0 - 50) / 40), linewidth=1, alpha=0.8)
    drop = (v[:50].mean() - v) / (v[:50].mean() - v[95:].mean())
    ax.plot(range(100), np.clip(drop, 0, None), color=INK, linewidth=2.4, linestyle="--", label="value's fall (scaled 0-1)")
    ax.scatter(list(CLOCK_SHARE), list(CLOCK_SHARE.values()), color="#eb6834", zorder=5, label="'the clock' as given to simulators")
    ax.set_xlabel("Episode trial", fontsize=9, color=INK); ax.set_ylabel("Mean activation per step (scaled to its max)", fontsize=9, color=INK)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = save(fig, out, "clock.png", "Right: each line is one feature in the candidate behaviour-matched set whose activity rises over the episode "
             "and starts at trial 50 or later (colour = onset, early purple to late yellow); 5-trial bins, test split.")
    plt.close(fig)
    return [p]


def _sim(gate, comp, dirs):
    for d in dirs:
        p = os.path.join(d, f"{gate}_{comp}.json")
        if os.path.exists(p):
            return json.load(open(p))["scores"]
    return None


def tests(out):
    """Are the explanations predictive? Blind simulation, per-component substitution and the story test,
    features (old and current explanations) vs units."""
    matplotlib, plt = _plt()
    matter = json.load(open(os.path.join(FIDELITY, "matter_sets.json")))
    old_dirs, cur_dirs = ["rl_features/rl_simulation_shared"], ["rl_features/rl_simulation_clock_adopted",
                                                               "rl_features/rl_simulation_refined", "rl_features/rl_simulation_shared"]
    sub_old = {(r["gate"], r["source"], r["id"]): r for r in json.load(open(os.path.join(FIDELITY, "conditional_tests.json")))["substitute"]}
    sub_ref = {(r["gate"], r["id"]): r for r in json.load(open(os.path.join(FIDELITY, "refined_substitution_after.json")))["substitute"]}
    st_old, st_cur = (json.load(open(os.path.join(FIDELITY, n))) for n in ("story_test.json", "story_test_clock.json"))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=SURFACE)
    series = [("features, first explanations", "#86b6ef"), ("features, current explanations", "#184f95"), ("GRU units", INK_2)]
    w = 0.26
    ax = axes[0]
    style(ax, "Blind words-only simulation (100 shared windows)")
    for i, (gate, rgate, _, _) in enumerate(GATES):
        fs = [f for f in matter[f"{gate}/features"] if (gate, f) not in NEGLIGIBLE]
        vals = [[s["correlation_beyond_timing"] for s in (_sim(rgate, f, old_dirs) for f in fs) if s],
                [s["correlation_beyond_timing"] for s in (_sim(rgate, f, cur_dirs) for f in fs) if s],
                [s["correlation_beyond_timing"] for s in (_sim(f"unit_{gate}", u, old_dirs) for u in matter[f"{gate}/units"]) if s]]
        for j, ((name, color), x) in enumerate(zip(series, vals)):
            ax.bar(i + (j - 1) * w, np.median(x), w, color=color, label=name if i == 0 else None)
            ax.text(i + (j - 1) * w, np.median(x) + 0.01, f"{np.median(x):.2f}", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(range(3), [t for _, _, t, _ in GATES]); ax.set_ylim(0, 1)
    ax.set_ylabel("Median correlation beyond timing (components that matter)", fontsize=9, color=INK)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=9, frameon=False, labelcolor=INK_2, loc="upper center", ncol=3)
    ax = axes[1]
    style(ax, "Per-component substitution where it matters")
    for i, (gate, _, _, _) in enumerate(GATES):
        fs = [f for f in matter[f"{gate}/features"] if (gate, f) not in NEGLIGIBLE]
        f_old = [sub_old[(gate, "features", f)] for f in fs if (gate, "features", f) in sub_old]
        f_cur = [sub_ref.get((gate, f)) or sub_old.get((gate, "features", f)) for f in fs]
        f_cur = [r for r in f_cur if r]
        u = [sub_old[(gate, "units", x)] for x in matter[f"{gate}/units"] if (gate, "units", x) in sub_old]
        for j, ((name, color), x) in enumerate(zip(series, (f_old, f_cur, u))):
            rate = np.mean([r["pass"] for r in x]) if x else 0
            ax.bar(i + (j - 1) * w, rate, w, color=color)
            ax.text(i + (j - 1) * w, rate + 0.01, f"{sum(r['pass'] for r in x)}/{len(x)}", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(range(3), [t for _, _, t, _ in GATES]); ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Share passing (recovers >= half the ablation damage)", fontsize=9, color=INK)
    ax = axes[2]
    style(ax, "Story test: whole behaviour-matched set from words")
    labels = []
    for i, (gate, _, title, _) in enumerate(GATES):
        for k, (m, mname, _) in enumerate(METRICS):
            x = i * 3.6 + k * 1.1
            labels.append((x, f"{title.split()[0]}\n{mname.split()[0]}"))
            for j, (st, src) in enumerate(((st_old, "features"), (st_cur, "features"), (st_cur, "units"))):
                ax.bar(x + (j - 1) * 0.3, st[gate][src]["recovery"][m], 0.3, color=series[j][1])
    ax.set_xticks([x for x, _ in labels], [l for _, l in labels], fontsize=7)
    ax.set_ylabel("Recovery (0 = set at phase means, 1 = set kept)", fontsize=9, color=INK)
    ax.axhline(0, color=GRID, linewidth=0.8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    p = save(fig, out, "explanation_tests.png",
             "First explanations: the original words / short forms. Current: refined (rl_features/rl_refined) and clock-round "
             "words where adopted. Substitution covers components whose ablation hurts behaviour and that have an explanation. "
             "Story test: rl_fidelity/story_test.json (first) and story_test_clock.json (current); units unchanged.")
    plt.close(fig)
    return [p]


COMMANDS = {"interp": interp, "fidelity": fidelity, "sets": sets, "substitutes": substitutes, "probe": probe,
            "scorecards": scorecards, "time": time_profiles, "magnitude": magnitude, "ranked": ranked,
            "substitution": substitution, "conditional": conditional, "breakdown": breakdown, "clock": clock,
            "tests": tests}
# figures of abandoned analyses (docs/figures/rl/archived/README.md): written to <out>/archived, skipped by `all`
ARCHIVED = {"fidelity", "sets", "substitutes", "probe", "scorecards", "time", "ranked", "substitution"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=list(COMMANDS) + ["all"])
    ap.add_argument("--out", default="docs/figures/rl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, fn in COMMANDS.items():
        if name in ARCHIVED:
            if args.cmd == name:                 # abandoned analyses: only on request, into archived/
                os.makedirs(os.path.join(args.out, "archived"), exist_ok=True)
                fn(os.path.join(args.out, "archived"))
        elif args.cmd in (name, "all"):
            fn(args.out)


if __name__ == "__main__":
    main()
