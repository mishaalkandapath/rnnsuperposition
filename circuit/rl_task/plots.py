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
  magnitude    magnitude explanations, features vs units: timing alone, short form, free fit
  ranked       budget-matched ranked fidelity by explanation tier (abandoned analysis)
  substitution whole-set substitution by explanations (abandoned analysis)
  conditional  per-component importance and first-round substitution (its clamp figure goes to archived/)
  breakdown    how the features that matter are explained (refined), firing-rule coverage and discrete levels, vs units
  clock        the value head's late fall and the late-onset candidate features that mirror it
  tests        blind simulation, per-component substitution and story test: features (first / current) vs units
  credit       stay table, credit to the state vs the action, what h keeps; steering the credit detectors (rl_circuits/)
  circuit      the credit circuit: contrastive graph of a rewarded a1->S1 trial vs the same trial ending in S2
  reversal     switching after a reversal: behaviour, choice logit, flow of each previous outcome into the switch
  checks       minimal circuits: sufficiency and necessity of the top-k features per step vs random features
  value        the value circuit: V vs the return-to-go, what V ignores, the features carrying episode time into V
  bayes        phase portraits (logit vs logit change), agent vs ideal observer, and what separates the bands
  axis         one belief or two values: survival of an outcome's effect, S1 vs S2 evidence in the graph
  band_circuit the bands in the attribution graph: offset features, patching, history-specific detectors
  lag          the switching lag: miss readers in stick vs follow trials, through vs around the belief, patching
  swap         does the agent act on the belief information: memory-part swaps and later choices
  trace_rpe    the reward trace's share of the bands and its behavioural role; RPEs as in Wang et al.

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
    the refined explanation's kind (rl_features/rl_refined) where one exists, else the firing record, else the
    magnitude record (rl_features/rl_magnitude, observed variables only)."""
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
            elif (os.path.exists(mag := os.path.join("rl_features/rl_magnitude", f"{rgate}_{f}.json"))
                  and json.load(open(mag)).get("final_verdict") == "explained"):
                out[gate][KINDS[3][0]].append(f)      # no firing rule without model outputs (candidate 33)
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
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)          # headroom so the legend sits above the bars
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper center", ncol=2)
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
        for j, (vals, color, name) in enumerate(((fv, "#eb6834", f"{len(fv)} discrete-magnitude features"), (uv, INK_2, "35 units that matter"))):
            ax.bar(i + (j - 0.5) * w * 1.1, np.median(vals), w, color=color, label=name if i == 0 else None)
    ax.set_xticks(range(3), [c[0] for c in cols]); ax.set_ylabel("Median (R² beyond timing; timing: R² of timing alone)", fontsize=9, color=INK)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper center", ncol=2)
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


def tests(out, first=True, units=True):
    """Are the explanations predictive? Blind simulation, per-component substitution and the story test,
    features (old and current explanations) vs units. first=False: no first explanations; units=False: no units."""
    matplotlib, plt = _plt()
    matter = json.load(open(os.path.join(FIDELITY, "matter_sets.json")))
    old_dirs, cur_dirs = ["rl_features/rl_simulation_shared"], ["rl_features/rl_simulation_clock_adopted",
                                                               "rl_features/rl_simulation_refined", "rl_features/rl_simulation_shared"]
    sub_old = {(r["gate"], r["source"], r["id"]): r for r in json.load(open(os.path.join(FIDELITY, "conditional_tests.json")))["substitute"]}
    sub_ref = {(r["gate"], r["id"]): r for r in json.load(open(os.path.join(FIDELITY, "refined_substitution_after.json")))["substitute"]}
    st_old, st_cur = (json.load(open(os.path.join(FIDELITY, n))) for n in ("story_test.json", "story_test_clock.json"))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=SURFACE)
    series = [("features, first explanations", "#86b6ef"), ("features, current explanations", "#184f95"), ("GRU units", INK_2)]
    keep = [j for j, on in enumerate((first, True, units)) if on]   # which series to draw
    w = {3: 0.26, 2: 0.36, 1: 0.5}[len(keep)]
    off = lambda j: (keep.index(j) - (len(keep) - 1) / 2) * w   # bar offset of series j
    ax = axes[0]
    style(ax, "Blind words-only simulation (100 shared windows)")
    for i, (gate, rgate, _, _) in enumerate(GATES):
        fs = [f for f in matter[f"{gate}/features"] if (gate, f) not in NEGLIGIBLE]
        vals = [[s["correlation_beyond_timing"] for s in (_sim(rgate, f, old_dirs) for f in fs) if s],
                [s["correlation_beyond_timing"] for s in (_sim(rgate, f, cur_dirs) for f in fs) if s],
                [s["correlation_beyond_timing"] for s in (_sim(f"unit_{gate}", u, old_dirs) for u in matter[f"{gate}/units"]) if s]]
        for j, ((name, color), x) in enumerate(zip(series, vals)):
            if j not in keep:
                continue
            ax.bar(i + off(j), np.median(x), w, color=color, label=name if i == 0 else None)
            ax.text(i + off(j), np.median(x) + 0.01, f"{np.median(x):.2f}", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(range(3), [t for _, _, t, _ in GATES]); ax.set_ylim(0, 1)
    ax.set_ylabel("Median correlation beyond timing (components that matter)", fontsize=9, color=INK)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=9, frameon=False, labelcolor=INK_2, loc="upper center", ncol=len(keep))
    ax = axes[1]
    style(ax, "Per-component substitution where it matters")
    for i, (gate, _, _, _) in enumerate(GATES):
        fs = [f for f in matter[f"{gate}/features"] if (gate, f) not in NEGLIGIBLE]
        f_old = [sub_old[(gate, "features", f)] for f in fs if (gate, "features", f) in sub_old]
        f_cur = [sub_ref.get((gate, f)) or sub_old.get((gate, "features", f)) for f in fs]
        f_cur = [r for r in f_cur if r]
        u = [sub_old[(gate, "units", x)] for x in matter[f"{gate}/units"] if (gate, "units", x) in sub_old]
        for j, ((name, color), x) in enumerate(zip(series, (f_old, f_cur, u))):
            if j not in keep:
                continue
            rate = np.mean([r["pass"] for r in x]) if x else 0
            ax.bar(i + off(j), rate, w, color=color)
            ax.text(i + off(j), rate + 0.01, f"{sum(r['pass'] for r in x)}/{len(x)}", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(range(3), [t for _, _, t, _ in GATES]); ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Share passing (recovers >= half the ablation damage)", fontsize=9, color=INK)
    ax = axes[2]
    style(ax, "Story test: whole behaviour-matched set from words")
    labels = []
    for i, (gate, _, title, _) in enumerate(GATES):
        for k, (m, mname, _) in enumerate(METRICS):
            x = i * 3.6 + k * 1.1
            labels.append((x, f"{ {'Candidate': 'Cand.'}.get(title.split()[0], title.split()[0]) }\n{mname.split()[0]}"))
            bw = {3: 0.3, 2: 0.42, 1: 0.6}[len(keep)]
            for j, (st, src) in enumerate(((st_old, "features"), (st_cur, "features"), (st_cur, "units"))):
                if j in keep:
                    ax.bar(x + (keep.index(j) - (len(keep) - 1) / 2) * bw, st[gate][src]["recovery"][m], bw, color=series[j][1])
    ax.set_xticks([x for x, _ in labels], [l for _, l in labels], fontsize=7)
    ax.set_ylabel("Recovery (0 = set at phase means, 1 = set kept)", fontsize=9, color=INK)
    ax.axhline(0, color=GRID, linewidth=0.8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    name = "explanation_tests.png" if first else "explanation_tests_current.png" if units else "explanation_tests_features.png"
    p = save(fig, out, name,
             ("First explanations: the original words / short forms. " if first else "") +
             "Current: refined (rl_features/rl_refined) and clock-round "
             "words where adopted. Substitution covers components whose ablation hurts behaviour and that have an explanation. "
             "Story test: " + ("rl_fidelity/story_test.json (first) and " if first else "rl_fidelity/") +
             "story_test_clock.json (current)" + ("; units unchanged." if units else "."))
    plt.close(fig)
    return [p]


CIRCUITS = "rl_features/rl_circuits"
GATE_OF = {"r": "reset", "z": "update", "n": "candidate"}
# short names of the features drawn in the circuit diagram (from their firing explanations, rl_features/rl_autointerp)
SHORT = {"n:219": "a1→S1", "n:147": "a1→S2", "n:210": "a2→S1", "n:167": "a2→S2", "n:243": "a1→S2, fades",
         "n:10": "a1→S1, fades", "r:80": "a1→S2", "r:98": "D2, S2 higher", "r:10": "after a1", "r:108": "a1→S1",
         "r:120": "rewarded S1", "n:77": "rewarded a1→S1", "r:59": "rewarded S2", "n:215": "rewarded S2",
         "r:18": "a1→S2 rare", "n:8": "rewarded S2", "z:250": "after S2", "n:66": "rewarded S1",
         "n:208": "unrewarded, not S1", "n:18": "unrewarded a1→S1", "n:165": "unrewarded a1→S2",
         "z:55": "after rewarded S1", "n:121": "S1-directed choice", "z:235": "after rewarded a1→S2",
         "n:235": "after rewarded a1→S2", "z:206": "a1 after unrewarded S2", "n:146": "after unrewarded S2",
         "z:5": "after unrewarded a1→S1", "n:145": "rewarded a1→S2 rare", "n:55": "after rewarded, a1", "z:150": "", "z:134": ""}


def credit(out):
    """Model-based credit: the stay table, the factorial effects on the next choice, what h keeps, and steering."""
    matplotlib, plt = _plt()
    beh = json.load(open(os.path.join(CIRCUITS, "reversal_flow.json")))["behaviour"]
    fac = json.load(open(os.path.join(CIRCUITS, "credit_factorial.json")))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), facecolor=SURFACE)
    ax = axes[0]
    style(ax, "Behaviour: P(repeat the choice)")
    st = beh["stay_table"]
    for j, (r, color) in enumerate((("rewarded", "#2a78d6"), ("unrewarded", "#eb6834"))):
        vals = [st[f"common {r}"], st[f"rare {r}"]]
        ax.bar(np.arange(2) + (j - 0.5) * 0.36, vals, 0.34, color=color, label=f"previous trial {r}")
        for i, v in enumerate(vals):
            ax.text(i + (j - 0.5) * 0.36, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(2), ["common transition", "rare transition"]); ax.set_ylim(0, 1.25)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper center", ncol=2)
    ax = axes[1]
    style(ax, "Next-choice logit: what a reward is credited to")
    rows = [("credit_to_state", "reward × state\n(credit to the state)"), ("credit_to_action", "reward × action\n(credit to the action)"),
            ("perseveration", "action taken\n(repeat bias)"), ("reward_x_state_x_action", "reward × state × action")]
    for i, (k, name) in enumerate(rows):
        m, (lo, hi) = fac[k]["mean"], fac[k]["q10_q90"]
        ax.barh(i, m, 0.55, color="#2a78d6" if i == 0 else INK_2)
        ax.plot([lo, hi], [i, i], color=INK, linewidth=1)
        ax.text(max(m, hi) + 0.4, i, f"{m:+.1f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), [n for _, n in rows], fontsize=8.5); ax.invert_yaxis()
    ax.axvline(0, color=GRID); ax.grid(axis="x", color=GRID, linewidth=0.8); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Effect on logit a1 − a2 (mean; line = 10th–90th percentile of windows)", fontsize=8.5, color=INK)
    ax = axes[2]
    style(ax, "How much of the state / action the hidden state keeps")
    held = fac["held_in_h"]
    steps = ["c.D2", "c+1.D1", "c+1.GO"]
    for which, color in (("state", "#2a78d6"), ("action", "#eb6834")):
        v = [np.mean([held[f"{which} effect after {s}, r={r}"]["norm_mean"] for r in (1, 0)]) for s in steps]
        ax.plot(range(3), v, color=color, linewidth=2, marker="o", markersize=8, label=f"{which} swapped")
        for i, x in enumerate(v):
            ax.text(i, x + 0.08, f"{x:.2f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(3), ["after D2\n(state shown)", "after next D1\n(reward shown)", "after next GO\n(choice)"])
    ax.set_ylabel("|Δh| between the two runs", fontsize=9, color=INK); ax.set_ylim(0, 3.3)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p1 = save(fig, out, "credit_assignment.png",
              "Left: 3000 closed-loop episodes. Middle and right: 600 held-out trace windows, the core trial rerun in all 8 "
              "choice × state × reward combinations (rl_circuits/credit_factorial.json); right: mean over reward 1 and 0.")
    plt.close(fig)
    stg = json.load(open(os.path.join(CIRCUITS, "credit_steering.json")))["results"]
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=SURFACE)
    style(ax, "Steering the next-D1 candidate detectors by label (input unchanged)")
    swaps = [("state_swap", "state swap\n(other state, same reward)"), ("reward_swap", "reward swap\n(same state, other reward)"),
             ("double_swap", "double swap\n(other state and reward:\npredicted ≈ no change)")]
    bars = [("off_only", "this cell's detectors off", "#86b6ef"), ("on_only", "other cell's detectors on", "#2a78d6"),
            ("both", "both", "#184f95"), ("control", "random features (both)", "#c3c2b7"), ("real", "real input swapped", INK)]
    w = 0.16
    for i, (s, _) in enumerate(swaps):
        for j, (k, name, color) in enumerate(bars):
            key = (f"{s}/real_input" if k == "real" else f"{s}/c+1.D1/n" if k == "both" else f"{s}/c+1.D1/n/{k}")
            v = stg[key + "/shift_vs_state_swap"]["median"]
            ax.bar(i + (j - 2) * w, v, w * 0.92, color=color, label=name if i == 0 else None)
            ax.text(i + (j - 2) * w, v + (0.02 if v >= 0 else -0.08), f"{v:.2f}", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(range(3), [n for _, n in swaps], fontsize=8.5)
    ax.axhline(0, color=GRID); ax.set_ylim(-0.15, 1.25)
    ax.set_ylabel("Shift of the next choice\n(units of a real state swap's effect, median)", fontsize=9, color=INK)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper right", ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p2 = save(fig, out, "credit_steering.png",
              "396 held-out windows (selection on another 400). Detectors: features at the next D1 at least 3× more active in one "
              "(state, reward) cell than in any other. Steered only at the next D1, candidate features. All bars in units of the "
              "real state swap (per-window ratio, median). Compared window by window with the real reward swap itself, "
              "the reward steering ('both') recovers 0.90 of it (rl_circuits/credit_steering.json).")
    plt.close(fig)
    return [p1, p2]


def circuit(out):
    """The credit circuit for one pattern: rewarded a1→S1 vs the same trial ending in S2 (state contrast)."""
    matplotlib, plt = _plt()
    from matplotlib.lines import Line2D
    g = json.load(open(os.path.join(CIRCUITS, "contrast_flow.json")))["contrasts"]["state"]["a1->S1 r=1"]
    steps = [("c.D2", "D2: state shown\n(S1 vs S2)"), ("c+1.D1", "next D1: reward shown"), ("c+1.GO", "next GO: choice")]
    per = {"c.D2": 6, "c+1.D1": 8, "c+1.GO": 5}
    pos, nodes = {}, []
    for i, (s, _) in enumerate(steps):
        top = g["top_features_by_step"][s][:per[s]]
        for j, f in enumerate(top):
            y = 1 - j / (per[s] - 1) if per[s] > 1 else 0.5
            pos[f"{s}/{f['feature']}"] = (i, y)
            nodes.append((s, f))
    fig, ax = plt.subplots(figsize=(15, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE); ax.axis("off")
    emax = max(abs(v) for v in g["top_edges"].values())
    for e, v in sorted(g["top_edges"].items(), key=lambda kv: abs(kv[1])):
        a, b = e.split(" -> ")
        if a in pos and b in pos and abs(v) >= 0.03 * emax and pos[a][0] < pos[b][0]:
            (x0, y0), (x1, y1) = pos[a], pos[b]
            ax.plot([x0 + 0.52, x1 - 0.04], [y0, y1], color=INK if v > 0 else INK_2, alpha=0.75,
                    linewidth=0.6 + 5 * abs(v) / emax, linestyle="-" if v > 0 else "--", zorder=1)
    fmax = max(abs(f["flow"]) for _, f in nodes)
    for s, f in nodes:
        x, y = pos[f"{s}/{f['feature']}"]
        k = f["feature"]
        ax.scatter([x], [y], s=120 + 1400 * abs(f["flow"]) / fmax, color=GATE_COLORS[GATE_OF[k[0]]],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        name = {"r": "reset", "z": "update", "n": "cand."}[k[0]] + " " + k[2:]
        ax.text(x + 0.07, y, f"{name}  {SHORT.get(k, '')}\nflow {f['flow']:+.1f}", fontsize=8, color=INK, va="center", zorder=4)
    for i, (_, t) in enumerate(steps):
        ax.text(i, 1.13, t, ha="center", fontsize=10.5, color=INK)
    ax.text(3.25, 0.5, f"next choice\nΔ logit a1−a2\n= {g['d_target_mean']:+.1f}", fontsize=10.5, color=INK, va="center", ha="center")
    ax.annotate("", xy=(3.02, 0.5), xytext=(2.72, 0.5), arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_xlim(-0.15, 3.45); ax.set_ylim(-0.12, 1.22)
    handles = [Line2D([], [], marker="o", linestyle="", color=GATE_COLORS[gt], markersize=9, label=f"{gt} gate feature")
               for gt in ("reset", "update", "candidate")]
    handles += [Line2D([], [], color=INK, label="edge pushes towards a1 (flow > 0)"),
                Line2D([], [], color=INK_2, linestyle="--", label="edge pushes towards a2")]
    ax.legend(handles=handles, fontsize=8, frameon=False, labelcolor=INK_2, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    p = save(fig, out, "credit_circuit.png",
             "Contrastive graph, mean over 300 held-out windows whose core trial was a rewarded a1→S1, against the same windows with "
             "the state swapped to S2 (rl_circuits/contrast_flow.json). Node size = |flow| (its total effect on the change); "
             "features of all three gates, top features per step, edges between steps; opposite pairs at a step largely cancel.")
    plt.close(fig)
    return [p]


def reversal(out):
    """Switching after a reversal: behaviour, the choice logit, and where the switch's evidence comes from."""
    matplotlib, plt = _plt()
    d = json.load(open(os.path.join(CIRCUITS, "reversal_flow.json")))
    beh = d["behaviour"]
    ks = sorted(int(k) for k in beh["p_new_action_by_trials_since_reversal"])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), facecolor=SURFACE)
    ax = axes[0]
    style(ax, "P(choose the new good action)")
    for who, color, name in (("agent", "#2a78d6", "agent"), ("ideal_observer", INK_2, "ideal observer (greedy)")):
        ax.plot(ks, [beh["p_new_action_by_trials_since_reversal"][str(k)][who] for k in ks], color=color, linewidth=2, label=name)
    ax.axvline(0, color=GRID, linestyle=":")
    ax.set_xlabel("Trials since the reversal", fontsize=9, color=INK); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="lower right")
    ax = axes[1]
    style(ax, "Agent's choice logit, towards the new action")
    lg = beh["choice_logit_towards_new_action"]
    m = [lg[str(k)]["mean"] for k in ks]
    ax.fill_between(ks, [lg[str(k)]["q10_q90"][0] for k in ks], [lg[str(k)]["q10_q90"][1] for k in ks], color="#86b6ef", alpha=0.5, linewidth=0)
    ax.plot(ks, m, color="#2a78d6", linewidth=2)
    ax.axhline(0, color=INK_2, linewidth=0.8); ax.axvline(0, color=GRID, linestyle=":")
    ax.set_xlabel("Trials since the reversal (band: 10th–90th percentile)", fontsize=9, color=INK)
    for a in axes[:2]:
        a.set_xticks(range(-5, 15, 5))
    ax = axes[2]
    style(ax, "Where the switch comes from (600 switches)")
    src = d["old_world"]["flow_by_source"]
    lab = ["S−1", "S−2", "S−3", "S−4", "S−5"]
    vals = [src.get(f"input@{'S' if k == 0 else f'S-{k}'}.D1", 0.0) for k in range(5)]
    ax.bar(range(5), vals, 0.6, color="#2a78d6")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(5), lab)
    ax.set_xlabel("Outcome of trial (S = the switch trial)", fontsize=9, color=INK)
    ax.set_ylabel("Flow from that trial's outcome (logit)", fontsize=9, color=INK)
    tot = d["old_world"]["d_target_mean"]
    ax.text(4.3, max(vals) * 0.97,
            f"logit change, all 5 outcomes replaced: {tot:.1f}\n(bars sum to {sum(vals):.1f}; transcoder errors {tot - sum(vals):+.1f})\n"
            f"logit change, only S−1's outcome replaced: {d['last']['d_target_mean']:.1f}\n(a separate rerun; not additive)",
            ha="right", va="top", fontsize=8, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = save(fig, out, "reversal_switch.png",
             "3000 closed-loop episodes, reversals after which the agent had settled. Right: each switch rerun with the outcomes of "
             "the 5 trials before replaced by those the old belief expects; exact flow from each trial's reward input to the switch "
             "trial's choice (rl_circuits/reversal_flow.json).")
    plt.close(fig)
    return [p]


def checks(out):
    """Minimal circuits: sufficiency and necessity of each window's top-k features per step."""
    matplotlib, plt = _plt()
    d = json.load(open(os.path.join(CIRCUITS, "circuit_checks.json")))
    ks = (1, 3, 5, 10, 20)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), facecolor=SURFACE, sharey=True)
    names = {"state": "State reached swapped → next choice", "reward": "Reward swapped → next choice",
             "reversal": "Rewards of the 5 trials before the switch\nset to the old world's → the switch"}
    modes = [("sufficiency_free", "sufficiency, free", "#2a78d6", "-"), ("sufficiency_clamped", "sufficiency, clamped", "#2a78d6", "--"),
             ("necessity_free", "necessity, free", "#eb6834", "-"), ("necessity_clamped", "necessity, clamped", "#eb6834", "--")]
    for ax, c in zip(axes, ("state", "reward", "reversal")):
        style(ax, f"{names[c]} (n={d[c]['n']})")
        for m, name, color, ls in modes:
            ax.plot(range(5), [min(d[c][f"top{k}"][m + "_share"], 1.3) for k in ks], color=color, linestyle=ls, linewidth=2,
                    marker="o", markersize=7, markerfacecolor=color if ls == "-" else "white", markeredgewidth=1.8,
                    label=name + (" (solid, filled)" if ls == "-" else " (dashed, hollow)"))
        for m, ls in (("sufficiency_free", "-"), ("necessity_free", "--")):
            ax.plot(range(5), [d[c][f"random{k}"][m + "_share"] for k in ks], color="#c3c2b7", linestyle=ls, linewidth=1.6,
                    label=f"random features, {m.split('_')[0]}, free" + (" (solid)" if ls == "-" else " (dashed)"))
        ax.set_xticks(range(5), [str(k) for k in ks]); ax.set_xlabel("Features per step (top by |flow|)", fontsize=9, color=INK)
        ax.axhline(1, color=GRID, linewidth=0.8)
    axes[0].set_ylabel("Share of the choice change carried (median)", fontsize=9, color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False, labelcolor=INK_2, loc="upper center",
               ncol=3, handlelength=4.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    p = save(fig, out, "circuit_checks.png",
             "Real GRU; patched gates are transcoder reconstructions plus the patched run's own error, kept in the gate's range. "
             "Free: the rest recomputed; clamped: every other feature held. Right: each switch trial S rerun with the rewards of "
             "trials S−5..S−1 replaced by what the old belief expects (1 in the old good state, 0 in the old bad one; on average "
             "2.85 of the 5 actually differ, S−1 always); target the choice at S. Values above 1.3 drawn at 1.3 (single features that "
             "overshoot when their cancelling partner is not patched). rl_circuits/circuit_checks.json.")
    plt.close(fig)
    return [p]


def value(out):
    """The value circuit: V vs the return-to-go, what V ignores, and the features carrying episode time into V."""
    matplotlib, plt = _plt()
    d = json.load(open(os.path.join(CIRCUITS, "value_circuit.json")))
    w = d["what"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), facecolor=SURFACE)
    ax = axes[0]
    style(ax, "Value head vs discounted return-to-go (γ = 0.97)")
    ax.plot(range(100), w["V_by_trial"], color="#2a78d6", linewidth=2, label="value output V")
    ax.plot(range(100), w["G_by_trial"], color=INK_2, linewidth=2, linestyle="--", label="actual return-to-go G (own policy)")
    ax.set_xlabel("Episode trial", fontsize=9, color=INK); ax.set_ylim(0, 8.5)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="lower left")
    ax = axes[1]
    style(ax, "What V reacts to beyond time")
    ks = sorted(int(k) for k in w["around_reversals_beyond_time"])
    ax.plot(ks, [w["around_reversals_beyond_time"][str(k)]["V"] for k in ks], color="#2a78d6", linewidth=2, marker="o",
            markersize=7, label="V around a reversal")
    ax.plot(ks, [w["around_reversals_beyond_time"][str(k)]["G"] for k in ks], color=INK_2, linewidth=2, linestyle="--",
            label="G around a reversal (true)")
    d2 = w["at_D2_by_state_reached"]
    ax.axhline(d2["believed-good state"]["V_beyond_time_at_D2"], color="#1baf7a", linewidth=1.6, label="V at D2, believed-good state reached")
    ax.axhline(d2["believed-bad state"]["V_beyond_time_at_D2"], color="#eb6834", linewidth=1.6, label="V at D2, believed-bad state reached")
    ax.axhline(0, color=GRID)
    ax.set_xlabel("Trials since a reversal", fontsize=9, color=INK)
    ax.set_ylabel("Beyond the time profile", fontsize=9, color=INK)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="lower right")
    ax = axes[2]
    t = d["time"]["value"]
    style(ax, f"Late vs early episode, same recent trials: ΔV = {t['d_target_mean']:.2f}")
    feats = list(t["features_by_total_flow"].items())[:10]
    for i, (k, v) in enumerate(feats):
        ax.barh(i, v, 0.6, color=GATE_COLORS[GATE_OF[k[0]]])
        ax.text(v - 0.05 if v < 0 else v + 0.05, i, f"{v:+.2f}", va="center", ha="right" if v < 0 else "left", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(feats)), [{"r": "reset ", "z": "update ", "n": "cand. "}[k[0]] + k[2:] for k, _ in feats], fontsize=8.5)
    ax.invert_yaxis(); ax.axvline(0, color=GRID); ax.grid(axis="x", color=GRID, linewidth=0.8); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Flow into the value at the next GO (summed over steps)", fontsize=9, color=INK)
    ax.set_xlim(min(v for _, v in feats) * 1.25, 0.4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = save(fig, out, "value_circuit.png",
             "3000 closed-loop episodes (rl_circuits/value_circuit.json). Right: 400 pairs of a late (trials 80-95) and an early "
             "(30-50) carried state with the same good state and the same previous 3 trials, run on the same inputs; bars = features "
             "(colour = gate) carrying the value difference.")
    plt.close(fig)
    return [p]


def bayes(out):
    """Phase portraits (Ji-An et al. 2025 style): the agent vs the ideal observer, and what the bands are."""
    matplotlib, plt = _plt()
    from matplotlib.lines import Line2D
    d = json.load(open(os.path.join(CIRCUITS, "bayes_vs_mb.json")))
    P = {k: np.asarray(v) for k, v in d["portrait"]["points"].items()}
    types = ["S1 R=1", "S1 R=0", "S2 R=1", "S2 R=0"]
    tcol = {0: "#184f95", 1: "#86b6ef", 2: "#a8381b", 3: "#f2a07e"}     # S1 dark / light blue, S2 dark / light orange
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), facecolor=SURFACE)
    for ax, (x, y, title) in zip(axes[0], ((P["L"], P["dL"], "Our agent: logit vs logit change"),
                                           (P["observer_belief_logit"], P["observer_dlam"], "Ideal observer (belief logit), same trials"))):
        style(ax, title)
        for t in range(4):
            k = P["typ"] == t
            ax.scatter(x[k], y[k], s=4, color=tcol[t], alpha=0.35, linewidths=0)
        ax.axhline(0, color=GRID); ax.axvline(0, color=GRID)
        ax.set_xlabel("Logit L(t)", fontsize=9, color=INK); ax.set_ylabel("Logit change L(t+1) − L(t)", fontsize=9, color=INK)
    axes[0][0].legend(handles=[Line2D([], [], marker="o", linestyle="", color=tcol[t], markersize=7, label=types[t]) for t in range(4)],
                      fontsize=8, frameon=False, labelcolor=INK_2, loc="lower left", ncol=2)
    ax = axes[1][0]
    style(ax, "One input type (S1 R=1): its bands are the previous trial")
    pcol = {0: "#184f95", 1: "#86b6ef", 2: "#a8381b", 3: "#f2a07e"}
    for pt in range(4):
        k = (P["typ"] == 0) & (P["prev_typ"] == pt)
        ax.scatter(P["L"][k], P["dL"][k], s=6, color=pcol[pt], alpha=0.5, linewidths=0, label=f"previous trial {types[pt]}")
    ax.axhline(0, color=GRID); ax.axvline(0, color=GRID)
    ax.set_xlabel("Logit L(t)", fontsize=9, color=INK); ax.set_ylabel("Logit change", fontsize=9, color=INK)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, markerscale=2, loc="upper right")
    ax = axes[1][1]
    style(ax, "What separates the bands (share of the spread around the curves)")
    add = d["bands"]["added"]
    rows = [("previous input type and transition", "previous input type\n+ transition"), ("previous input type", "previous input type"),
            ("previous reward only", "previous reward only"), ("MB-RL Q(S1) + Q(S2) beyond the belief (quintiles)", "MB-RL Q(S1)+Q(S2)\nbeyond the belief"),
            ("previous state only", "previous state only"), ("input type two trials back", "input type 2 back"),
            ("episode time (decile)", "episode time"), ("current transition", "current transition")]
    go = d["go_split"]["bands_explained"]["GO write beyond L (quintiles)"]
    vals = [add[k]["share_of_spread_explained"] for k, _ in rows] + [go]
    labels = [n for _, n in rows] + ["GO write beyond L\n(circuit variable)"]
    for i, v in enumerate(vals):
        ax.barh(i, v, 0.6, color="#1baf7a" if i == len(vals) - 1 else "#2a78d6")
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(vals)), labels, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = save(fig, out, "phase_portrait.png",
             f"3000 closed-loop episodes, 6000 sampled trials (rl_circuits/bayes_vs_mb.json). A curve per input type explains "
             f"{d['bands']['r2_curve_per_input_type']:.2f} of the logit change; bottom right: share of the rest each history variable explains.")
    plt.close(fig)
    return [p]


def axis(out):
    """One belief or two values: survival of an outcome's effect, and S1 vs S2 evidence in the graph."""
    matplotlib, plt = _plt()
    d = json.load(open(os.path.join(CIRCUITS, "belief_axis.json")))
    sv = d["survival_median"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), facecolor=SURFACE)
    ax = axes[0]
    style(ax, "How much of an outcome's effect survives the next trial")
    cells = [("same state, common transition, confirming", "same state\nconfirming"),
             ("other state, common transition, confirming", "other state\nconfirming"),
             ("same state, common transition, opposing", "same state\nopposing"),
             ("other state, common transition, opposing", "other state\nopposing"),
             ("other state, rare transition", "other state\nby a rare transition")]
    who = [("agent", "agent", "#2a78d6"), ("ideal observer", "ideal observer", INK_2), ("fitted Bayesian", "fitted Bayesian", "#1baf7a"),
           ("fitted MB-RL", "fitted MB-RL (two values)", "#eb6834")]
    wd = 0.2
    for j, (k, name, color) in enumerate(who):
        vals = [sv[k][c]["median"] for c, _ in cells]
        ax.bar(np.arange(len(cells)) + (j - 1.5) * wd, vals, wd * 0.92, color=color, label=name)
    ax.set_xticks(range(len(cells)), [n for _, n in cells], fontsize=8)
    ax.set_ylabel("Effect on the choice one trial later / right after", fontsize=9, color=INK)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper left")
    ax = axes[1]
    style(ax, "S1 vs S2 evidence on the way to the choice two trials later")
    steps = list(d["graph"])
    labels = ["reward read\n(next D1)", "next GO", "next D2", "D1 after", "GO after\n(target)"]
    ax.plot(range(len(steps)), [d["graph"][s]["flow_vector_cosine_S1_vs_S2_evidence"] for s in steps], color="#2a78d6",
            linewidth=2, marker="o", markersize=8, label="same features? (cosine of the flow vectors)")
    ax.plot(range(len(steps)), [d["graph"][s]["dh_rest_cosine_oriented_by_reward"] for s in steps], color=INK_2,
            linewidth=2, linestyle="--", marker="o", markersize=7, label="rest of the change in h, aligned by reward")
    ax.axhline(0, color=GRID)
    ax.set_xticks(range(len(steps)), labels, fontsize=8); ax.set_ylim(-0.4, 1)
    ax.set_ylabel("Cosine between S1-outcome and S2-outcome evidence", fontsize=9, color=INK)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="lower right")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = save(fig, out, "belief_axis.png",
             "3000 events from closed-loop episodes: one trial's reward swapped (rl_circuits/belief_axis.json). Left: medians; "
             "'confirming' = the next trial's outcome is evidence for the same state. Right: exact attribution graphs to the choice at "
             "the GO two trials later.")
    plt.close(fig)
    return [p]


FEAT_NAME = lambda k: {"r": "reset", "z": "update", "n": "cand."}[k[0]] + " " + k[2:]


def band_circuit(out):
    """The bands in the attribution graph: the features carrying the band offset, patching, history-specific
    detectors at matched logit."""
    matplotlib, plt = _plt()
    d = json.load(open(os.path.join(CIRCUITS, "band_flow.json")))
    g = d["contrasts"]["prev_reward"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.6, 1, 1]})
    ax = axes[0]
    style(ax, "Features carrying the band offset (previous reward swapped)")
    steps = ["t.D1", "t.D2", "t.GO", "t+1.D1", "t+1.GO"]
    names = {"t.D1": "D1 of trial t\n(reads trial t−1's reward)", "t.D2": "D2 of trial t\n(its state shown)",
             "t.GO": "GO of trial t", "t+1.D1": "next D1\n(reads trial t's reward)", "t+1.GO": "next GO"}
    feats = [(k.split("/")[0], k.split("/")[1], v) for k, v in g["top_offset_features"].items()]
    fmax = max(abs(v) for _, _, v in feats)
    per = {s: [f for f in feats if f[0] == s] for s in steps}
    for i, s in enumerate(steps):
        for j, (_, k, v) in enumerate(sorted(per[s], key=lambda f: -abs(f[2]))):
            y = 1 - j * 0.16
            ax.scatter([1.4 * i], [y], s=80 + 900 * abs(v) / fmax, color=GATE_COLORS[GATE_OF[k[0]]], edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(1.4 * i + 0.17, y, f"{FEAT_NAME(k)}  {v:+.1f}", fontsize=7.5, color=INK, va="center")
        ax.text(1.4 * i + 0.3, 1.2, names[s], ha="center", fontsize=8, color=INK_2)
    ax.set_xlim(-0.4, 6.9); ax.set_ylim(-0.25, 1.35); ax.axis("off")
    ax = axes[1]
    style(ax, "Patching the top features per step carries the band")
    ks = [1, 3, 5, 10]
    pat = g["patching_offset_share_median"]
    for mode, ls in (("sufficiency", "-"), ("necessity", "--")):
        ax.plot(range(4), [pat[f"top{k}/{mode}"] for k in ks], color="#2a78d6", linestyle=ls, linewidth=2, marker="o",
                markersize=7, label=f"top features, {mode}")
        ax.plot(range(4), [pat[f"random{k}/{mode}"] for k in ks], color="#c3c2b7", linestyle=ls, linewidth=1.6,
                label=f"random features, {mode}")
    ax.axhline(1, color=GRID); ax.axhline(0, color=GRID)
    ax.set_xticks(range(4), [str(k) for k in ks]); ax.set_xlabel("Features patched per step", fontsize=9, color=INK)
    ax.set_ylabel("Share of the band offset carried (median)", fontsize=9, color=INK)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="center right")
    ax = axes[2]
    m = d["matched_logit"]["S1 R=1, logit 6 to 10"]
    style(ax, "Same logit, same new outcome (rewarded S1)")
    prevs = [("previous S1 R=1", "trial before:\nrewarded S1", "#184f95"), ("previous S2 R=0", "trial before:\nunrewarded S2", "#f2a07e")]
    dets = [("77", "cand. 77\n'rewarded a1→S1'"), ("118", "cand. 118\n'rewarded a1→S1'")]
    w = 0.36
    for j, (pk, lab, color) in enumerate(prevs):
        vals = [m[pk]["detectors_next_D1"][f] for f, _ in dets]
        ax.bar(np.arange(2) + (j - 0.5) * w, vals, w * 0.92, color=color,
               label=f"{lab.replace(chr(10), ' ')} (logit {m[pk]['logit']:.1f}; next change {m[pk]['next_change']:+.2f})")
    ax.set_xticks(range(2), [n for _, n in dets], fontsize=8)
    ax.set_ylabel("Activation at the next D1 (reading the new outcome)", fontsize=9, color=INK)
    ax.set_ylim(0, 2.8)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="upper right")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = save(fig, out, "band_circuit.png",
             "rl_circuits/band_flow.json, 400 closed-loop events: the same window with trial t−1's reward swapped, trial t's input kept. "
             "Left: top features by flow into the band offset (exact attribution graph; colour = gate; opposite pairs largely cancel). "
             "Right: feature activations at matched logit (both previous outcomes are evidence for S1).")
    plt.close(fig)
    return [p]


def lag_circuit(out):
    """The switching lag: miss readers in stick vs follow trials, through-belief vs around-belief, patching."""
    matplotlib, plt = _plt()
    d = json.load(open(os.path.join(CIRCUITS, "choice_lag.json")))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=SURFACE)
    ax = axes[0]
    style(ax, "Readers of a missed reward: stick vs follow trials")
    act = d["graph"]["d1_feature_activations_mean"]
    order = [("n:208", "cand. 208\n'after an unrewarded\ntrial' (switch)"), ("n:11", "cand. 11\n'after an unrewarded trial,\nfades late' (switch)"),
             ("r:4", "reset 4\n'a2→S2, stronger\nunrewarded' (switch)"), ("n:220", "cand. 220\n'unrewarded a2→S2,\nafter runs of a2' (stay)")]
    w = 0.2
    cols = [("reads trial t-2's outcome (t-1.D1)", "stick", "#eb6834", "stick, 1st miss"), ("reads trial t-2's outcome (t-1.D1)", "follow", "#2a78d6", "follow, 1st miss"),
            ("reads trial t-1's outcome (t.D1)", "stick", "#f2a07e", "stick, 2nd miss"), ("reads trial t-1's outcome (t.D1)", "follow", "#86b6ef", "follow, 2nd miss")]
    for j, (when, grp, color, lab) in enumerate(cols):
        ax.bar(np.arange(4) + (j - 1.5) * w, [act[f][when][grp] for f, _ in order], w * 0.92, color=color, label=lab)
    ax.set_xticks(range(4), [n for _, n in order], fontsize=7.5)
    ax.set_ylabel("Mean activation", fontsize=9, color=INK)
    ax.set_ylim(0, 2.6)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="upper left", ncol=2)
    ax = axes[1]
    sp = d["split"]
    style(ax, "Through the belief or around it?")
    parts = [("belief subspace", "belief\ndirections"), ("choice read-out (outside the belief subspace)", "choice read-out\ndirection"),
             ("the rest", "the rest")]
    tot = sp["graph_flow_of_each_part_mean"]["whole"]
    gv = [sp["graph_flow_of_each_part_mean"][k] / tot for k, _ in parts]
    sv = [sp["swap_share_of_difference_removed_median"][k] for k, _ in parts]
    ax.bar(np.arange(3) - 0.19, gv, 0.36, color="#2a78d6", label="attribution graph: share of the flow (from the carried-in difference)")
    ax.bar(np.arange(3) + 0.19, sv, 0.36, color="#eb6834", label="real network: share removed by swapping that part (just before GO)")
    for i in range(3):
        ax.text(i - 0.19, gv[i] + 0.02, f"{gv[i]:.2f}", ha="center", fontsize=8, color=INK_2)
        ax.text(i + 0.19, sv[i] + 0.02, f"{sv[i]:.2f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(3), [n for _, n in parts], fontsize=8.5); ax.set_ylim(0, 1)
    ax.set_ylabel("Share of the stick-vs-follow difference", fontsize=9, color=INK)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="upper right")
    ax = axes[2]
    style(ax, "Patching the top features per step")
    pat = d["patching_share_of_difference_carried_median"]
    ks = [1, 3, 5, 10]
    ax.plot(range(4), [pat[f"top{k}"] for k in ks], color="#2a78d6", linewidth=2, marker="o", markersize=7, label="top features by flow")
    ax.plot(range(4), [pat[f"random{k}"] for k in ks], color="#c3c2b7", linewidth=1.6, marker="o", markersize=6, label="random changed features")
    ax.axhline(1, color=GRID); ax.set_ylim(-0.05, 1.3)
    ax.set_xticks(range(4), [str(k) for k in ks]); ax.set_xlabel("Features patched per step (from the follow run)", fontsize=9, color=INK)
    ax.set_ylabel("Share of the difference carried (median)", fontsize=9, color=INK)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = save(fig, out, "lag_circuit.png",
             "rl_circuits/choice_lag.json, 400 pairs. Stick trial: the choice repeats the previous action against the belief information; "
             "follow trial: the choice agrees with it; same two previous trials and belief, run on identical inputs from their own memory. "
             "Typical case: two missed rewards on a2→S2.")
    plt.close(fig)
    return [p]


def belief_swap_fig(out):
    """Does the agent act on the belief information? Swapping parts of the memory between runs."""
    matplotlib, plt = _plt()
    d = json.load(open(os.path.join(CIRCUITS, "belief_swap.json")))
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), facecolor=SURFACE)
    steps = ["t.GO", "(t+1).GO", "(t+2).GO"]
    parts = [("belief subspace", "belief directions", "#2a78d6"), ("choice read-out (outside the belief subspace)", "choice read-out direction", "#eb6834"),
             ("random", "random directions (mean of 3)", "#c3c2b7")]
    for ax, moment in zip(axes, ("entering t.D1", "entering t.GO")):
        style(ax, f"Swapped just before {'D1' if 'D1' in moment else 'GO'}")
        r = d[moment]
        w = 0.26
        for j, (k, lab, color) in enumerate(parts):
            if k == "random":
                vals = [np.mean([r[f"random subspace {i}"][f"share of the whole-state swap at {s}"] for i in range(3)]) for s in steps]
            else:
                vals = [r[k][f"share of the whole-state swap at {s}"] for s in steps]
            ax.bar(np.arange(3) + (j - 1) * w, vals, w * 0.92, color=color, label=lab)
            for i, v in enumerate(vals):
                ax.text(i + (j - 1) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5, color=INK_2)
        ax.set_xticks(range(3), ["this trial's choice", "next trial's choice", "the one after"])
        ax.set_ylim(0, 0.8)
        ax.set_ylabel("Change in the choice, as a share of\nswapping the whole memory (median)", fontsize=9, color=INK)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper right")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p = save(fig, out, "belief_swap.png",
             f"rl_circuits/belief_swap.json, {d['n_pairs']} pairs of memory states from different episodes with the same two previous trials and ideal "
             "beliefs at least 2 apart; one part of the donor's memory given to the recipient, which then runs on its own inputs (real GRU).")
    plt.close(fig)
    return [p]


def trace_rpe(out):
    """The reward trace's share of the bands and its (lack of) behavioural role; RPEs as in Wang et al."""
    matplotlib, plt = _plt()
    rt = json.load(open(os.path.join(CIRCUITS, "rare_and_trace.json")))["trace"]
    tr = json.load(open(os.path.join(CIRCUITS, "trace_role.json")))
    rp = json.load(open(os.path.join(CIRCUITS, "rpe_analysis.json")))
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), facecolor=SURFACE)
    ax = axes[0]
    style(ax, "How much of the bands the reward trace carries")
    dims = [1, 3, 5]
    plain = [rt["results"][f"trace/{k}d"]["offset_removed_share_median"] for k in dims]
    prot = [rt["preference_protected"][f"trace/{k}d"]["offset_removed_share_median"] for k in dims]
    rnd = [np.mean([rt["preference_protected"][f"random_{j}/{k}d"]["offset_removed_share_median"] for j in range(5)]) for k in dims]
    w = 0.26
    for j, (vals, lab, color) in enumerate(((plain, "whole trace difference removed", "#86b6ef"),
                                            (prot, "only the part the action preference\ncannot explain removed", "#184f95"),
                                            (rnd, "random directions (same method, mean of 5)", "#c3c2b7"))):
        ax.bar(np.arange(3) + (j - 1) * w, vals, w * 0.92, color=color, label=lab)
    ax.set_xticks(range(3), [f"{k} direction{'s' if k > 1 else ''}" for k in dims])
    ax.set_ylabel("Band offset removed (median per event)", fontsize=9, color=INK); ax.set_ylim(-0.05, 1)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="upper left")
    ax = axes[1]
    style(ax, "Switching after a reversal without the trace")
    ks = [str(k) for k in range(1, 7)]
    for name, r, color, ls in (("intact", tr["intact"], INK, "-"), ("trace removed (plain)", tr["ablated"]["trace"], "#86b6ef", "--"),
                               ("trace removed (clean: belief-predicted part kept)", tr["conditional"]["trace"], "#184f95", "-")):
        ax.plot(range(1, 7), [r["p_new_action_after_reversal"][k] for k in ks], color=color, linestyle=ls, linewidth=2,
                marker="o", markersize=6, label=name)
    ax.set_xlabel("Trials since the reversal", fontsize=9, color=INK)
    ax.set_ylabel("P(choosing the newly good action)", fontsize=9, color=INK); ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="lower right")
    ax = axes[2]
    style(ax, "RPEs as in Wang et al.: correlation with the agent's")
    refs = [("bayesian", "Bayesian\n(Wang's 'model-based')"), ("mb_rl", "model-based RL\n(Daw)"), ("model_free", "model-free\n(SARSA)"),
            ("reward_only", "reward only\n(no learning)")]
    c = rp["correlation_with_agent_rpe"]
    vals = [c[k]["pooled (reveal and outcome)"] for k, _ in refs]
    ax.bar(range(4), vals, 0.6, color=["#2a78d6", "#2a78d6", "#2a78d6", "#c3c2b7"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(4), [n for _, n in refs], fontsize=8); ax.set_ylim(0, 1)
    ax.set_ylabel("Correlation (state-reveal and reward steps)", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = save(fig, out, "trace_rpe.png",
             "Left: rl_circuits/rare_and_trace.json (band contrast, removal after the current choice). Middle: rl_circuits/trace_role.json, "
             "3000 closed-loop episodes. Right: rl_circuits/rpe_analysis.json; the references' RPEs correlate 0.85–0.91 with each other.")
    plt.close(fig)
    return [p]


COMMANDS = {"interp": interp, "fidelity": fidelity, "sets": sets, "substitutes": substitutes, "probe": probe,
            "scorecards": scorecards, "time": time_profiles, "magnitude": magnitude, "ranked": ranked,
            "substitution": substitution, "conditional": conditional, "breakdown": breakdown, "clock": clock,
            "tests_current": lambda out: tests(out, first=False),
            "tests_features": lambda out: tests(out, first=False, units=False),
            "tests": tests, "credit": credit, "circuit": circuit, "reversal": reversal, "checks": checks, "value": value, "bayes": bayes, "axis": axis, "band_circuit": band_circuit, "lag": lag_circuit, "swap": belief_swap_fig, "trace_rpe": trace_rpe}
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
