"""Firing explanations of units (binarised at 1% / matched / 5% density) vs features, on the fair sets.
Per component: is a firing rule explained (final verdict 'explained'), its test F1, and how much of the component's
whole activity (R^2 beyond timing of the full value, standard fit) the on/off rule accounts for.
Sets: the components that matter (rl_fidelity/matter_sets.json, without the 4 negligible features) and the
behaviour-matched sets (rl_fidelity/matched_sets.json). Features use their refined firing rule where one exists.

    python -m circuit.rl_task.firing_coverage rl_features/rl_fidelity/firing_coverage.json
"""
import json, os, sys
import numpy as np
from circuit.rl_task.autointerp_magnitude import Mag

STORE = "rl_features/rl_chosen/chosen_rl_v1_r128_store"
matter = json.load(open("rl_features/rl_fidelity/matter_sets.json"))
matched = json.load(open("rl_features/rl_fidelity/matched_sets.json"))
NEG = {("update", 144), ("update", 131), ("candidate", 153), ("candidate", 192)}
FG = {"reset": "reset", "update": "update", "candidate": "hidden"}
DIRS = {"1%": "rl_features/rl_neuron_d01_autointerp", "matched": "rl_features/rl_neuron_matched_autointerp",
        "5%": "rl_features/rl_neuron_d05_autointerp"}
UNAME = {"reset": ["neuron_reset"], "update": ["neuron_update"], "candidate": ["neuron_cand_pos", "neuron_cand_neg"]}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def r2_onoff(m, preds):
    terms = " ;; ".join(f"({p})" for p in preds)
    return m.evaluate(terms, split="test")["r2_beyond_timing"]


out = {}
for setname in ("matter", "matched"):
    for g in ("reset", "update", "candidate"):
        units = matter[f"{g}/units"] if setname == "matter" else matched[g]["units"]["ids"]
        feats = [f for f in (matter[f"{g}/features"] if setname == "matter" else matched[g]["features"]["ids"])
                 if (g, f) not in NEG]
        # units
        for dname, d in DIRS.items():
            if dname == "matched" and g == "update":
                d = DIRS["1%"]            # matched update density is 1%
            rows = []
            m_cache = {}
            for u in units:
                recs = [load(os.path.join(d, f"{n}_{u}.json")) for n in UNAME[g]]
                recs = [r for r in recs if r]
                ok = [r for r in recs if r.get("final_verdict") == "explained" and r.get("predicate")]
                f1s = [r["test_scores"]["f1"] for r in recs if r.get("test_scores")]
                r2 = None
                if ok:
                    m = Mag(STORE, f"unit_{g}", u)
                    r2 = r2_onoff(m, [r["predicate"] for r in ok])
                rows.append({"id": u, "n_ends": len(recs), "n_explained": len(ok), "f1": f1s, "r2_onoff": r2})
            out[f"{setname}/{g}/units/{dname}"] = rows
        # features (natural density): firing-only from the firing records, refined where present
        rows = []
        for f in feats:
            ref = load(f"rl_features/rl_refined/{FG[g]}_{f}.json")
            fr = load(f"rl_features/rl_autointerp/{FG[g]}_{f}.json")
            if ref and ref.get("predicate"):
                pred, f1, ok = ref["predicate"], ref["firing"]["test"]["f1"], ref["kind"] == "firing+modifier"
            elif ref:
                pred, f1, ok = None, (fr or {}).get("test_scores", {}).get("f1"), False
            else:
                pred, f1, ok = fr.get("predicate"), fr["test_scores"]["f1"], fr.get("final_verdict") == "explained"
            r2 = r2_onoff(Mag(STORE, FG[g], f), [pred]) if ok and pred else None
            rows.append({"id": f, "explained": bool(ok), "f1": f1, "r2_onoff": r2})
        out[f"{setname}/{g}/features"] = rows
        print(setname, g, "done", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
