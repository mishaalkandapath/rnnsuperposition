"""Token-specific copy circuits: one real sequence, one slot, in features.

For a chosen sequence and slot i (e.g. a sequence with `g` in slot 2):

  flow       contrastive flow (original minus a run with token i swapped,
             feedback changed to match; circuit.copy_task.feature_circuits.change_flow):
             the candidate features that carry this token at each step, how
             much each moved because of it (activation change), and the flow
             between steps and into the logit.
  gates      this sequence's update / reset gate features with the largest share
             of erasing / reading the token's trace (circuit.copy_task.gate_mechanism).
  tests      on this sequence, per step: its own top-5 carriers mean-ablated
             (probability of the correct token at output i), and its top-20
             carriers' magnitudes moved by the slot-token tables as if the slot
             held the swapped token (does output i switch to it?).
  pop        the logit of this token at output i+1 with the true feedback, and
             with the feedback replaced by another token (no match to pop).

Labels: magnitude labels (features/<run>_magnitude, circuit.copy_task.autointerp_magnitude)
for dense features, firing labels (autointerp) otherwise and for gate features.

    python -m circuit.copy_task.token_circuits --rnn RNN.ckpt --traces T4.pt \\
        --transcoders reset=R.ckpt update=U.ckpt candidate=C.ckpt \\
        --feature_results features/<run>_autointerp --magnitude_results features/<run>_magnitude \\
        --token g --out token_circuits.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import StackDataset

from circuit.copy_task.autointerp import TOKEN_IDX, TOKENS
from circuit.copy_task.copy_circuits import gru_trace, step_label, swap_token
from circuit.edge_att import CircuitTracer
from circuit.copy_task.explained_fidelity import load_transcoder
from circuit.copy_task.feature_circuits import (ablated_candidate_run, candidate_acts, change_flow, edited_run,
                                      selectivity)
from circuit.copy_task.gate_mechanism import slot_trace
from training.splice_eval import build_task_rnn

torch.serialization.add_safe_globals([StackDataset])


def short(text, n=60):
    """First clause of a label, cut at a word boundary."""
    text = (text or "?").split(";")[0].strip()
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return cut + " ..."


def load_label_sets(feature_results, magnitude_results):
    labels = defaultdict(dict)
    for kind, d in (("firing", feature_results), ("magnitude", magnitude_results)):
        if not d or not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".json") and fn.split("_")[0] in ("hidden", "update", "reset"):
                r = json.load(open(os.path.join(d, fn)))
                labels[(r["gate"], r["feature"])][kind] = {
                    "label": r.get("final_label") or r.get("label"), "verdict": r.get("final_verdict"),
                    "score": (r.get("test_scores") or {}).get("r2_beyond_timing" if kind == "magnitude" else "f1")}
    return labels


def label_of(labels, gate, f):
    rec = labels.get((gate, f), {})
    if "magnitude" in rec:
        return {"label": short(rec["magnitude"]["label"]), "source": "magnitude", **rec["magnitude"]}
    if "firing" in rec:
        return {"label": short(rec["firing"]["label"]), "source": "firing", **rec["firing"]}
    return {"label": "?", "source": None}


@torch.no_grad()
def example(args, rnn, tc, upd, rst, tracer, x, L, i, new, tables, rate, means, labels):
    orig = [int(v) for v in x[:L, :-1].argmax(-1)]
    tau = L + i
    run_o = tracer.run_forward_pass(gru_trace(rnn, x))
    run_s = tracer.run_forward_pass(gru_trace(rnn, swap_token(x, L, i, new)))
    fl = change_flow(tracer, run_o, run_s, L, orig, i)
    nodes = fl["node_list"]
    by_t = defaultdict(list)
    for nd, v in fl["nodes"].items():
        by_t[nd[0]].append((v, nd[1]))

    # gates for this sequence
    gsteps = slot_trace(rnn, upd, rst, x[None], L, i, torch.tensor([new]))
    gates = {}
    for k, s in enumerate(gsteps):
        t = i + k
        g = {}
        for name, gate in (("erase_by", "update"), ("read_by", "reset")):
            if s.get(name) is None:
                continue
            feats = s[name]["features"][0]
            j = int(torch.argmax(feats.abs()))
            g[name] = {"feature": j, "share": float(feats[j]), "bias": float(s[name]["bias"][0]),
                       **label_of(labels, gate, j)}
        g["kept_frac"] = float(s["kept_frac"][0])
        gates[t] = g

    dec = tc.features_to_outputs.weight
    xb = x[None]
    base = edited_run(rnn, xb, {})[0, i]
    steps = []
    for t in range(i, tau + 1):
        items = sorted(by_t.get(t, []), key=lambda vf: -vf[0])
        carriers = [f for v, f in items if v > 0]
        top = []
        for v, f in items[:args.top]:
            top.append({"feature": f, "flow": v, "da": fl["da"][(t, f)],
                        "act": float(run_o["f_n"][t, f]), "act_swapped": float(run_s["f_n"][t, f]),
                        "kind": "sparse" if rate[t, f] < 0.2 else "dense", "fire_rate": float(rate[t, f]),
                        **label_of(labels, "hidden", f)})
        # this sequence's interventions at step t
        mask = torch.zeros(1, tc.n_feats)
        mask[0, carriers[:5]] = 1.0
        ab = ablated_candidate_run(rnn, tc, xb, t, mask, means[t])[0, i]
        F = carriers[:20]
        delta = (tables[i, new, t][F] - tables[i, orig[i], t][F]) @ dec[:, F].T
        st = edited_run(rnn, xb, {t: delta[None]})[0, i]
        steps.append({"step": step_label(t, L), "t": t, "flow_total": sum(v for v, _ in items),
                      "top": top, "gates": gates.get(t, {}),
                      "ablate5_p_correct": float(ab.softmax(-1)[orig[i]]),
                      "ablate5_pred": TOKENS[int(ab.argmax())],
                      "steer20_pred": TOKENS[int(st.argmax())], "steer20_p_new": float(st.softmax(-1)[new])})
    step_edges = defaultdict(float)
    for gi, fi in torch.nonzero(fl["edges"].abs() > 1e-4).tolist():
        step_edges[(nodes[fi][0], nodes[gi][0])] += float(fl["edges"][gi, fi])
    for g_, (t, _) in enumerate(nodes):
        step_edges[(t, "target")] += float(fl["target_direct"][g_])
    inputs = defaultdict(float)
    for g_, (t, _) in enumerate(nodes):
        inputs[t] += float(fl["into_nodes"]["input"][g_])

    # pop: logit of orig[i] at output i+1 with true vs mismatched feedback
    pop = None
    if i + 1 < L:
        x3 = x.clone()
        x3[L + 1 + i] = 0
        x3[L + 1 + i, new] = 1
        cen = lambda lg: lg - lg.mean()
        true_fb = cen(edited_run(rnn, xb, {})[0, i + 1])
        bad_fb = cen(edited_run(rnn, x3[None], {})[0, i + 1])
        pop = {"logit_true_feedback": float(true_fb[orig[i]]), "logit_mismatched_feedback": float(bad_fb[orig[i]]),
               "pred_mismatched": TOKENS[int(bad_fb.argmax())]}
    return {"slot": i, "sequence": [TOKENS[int(v)] for v in x.argmax(-1)], "token": TOKENS[orig[i]],
            "swapped_to": TOKENS[new], "dtarget": fl["dtarget"], "check": fl["check"],
            "base_pred": TOKENS[int(base.argmax())], "steps": steps,
            "step_edges": [{"src": step_label(s, L), "dst": d if d == "target" else step_label(d, L), "flow": v}
                           for (s, d), v in step_edges.items()],
            "input_flow": {step_label(t, L): v for t, v in inputs.items() if abs(v) > 1e-3}, "pop": pop}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--traces", required=True, help="trace file of one length")
    ap.add_argument("--transcoders", nargs=3, required=True, metavar="GATE=PATH")
    ap.add_argument("--feature_results", default=None)
    ap.add_argument("--magnitude_results", default=None)
    ap.add_argument("--token", default=None, help="pick sequences with this token in the slot")
    ap.add_argument("--slots", type=int, nargs="*", help="default: all")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = dict(item.split("=", 1) for item in args.transcoders)
    rnn = build_task_rnn("copy", 31, 128)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    tc, upd, rst = (load_transcoder(paths[k]) for k in ("candidate", "update", "reset"))
    tracer = CircuitTracer(rnn, upd, tc, device="cpu", reset_transcoder=rst)
    labels = load_label_sets(args.feature_results, args.magnitude_results)
    data = torch.load(args.traces, map_location="cpu", mmap=True).datasets
    cols = {k: data[k] for k in ("inputs", "h_prevs", "r_ts")}
    L = data["outputs"].shape[1]
    tokens = cols["inputs"][:, :L, :-1].argmax(-1)
    acts = candidate_acts(tc, cols)
    _, rate, tables = selectivity(acts, tokens)
    means = acts.mean(0)
    rng = np.random.default_rng(args.seed)
    distinct = np.array([len(set(r)) == L for r in tokens.tolist()])
    out = []
    for i in (args.slots if args.slots is not None else range(L)):
        ok = distinct & ((tokens[:, i] == TOKEN_IDX[args.token]).numpy() if args.token else True)
        row = int(rng.choice(np.flatnonzero(ok)))
        orig = tokens[row].tolist()
        new = int(rng.choice([v for v in range(30) if v not in orig]))
        ex = example(args, rnn, tc, upd, rst, tracer, cols["inputs"][row].float(), L, i, new, tables, rate, means, labels)
        ex["row"] = row
        out.append(ex)
        print(f"slot {i}: {' '.join(ex['sequence'])} | token {ex['token']} (vs {ex['swapped_to']}), "
              f"logit change {ex['dtarget']:+.1f} (check {ex['check']:.1e})")
        for s in ex["steps"]:
            print(f"   {s['step']:>4} flow {s['flow_total']:+.1f}: " + "; ".join(
                f"{c['feature']} {c['kind'][0]} da {c['da']:+.2f} [{c['label'][:40]}]" for c in s["top"])
                  + f" | ablate5 p {s['ablate5_p_correct']:.2f}, steer20 -> {s['steer20_pred']}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "L": L, "examples": out}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
