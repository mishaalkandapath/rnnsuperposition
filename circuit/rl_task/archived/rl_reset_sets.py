"""Reset gate: single drops, top-k / behaviour cores for switch vs value (+ fve core), and joint backward elimination.
Selection sample: seed 1, 8000 windows; scores on the main eval set (seed 0, 20000 windows)."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/Users/mishaalk/personal/rnnsuperposition")
from circuit.rl_task.explained_fidelity import load_eval, run_all, splice, teacher_forced
from circuit.copy_task.explained_fidelity import Ablated, components, load_transcoder, load_f1
from training.splice_eval import build_task_rnn
torch.set_grad_enabled(False)
rnn = build_task_rnn("rl", 8, 48); rnn.load_state_dict(torch.load("data/models/rl_train/run_two_step_init_low_r1/run_two_step_init_low_r1.ckpt", map_location="cpu")); rnn.eval()
sel_g = load_eval("rl_features/rl_trace_sequences", 8000, seed=1); sel_r = [run_all(rnn, g) for g in sel_g]
ev_g = load_eval("rl_features/rl_trace_sequences", 20000, seed=0); ev_r = [run_all(rnn, g) for g in ev_g]
gate = "reset"
X, Y = teacher_forced(sel_g, gate)
tc = load_transcoder("rl_transcoders/reset_w128_mlsparse_nonorm_e100.ckpt")
means, alive, imp = components(X, Y, tc)
f1, _ = load_f1("rl_features/rl_autointerp", ["reset"], tc.n_feats)
mk = lambda ids: Ablated(torch.zeros(tc.n_feats, dtype=torch.bool).index_fill_(0, torch.tensor(sorted(ids), dtype=torch.long), True)
                         if len(ids) else torch.zeros(tc.n_feats, dtype=torch.bool), means, tc)
verdict = {i: json.load(open(f"rl_features/rl_autointerp/reset_{i}.json")) for i in range(tc.n_feats)}
full = splice(rnn, sel_g, sel_r, gate, mk(list(range(tc.n_feats)))); none = splice(rnn, sel_g, sel_r, gate, mk([]))
print(f"selection sample: full switch {full['switch_agreement']:.3f} go {full['go_agreement']:.3f} value {full['value_nmae']:.3f}; "
      f"none switch {none['switch_agreement']:.3f} go {none['go_agreement']:.3f} value {none['value_nmae']:.3f}")
# 1. single drops
drops = []
for i in torch.nonzero(alive).flatten().tolist():
    s = splice(rnn, sel_g, sel_r, gate, mk([j for j in range(tc.n_feats) if j != i]))
    drops.append({"feature": i, "f1": float(f1[i]), "switch_drop": full["switch_agreement"] - s["switch_agreement"],
                  "go_drop": full["go_agreement"] - s["go_agreement"], "value_nmae": s["value_nmae"]})
rank = {"switch": [d["feature"] for d in sorted(drops, key=lambda d: -d["switch_drop"])],
        "value": [d["feature"] for d in sorted(drops, key=lambda d: -d["value_nmae"])],
        "combined": [d["feature"] for d in sorted(drops, key=lambda d: -(d["switch_drop"] + d["go_drop"] + d["value_nmae"] - full["value_nmae"]))]}
print("top single drops by switch:", [(d["feature"], round(d["switch_drop"], 3)) for d in sorted(drops, key=lambda d: -d["switch_drop"])[:8]])
print("top single drops by value: ", [(d["feature"], round(d["value_nmae"] - full["value_nmae"], 3)) for d in sorted(drops, key=lambda d: -d["value_nmae"])[:8]])
# 2. behaviour cores
def core(order, metric, higher):
    target = none[metric] + 0.9 * (full[metric] - none[metric])
    ok = lambda k: (lambda v: v >= target if higher else v <= target)(splice(rnn, sel_g, sel_r, gate, mk(order[:k]))[metric])
    lo, hi = 1, len(order)
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(mid) else (mid + 1, hi)
    return order[:lo]
fid = json.load(open("rl_features/rl_fidelity/fidelity.json"))
cores = {"switch": core(rank["switch"], "switch_agreement", True), "value": core(rank["value"], "value_nmae", False),
         "fve": next(s["core_ids"] for s in fid["core"] if s["gate"] == gate)}
# 3. joint backward elimination
obj = lambda s: (1 - s["switch_agreement"]) + (1 - s["go_agreement"]) + s["value_nmae"]
kept, snap = torch.nonzero(alive).flatten().tolist(), {}
SIZES = [40, 30, 20, 10, 5]
while len(kept) > SIZES[-1]:
    costs = sorted((obj(splice(rnn, sel_g, sel_r, gate, mk([j for j in kept if j != i]))), i) for i in kept)
    target = next((s for s in SIZES if s < len(kept)), SIZES[-1])
    n_drop = max(1, min(len(kept) - target, int(np.ceil(0.1 * len(kept)))))
    drop = {i for _, i in costs[:n_drop]}
    kept = [i for i in kept if i not in drop]
    if len(kept) in SIZES:
        snap[len(kept)] = list(kept)
rows = []
def ev(name, ids):
    s = splice(rnn, ev_g, ev_r, gate, mk(list(ids)))
    ex = sum(verdict[i].get("final_verdict") == "explained" for i in ids)
    rows.append({"set": name, "ids": sorted(ids), "n_kept": len(ids), "explained": ex, **s})
    print(f"  {name:<22} kept {len(ids):3d} (explained {ex:3d})  switch {s['switch_agreement']:.3f}  go {s['go_agreement']:.3f}  value nmae {s['value_nmae']:.3f}")
print("eval set:")
ev("full", range(tc.n_feats)); ev("none", [])
for k in (5, 10, 20):
    ev(f"top {k} switch", rank["switch"][:k]); ev(f"top {k} value", rank["value"][:k])
    print(f"     overlap top {k}: {len(set(rank['switch'][:k]) & set(rank['value'][:k]))} of {k}")
for name, ids in cores.items():
    ev(f"core ({name})", ids)
c = {k: set(v) for k, v in cores.items()}
print(f"     cores: switch {len(c['switch'])}, value {len(c['value'])}, fve {len(c['fve'])}; switch&value {len(c['switch'] & c['value'])}, "
      f"switch&fve {len(c['switch'] & c['fve'])}, value&fve {len(c['value'] & c['fve'])}")
ev("core switch | value", c["switch"] | c["value"])
for n in SIZES:
    ev(f"joint backward {n}", snap[n])
print("members of the switch and value cores (* = in both):")
for i in sorted(c["switch"] | c["value"], key=lambda i: (i not in c["switch"] & c["value"], i)):
    r = verdict[i]
    where = "both" if i in c["switch"] & c["value"] else ("switch" if i in c["switch"] else "value")
    print(f"   [{where:>6}] reset/{i:>3} F1 {(r.get('test_scores') or {}).get('f1') or 0:.2f} [{r.get('final_verdict')}] {(r.get('final_label') or r.get('label') or '')[:105]}")
json.dump({"drops": drops, "rank": rank, "cores": cores, "joint_backward": snap, "eval": rows},
          open("rl_features/rl_fidelity/reset_sets.json", "w"), indent=1)
