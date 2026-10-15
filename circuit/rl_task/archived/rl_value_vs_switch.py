"""Top-k and core sets for the value vs for switching (single-drop rankings from single_drops.json, selection sample);
behaviour cores = smallest top-ranked set recovering 90% of the way from 'none' to 'full' on the selection sample.
Everything scored on the main eval set."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/Users/mishaalk/personal/rnnsuperposition")
from circuit.rl_task.explained_fidelity import load_eval, run_all, splice, teacher_forced
from circuit.copy_task.explained_fidelity import Ablated, components, load_transcoder
from training.splice_eval import build_task_rnn
torch.set_grad_enabled(False)
rnn = build_task_rnn("rl", 8, 48); rnn.load_state_dict(torch.load("data/models/rl_train/run_two_step_init_low_r1/run_two_step_init_low_r1.ckpt", map_location="cpu")); rnn.eval()
sel_g = load_eval("rl_features/rl_trace_sequences", 8000, seed=1); sel_r = [run_all(rnn, g) for g in sel_g]
ev_g = load_eval("rl_features/rl_trace_sequences", 20000, seed=0); ev_r = [run_all(rnn, g) for g in ev_g]
drops = json.load(open("rl_features/rl_fidelity/single_drops.json"))
fid = json.load(open("rl_features/rl_fidelity/fidelity.json"))
G = {"update": "update", "candidate": "hidden"}
out = {}
for gate, path in (("update", "rl_transcoders/update_w256_hlsparse_nonorm_final.ckpt"), ("candidate", "rl_transcoders/candidate_w256_mlsparse_ctd_e70.ckpt")):
    X, Y = teacher_forced(sel_g, gate)
    tc = load_transcoder(path)
    means, alive, _ = components(X, Y, tc)
    mk = lambda ids: Ablated(torch.zeros(tc.n_feats, dtype=torch.bool).index_fill_(0, torch.tensor(sorted(ids), dtype=torch.long), True)
                             if len(ids) else torch.zeros(tc.n_feats, dtype=torch.bool), means, tc)
    d = drops[gate]["drops"]
    rank = {"switch": [x["feature"] for x in sorted(d, key=lambda x: -x["switch_drop"])],
            "value": [x["feature"] for x in sorted(d, key=lambda x: -x["value_nmae"])]}
    fve_core = next(s["core_ids"] for s in fid["core"] if s["gate"] == gate)
    verdict = {i: json.load(open(f"rl_features/rl_autointerp/{G[gate]}_{i}.json")) for i in range(tc.n_feats)}
    # behaviour cores (bisection over k on the selection sample)
    none, full = splice(rnn, sel_g, sel_r, gate, mk([])), drops[gate]["full"]
    def core(order, metric, higher_better):
        target = (none[metric] + 0.9 * (full[metric] - none[metric]))
        ok = lambda k: (splice(rnn, sel_g, sel_r, gate, mk(order[:k]))[metric] >= target) if higher_better else \
                       (splice(rnn, sel_g, sel_r, gate, mk(order[:k]))[metric] <= target)
        lo, hi = 1, len(order)
        while lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if ok(mid) else (mid + 1, hi)
        return order[:lo]
    cores = {"switch": core(rank["switch"], "switch_agreement", True), "value": core(rank["value"], "value_nmae", False), "fve": fve_core}
    rows = []
    def ev(name, ids):
        s = splice(rnn, ev_g, ev_r, gate, mk(list(ids)))
        ex = sum(verdict[i].get("final_verdict") == "explained" for i in ids)
        rows.append({"set": name, "ids": sorted(ids), "n_kept": len(ids), "explained": ex, **s})
        print(f"  {name:<20} kept {len(ids):3d} (explained {ex:3d})  switch {s['switch_agreement']:.3f}  go {s['go_agreement']:.3f}  value nmae {s['value_nmae']:.3f}")
    print(f"{gate}: (full: switch {full['switch_agreement']:.3f} go {full['go_agreement']:.3f} value nmae {full['value_nmae']:.3f}; none: switch {none['switch_agreement']:.3f} go {none['go_agreement']:.3f} value nmae {none['value_nmae']:.3f}, selection sample)")
    for k in (5, 10, 20, 40):
        ev(f"top {k} switch", rank["switch"][:k]); ev(f"top {k} value", rank["value"][:k])
        print(f"     overlap top {k}: {len(set(rank['switch'][:k]) & set(rank['value'][:k]))} of {k}")
    for name, ids in cores.items():
        ev(f"core ({name})", ids)
    c = {k: set(v) for k, v in cores.items()}
    print(f"     cores: switch {len(c['switch'])}, value {len(c['value'])}, fve {len(c['fve'])}; switch&value {len(c['switch'] & c['value'])}, "
          f"switch&fve {len(c['switch'] & c['fve'])}, value&fve {len(c['value'] & c['fve'])}")
    ev("core switch | value", c["switch"] | c["value"])
    # members: the value core with labels
    print(f"  value core members (F1, verdict, label); * = also in the switch core:")
    for i in cores["value"]:
        r = verdict[i]
        print(f"    {'*' if i in c['switch'] else ' '} {gate}/{i:>3} F1 {(r.get('test_scores') or {}).get('f1') or 0:.2f} [{r.get('final_verdict')}] {(r.get('final_label') or r.get('label') or '')[:100]}")
    out[gate] = {"rank": rank, "cores": cores, "eval": rows}
json.dump(out, open("rl_features/rl_fidelity/value_vs_switch.json", "w"), indent=1)
