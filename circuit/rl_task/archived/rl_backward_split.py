"""Backward elimination separately for the choice ((1-switch)+(1-go)) and the value (value_nmae).
Selection sample: seed 1, 8000 windows; scores on the eval set (seed 0, 20000 windows)."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/Users/mishaalk/personal/rnnsuperposition")
from circuit.rl_task.explained_fidelity import load_eval, run_all, splice, teacher_forced
from circuit.copy_task.explained_fidelity import Ablated, components, load_transcoder
from training.splice_eval import build_task_rnn
torch.set_grad_enabled(False)
rnn = build_task_rnn("rl", 8, 48); rnn.load_state_dict(torch.load("data/models/rl_train/run_two_step_init_low_r1/run_two_step_init_low_r1.ckpt", map_location="cpu")); rnn.eval()
sel_g = load_eval("rl_features/rl_trace_sequences", 8000, seed=1); sel_r = [run_all(rnn, g) for g in sel_g]
ev_g = load_eval("rl_features/rl_trace_sequences", 20000, seed=0); ev_r = [run_all(rnn, g) for g in ev_g]
OBJ = {"choice": lambda s: (1 - s["switch_agreement"]) + (1 - s["go_agreement"]), "value": lambda s: s["value_nmae"]}
SIZES = [80, 60, 40, 30, 20, 10]
G = {"update": "update", "candidate": "hidden"}
out = {}
for gate, path in (("update", "rl_transcoders/update_w256_hlsparse_nonorm_final.ckpt"), ("candidate", "rl_transcoders/candidate_w256_mlsparse_ctd_e70.ckpt")):
    X, Y = teacher_forced(sel_g, gate)
    tc = load_transcoder(path)
    means, alive, _ = components(X, Y, tc)
    mk = lambda ids: Ablated(torch.zeros(tc.n_feats, dtype=torch.bool).index_fill_(0, torch.tensor(sorted(ids), dtype=torch.long), True), means, tc)
    sets = {}
    for name, obj in OBJ.items():
        kept, snap, t0 = torch.nonzero(alive).flatten().tolist(), {}, time.time()
        while len(kept) > SIZES[-1]:
            costs = sorted((obj(splice(rnn, sel_g, sel_r, gate, mk([j for j in kept if j != i]))), i) for i in kept)
            target = next((s for s in SIZES if s < len(kept)), SIZES[-1])
            n_drop = max(1, min(len(kept) - target, int(np.ceil(0.1 * len(kept)))))
            drop = {i for _, i in costs[:n_drop]}
            kept = [i for i in kept if i not in drop]
            if len(kept) in SIZES:
                snap[len(kept)] = list(kept)
        sets[name] = snap
        print(f"{gate} {name}: backward elimination in {time.time() - t0:.0f}s", flush=True)
    verdict = {i: json.load(open(f"rl_features/rl_autointerp/{G[gate]}_{i}.json")).get("final_verdict") for i in range(tc.n_feats)}
    rows = []
    print(f"{gate}: set | switch | go | value nmae   (n = size; overlap = shared features between the choice and value sets)")
    for n in SIZES:
        c, v = set(sets["choice"][n]), set(sets["value"][n])
        for name, ids in (("choice", c), ("value", v), ("union", c | v)):
            s = splice(rnn, ev_g, ev_r, gate, mk(list(ids)))
            ex = sum(verdict[i] == "explained" for i in ids)
            rows.append({"n": n, "set": name, "n_kept": len(ids), "ids": sorted(ids), "explained": ex, **s})
            print(f"  n={n:3d} {name:<6} kept {len(ids):3d} (explained {ex:3d})  switch {s['switch_agreement']:.3f}  go {s['go_agreement']:.3f}  value nmae {s['value_nmae']:.3f}")
        print(f"         overlap {len(c & v)} of {n} (Jaccard {len(c & v) / len(c | v):.2f})")
    out[gate] = {"sets": sets, "eval": rows}
    json.dump(out, open("rl_features/rl_fidelity/split_selection.json", "w"), indent=1)
