"""Feature sets chosen for switch agreement, GO agreement and value together.
Selection on a separate sample (seed 1, 8000 windows); scores on the main eval set (seed 0, 20000 windows)."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/Users/mishaalk/personal/rnnsuperposition")
from circuit.rl_task.explained_fidelity import load_eval, run_all, splice, teacher_forced, fve
from circuit.copy_task.explained_fidelity import Ablated, components, load_transcoder, GATES
from training.splice_eval import build_task_rnn
torch.set_grad_enabled(False)
rnn = build_task_rnn("rl", 8, 48); rnn.load_state_dict(torch.load("data/models/rl_train/run_two_step_init_low_r1/run_two_step_init_low_r1.ckpt", map_location="cpu")); rnn.eval()
sel_g = load_eval("rl_features/rl_trace_sequences", 8000, seed=1); sel_r = [run_all(rnn, g) for g in sel_g]
ev_g = load_eval("rl_features/rl_trace_sequences", 20000, seed=0); ev_r = [run_all(rnn, g) for g in ev_g]
loss = lambda s: (1 - s["switch_agreement"]) + (1 - s["go_agreement"]) + s["value_nmae"]
drops = json.load(open("rl_features/rl_fidelity/single_drops.json"))
out = {}
for gate, path in (("update", "rl_transcoders/update_w256_hlsparse_nonorm_final.ckpt"), ("candidate", "rl_transcoders/candidate_w256_mlsparse_ctd_e70.ckpt")):
    X, Y = teacher_forced(sel_g, gate)
    tc = load_transcoder(path)
    means, alive, imp = components(X, Y, tc)
    mk = lambda ids: Ablated(torch.zeros(tc.n_feats, dtype=torch.bool).index_fill_(0, torch.tensor(sorted(ids), dtype=torch.long), True)
                             if ids else torch.zeros(tc.n_feats, dtype=torch.bool), means, tc)
    full = drops[gate]["full"]
    # 1. combined single-drop score (from the single drops on the selection sample)
    score = {d["feature"]: d["switch_drop"] + d["go_drop"] + (d["value_nmae"] - full["value_nmae"]) for d in drops[gate]["drops"]}
    ranked = sorted(score, key=lambda i: -score[i])
    # 2. greedy forward selection on the combined objective, from the top 80 by combined score
    pool, chosen, path_ = ranked[:80], [], []
    t0 = time.time()
    for k in range(40):
        best = min(((loss(splice(rnn, sel_g, sel_r, gate, mk(chosen + [i]))), i) for i in pool if i not in chosen))
        chosen.append(best[1]); path_.append(best[0])
    print(f"{gate}: greedy selection over a pool of 80 in {time.time() - t0:.0f}s")
    res = []
    def ev(name, ids):
        s = splice(rnn, ev_g, ev_r, gate, mk(ids)); s.update(gate=gate, set=name, n_kept=len(ids), ids=list(ids), objective=loss(s))
        res.append(s)
        print(f"  {name:<26} kept {len(ids):3d}  switch {s['switch_agreement']:.3f}  go {s['go_agreement']:.3f}  value nmae {s['value_nmae']:.3f}  kl_go {s['kl_go']:.3f}  objective {s['objective']:.3f}")
    ev("full", list(range(tc.n_feats)))
    for k in (5, 10, 20, 30, 40):
        ev(f"combined top {k}", ranked[:k])
    for k in (5, 10, 20, 30, 40):
        ev(f"greedy {k}", chosen[:k])
    out[gate] = {"greedy_order": chosen, "greedy_selection_objective": path_, "combined_ranking": ranked[:80], "eval": res}
json.dump(out, open("rl_features/rl_fidelity/joint_selection.json", "w"), indent=1)
