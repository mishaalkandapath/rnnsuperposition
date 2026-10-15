"""Backward elimination on the combined objective (1-switch) + (1-go) + value_nmae.
Each round: single-removal loss for every kept feature on the selection sample; drop the ~10% least harmful."""
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
loss = lambda s: (1 - s["switch_agreement"]) + (1 - s["go_agreement"]) + s["value_nmae"]
SIZES = [80, 60, 40, 30, 20, 10]
out = {}
for gate, path in (("candidate", "rl_transcoders/candidate_w256_mlsparse_ctd_e70.ckpt"), ("update", "rl_transcoders/update_w256_hlsparse_nonorm_final.ckpt")):
    X, Y = teacher_forced(sel_g, gate)
    tc = load_transcoder(path)
    means, alive, _ = components(X, Y, tc)
    mk = lambda ids: Ablated(torch.zeros(tc.n_feats, dtype=torch.bool).index_fill_(0, torch.tensor(sorted(ids), dtype=torch.long), True), means, tc)
    kept = torch.nonzero(alive).flatten().tolist()
    snapshots, t0 = {}, time.time()
    while len(kept) > SIZES[-1]:
        costs = sorted((loss(splice(rnn, sel_g, sel_r, gate, mk([j for j in kept if j != i]))), i) for i in kept)
        target = next((s for s in SIZES if s < len(kept)), SIZES[-1])
        n_drop = max(1, min(len(kept) - target, int(np.ceil(0.1 * len(kept)))))
        drop = {i for _, i in costs[:n_drop]}
        kept = [i for i in kept if i not in drop]
        if len(kept) in SIZES:
            snapshots[len(kept)] = list(kept)
    print(f"{gate}: backward elimination in {time.time() - t0:.0f}s")
    res = []
    for n in SIZES:
        ids = snapshots[n]
        s = splice(rnn, ev_g, ev_r, gate, mk(ids)); s.update(gate=gate, n_kept=n, ids=ids, objective=loss(s)); res.append(s)
        print(f"  backward {n:3d}  switch {s['switch_agreement']:.3f}  go {s['go_agreement']:.3f}  value nmae {s['value_nmae']:.3f}  kl_go {s['kl_go']:.3f}  objective {s['objective']:.3f}")
    out[gate] = res
json.dump(out, open("rl_features/rl_fidelity/backward_selection.json", "w"), indent=1)
