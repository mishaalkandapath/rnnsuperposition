import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "/Users/mishaalk/personal/rnnsuperposition")
from circuit.rl_task.explained_fidelity import load_eval, run_all, splice, teacher_forced, fve
from circuit.copy_task.explained_fidelity import Ablated, components, load_f1, load_transcoder, GATES
from training.splice_eval import build_task_rnn
torch.set_grad_enabled(False)
rnn = build_task_rnn("rl", 8, 48); rnn.load_state_dict(torch.load("data/models/rl_train/run_two_step_init_low_r1/run_two_step_init_low_r1.ckpt", map_location="cpu")); rnn.eval()
groups = load_eval("rl_features/rl_trace_sequences", 8000, seed=1)       # separate sample from the main run
refs = [run_all(rnn, g) for g in groups]
ckpts = {"update": "rl_transcoders/update_w256_hlsparse_nonorm_final.ckpt", "candidate": "rl_transcoders/candidate_w256_mlsparse_ctd_e70.ckpt"}
out = {}
for gate, path in ckpts.items():
    X, Y = teacher_forced(groups, gate)
    tc = load_transcoder(path)
    means, alive, imp = components(X, Y, tc)
    f1, _ = load_f1("rl_features/rl_autointerp", [GATES[gate][1]], tc.n_feats)
    full = splice(rnn, groups, refs, gate, Ablated(torch.ones(tc.n_feats, dtype=torch.bool), means, tc))
    rows = []
    t0 = time.time()
    for i in torch.nonzero(alive).flatten().tolist():
        keep = torch.ones(tc.n_feats, dtype=torch.bool); keep[i] = False
        s = splice(rnn, groups, refs, gate, Ablated(keep, means, tc))
        rows.append({"feature": i, "f1": float(f1[i]), "importance": float(imp[i]),
                     "switch_drop": full["switch_agreement"] - s["switch_agreement"], "go_drop": full["go_agreement"] - s["go_agreement"],
                     "value_nmae": s["value_nmae"]})
    rows.sort(key=lambda r: -r["switch_drop"])
    out[gate] = {"full": full, "drops": rows}
    print(f"{gate}: full switch {full['switch_agreement']:.3f}; {len(rows)} single drops in {time.time() - t0:.0f}s")
    for r in rows[:15]:
        rec = json.load(open(f"rl_features/rl_autointerp/{GATES[gate][1]}_{r['feature']}.json"))
        print(f"  {gate}/{r['feature']:>3}  switch -{r['switch_drop']:.3f}  go -{r['go_drop']:.3f}  F1 {r['f1']:.2f}  "
              f"imp rank {int((np.array([x['importance'] for x in rows]) > r['importance']).sum())}  [{rec.get('final_verdict')}] {(rec.get('final_label') or rec.get('label'))[:110]}")
json.dump(out, open("rl_features/rl_fidelity/single_drops.json", "w"), indent=1)
