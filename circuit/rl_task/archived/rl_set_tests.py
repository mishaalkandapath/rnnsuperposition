import json, sys
import numpy as np, torch
sys.path.insert(0, "/Users/mishaalk/personal/rnnsuperposition")
from circuit.rl_task.explained_fidelity import load_eval, run_all, splice, teacher_forced, fve
from circuit.copy_task.explained_fidelity import Ablated, components, load_f1, load_transcoder, GATES
from training.splice_eval import build_task_rnn
torch.set_grad_enabled(False)
rnn = build_task_rnn("rl", 8, 48); rnn.load_state_dict(torch.load("data/models/rl_train/run_two_step_init_low_r1/run_two_step_init_low_r1.ckpt", map_location="cpu")); rnn.eval()
groups = load_eval("rl_features/rl_trace_sequences", 20000, seed=0)       # same eval set as the main run
refs = [run_all(rnn, g) for g in groups]
drops = json.load(open("rl_features/rl_fidelity/single_drops.json"))
out = []
for gate, path in (("candidate", "rl_transcoders/candidate_w256_mlsparse_ctd_e70.ckpt"), ("update", "rl_transcoders/update_w256_hlsparse_nonorm_final.ckpt")):
    X, Y = teacher_forced(groups, gate)
    tc = load_transcoder(path)
    means, alive, imp = components(X, Y, tc)
    f1, _ = load_f1("rl_features/rl_autointerp", [GATES[gate][1]], tc.n_feats)
    verdict = {i: json.load(open(f"rl_features/rl_autointerp/{GATES[gate][1]}_{i}.json")).get("final_verdict") for i in range(tc.n_feats)}
    explained = torch.tensor([verdict[i] == "explained" for i in range(tc.n_feats)])
    ranked = [d["feature"] for d in drops[gate]["drops"]]                  # by single-drop switch loss (separate sample)
    partial_essential = [i for i in ranked[:30] if verdict[i] != "explained"][:6]
    def test(name, keep):
        r = splice(rnn, groups, refs, gate, Ablated(keep, means, tc)); r.update(gate=gate, test=name, n_kept=int(keep.sum()), fve=fve(X, Y, Ablated(keep, means, tc)))
        out.append(r); print(f"  {gate:>9} {name:<46} kept {r['n_kept']:3d}  fve {r['fve']:.3f}  go {r['go_agreement']:.3f}  switch {r['switch_agreement']:.3f}")
    z = lambda ids: torch.zeros(tc.n_feats, dtype=torch.bool).index_fill_(0, torch.tensor(ids, dtype=torch.long), True) if ids else torch.zeros(tc.n_feats, dtype=torch.bool)
    print(f"{gate}: partially explained features among the top-30 single drops: {partial_essential}")
    test("explained", explained)
    for k in (2, 4, 6):
        test(f"explained + top {k} essential partial", explained | z(partial_essential[:k]))
    test("all but those 6", ~z(partial_essential))
    for k in (5, 10, 20, 40):
        test(f"top {k} by single-drop behaviour loss", z(ranked[:k]))
    test("explained + all F1>=0.5", explained | (f1 >= 0.5))
json.dump(out, open("rl_features/rl_fidelity/set_tests.json", "w"), indent=1)
