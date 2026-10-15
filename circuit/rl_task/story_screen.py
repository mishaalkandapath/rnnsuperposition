"""Story-test screen: which component's words cost the most behaviour? Baseline = every set member replaced by its
words-only simulation; then, one member at a time, that member gets a perfect 0-10 simulator (its true activation,
rounded) and the recovery gain is recorded.

    python -m circuit.rl_task.story_screen candidate rl_features/rl_simulation_shared rl_features/rl_simulation_refined \\
        rl_features/rl_fidelity/story_screen_candidate.json
"""
import json, os, sys
import numpy as np
import torch
from circuit.copy_task.explained_fidelity import load_transcoder
from circuit.rl_task.explained_fidelity import behaviour_metrics, run_all, teacher_forced, load_eval
from circuit.rl_task.explanation_substitution import eval_windows
from circuit.rl_task.matched_sets import phase_means
from circuit.rl_task.simulation_scoring import Component, shared_windows
from circuit.rl_task.story_test import run, METRICS
from circuit.rl_task.autointerp import Interp
from training.splice_eval import build_task_rnn

gate, sim_dir = sys.argv[1], sys.argv[2]
ref_dir = sys.argv[3] if len(sys.argv) > 4 else ""
out_path = sys.argv[-1]
torch.set_grad_enabled(False)
STORE = "rl_features/rl_chosen/chosen_rl_v1_r128_store"
TC = {"reset": "rl_transcoders/reset_w128_mlsparse_nonorm_e100.ckpt", "update": "rl_transcoders/update_w256_hlsparse_nonorm_final.ckpt",
      "candidate": "rl_transcoders/candidate_w256_mlsparse_ctd_e70.ckpt"}
NAME = {"reset": "reset", "update": "update", "candidate": "hidden"}[gate]
rnn = build_task_rnn("rl", input_size=8, hidden_size=48)
rnn.load_state_dict(torch.load("data/models/rl_train/run_two_step_init_low_r1/run_two_step_init_low_r1.ckpt", map_location="cpu")); rnn.eval()
ip = Interp(STORE); wins = shared_windows(ip)
groups, _, _ = eval_windows(STORE, "rl_features/rl_trace_sequences", len(wins), picks=wins)
refs = [run_all(rnn, g) for g in groups]
rs = np.asarray(ip.cols["row_start"]); rows = np.concatenate([np.arange(rs[w], rs[w + 1]) for w in wins])
big = load_eval("rl_features/rl_trace_sequences", 8000, seed=1)
Xb, Yb = teacher_forced(big, gate); tc = load_transcoder(TC[gate])
means, alive = phase_means(Xb, Yb, tc)
ids = json.load(open("rl_features/rl_fidelity/matched_sets.json"))[gate]["features"]["ids"]
keep = torch.zeros(len(alive), dtype=torch.bool); keep[ids] = True


def to_override(c, pred):
    comp = Component(STORE, NAME, c)
    vals = np.zeros(ip.n_rows); vals[rows] = comp.lo + np.asarray(pred, dtype=np.float64) / 10 * (comp.hi - comp.lo)
    return [torch.from_numpy(vals[g["rows"]]).float() for g in groups], comp


words, perfect = {}, {}
for c in ids:
    p = os.path.join(ref_dir, f"{NAME}_{c}.json") if ref_dir else ""
    p = p if p and os.path.exists(p) else os.path.join(sim_dir, f"{NAME}_{c}.json")
    words[c], comp = to_override(c, json.load(open(p))["predictions"])
    perfect[c], _ = to_override(c, np.round(comp.scaled[rows]))


def behave(ov, k=keep):
    outs = [run(rnn, g["inputs"], g["h0"], gate, k, means, tc, {c: v[gi] for c, v in ov.items()}) for gi, g in enumerate(groups)]
    return {m: v for m, v in behaviour_metrics(outs, refs).items() if m in METRICS}


kept, mean = behave({}), behave({}, torch.zeros(len(alive), dtype=torch.bool))
rec = lambda r: {m: (r[m] - mean[m]) / (kept[m] - mean[m]) for m in METRICS}
base = rec(behave(words))
print("baseline (all words):", {m: round(v, 3) for m, v in base.items()}, flush=True)
res = {"baseline": base, "per_component": {}}
for c in ids:
    ov = dict(words); ov[c] = perfect[c]
    r = rec(behave(ov))
    res["per_component"][c] = {m: r[m] - base[m] for m in METRICS}
    print(f"{c:4d} gain if perfect: switch {r['switch_agreement'] - base['switch_agreement']:+.3f} "
          f"go {r['go_agreement'] - base['go_agreement']:+.3f} value {r['value_nmae'] - base['value_nmae']:+.3f}", flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
