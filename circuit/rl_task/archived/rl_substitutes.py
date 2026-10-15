"""Are the choice-only / value-only backward-elimination features substitutes for each other?
Feature g's contribution to the gate at step t: c_g(t) = (a_g(t) - mean_g) d_g (what mean ablation removes).
Similarity / R^2 use the Gram K_gh = sum_t c_g(t).c_h(t) = (A_c^T A_c)_gh (d_g.d_h) on the selection sample.
1. representational: R^2 of each choice-only feature's contribution regressed on the whole value set's
   contributions (and vice versa), against nulls (random same-size sets; features dropped from both sets).
2. functional: replace each choice-only feature by its most similar feature outside the choice set
   (one-to-one), evaluate on the eval set; control: random replacements."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/Users/mishaalk/personal/rnnsuperposition")
from circuit.rl_task.explained_fidelity import load_eval, run_all, splice, teacher_forced
from circuit.copy_task.explained_fidelity import Ablated, components, load_transcoder
from training.splice_eval import build_task_rnn
torch.set_grad_enabled(False)
rng = np.random.default_rng(0)
rnn = build_task_rnn("rl", 8, 48); rnn.load_state_dict(torch.load("data/models/rl_train/run_two_step_init_low_r1/run_two_step_init_low_r1.ckpt", map_location="cpu")); rnn.eval()
sel_g = load_eval("rl_features/rl_trace_sequences", 8000, seed=1)
ev_g = load_eval("rl_features/rl_trace_sequences", 20000, seed=0); ev_r = [run_all(rnn, g) for g in ev_g]
split = json.load(open("rl_features/rl_fidelity/split_selection.json"))
PATHS = {"update": "rl_transcoders/update_w256_hlsparse_nonorm_final.ckpt", "candidate": "rl_transcoders/candidate_w256_mlsparse_ctd_e70.ckpt"}
out = {}
for gate, path in PATHS.items():
    X, Y = teacher_forced(sel_g, gate)
    tc = load_transcoder(path)
    means, alive, _ = components(X, Y, tc)
    G = 0
    for i in range(0, len(X), 65536):
        a = tc.act(tc.input_to_features(X[i:i + 65536])) - means
        G = G + (a.T.double() @ a.double())
    D = tc.features_to_outputs.weight.double()
    K = G * (D.T @ D)                                                     # (feats, feats)
    diag = K.diagonal().clamp_min(1e-12)
    sim = K / torch.sqrt(diag[:, None] * diag[None, :])
    def r2(f, S):
        S = [s for s in S if s != f]
        kss = K[S][:, S] + 1e-9 * torch.eye(len(S), dtype=K.dtype) * K[S][:, S].diagonal().mean()
        ksf = K[S, f]
        return float(ksf @ torch.linalg.solve(kss, ksf) / K[f, f])
    pool = torch.nonzero(alive).flatten().tolist()
    mk = lambda ids: Ablated(torch.zeros(tc.n_feats, dtype=torch.bool).index_fill_(0, torch.tensor(sorted(ids), dtype=torch.long), True), means, tc)
    out[gate] = {}
    print(f"\n{gate}: {len(pool)} features fire", flush=True)
    for n in ("20", "40", "60"):
        C, V = set(split[gate]["sets"]["choice"][n]), set(split[gate]["sets"]["value"][n])
        c_only, v_only, both = sorted(C - V), sorted(V - C), sorted(C & V)
        neither = [f for f in pool if f not in C | V]
        rand_null = lambda f, k: np.mean([r2(f, rng.choice([p for p in pool if p != f], size=k, replace=False).tolist()) for _ in range(10)])
        rep = {}
        for name, only, other in (("choice-only on value set", c_only, V), ("value-only on choice set", v_only, C)):
            got = [r2(f, other) for f in only]
            null = [rand_null(f, len(other)) for f in only]
            base = [r2(f, other) for f in neither]
            best = [float(sim[f, sorted(other)].abs().max()) for f in only]
            rep[name] = {"r2": got, "null_random": null, "dropped_both_r2": base, "best_pair_sim": best}
            print(f"  n={n:>2} {name:<26} ({len(only):2d} feats): R2 median {np.median(got):.2f} (>0.5 for {np.mean(np.array(got) > 0.5):.0%}); "
                  f"random-set null {np.median(null):.2f}; dropped-from-both {np.median(base):.2f}; best single |sim| median {np.median(best):.2f}", flush=True)
        # functional swap
        def swap(only, base_set, others):
            used, picks = set(base_set), []
            for f in sorted(only, key=lambda f: -float(sim[f, [g for g in pool if g not in base_set]].abs().max())):
                cands = [g for g in pool if g not in used]
                g = max(cands, key=lambda g: float(sim[f, g].abs())); used.add(g); picks.append(g)
            return [f for f in base_set if f not in only] + picks
        rows = {}
        for label, only, base_set, metric in (("choice", c_only, C, "switch_agreement"), ("value", v_only, V, "value_nmae")):
            kept = [f for f in base_set if f not in only]
            outside = [g for g in pool if g not in base_set]
            sets = {"original": sorted(base_set), "shared only": kept, "nearest substitutes": swap(only, base_set, outside)}
            for r in range(3):
                sets[f"random substitutes #{r}"] = kept + rng.choice(outside, size=len(only), replace=False).tolist()
            for name, ids in sets.items():
                s = splice(rnn, ev_g, ev_r, gate, mk(ids))
                rows[f"{label}: {name}"] = {"ids": sorted(ids), **s}
                print(f"  n={n:>2} {label} set, {name:<22} switch {s['switch_agreement']:.3f}  go {s['go_agreement']:.3f}  value nmae {s['value_nmae']:.3f}", flush=True)
        out[gate][n] = {"representational": rep, "functional": rows}
        json.dump(out, open("rl_features/rl_fidelity/substitutes.json", "w"), indent=1)
