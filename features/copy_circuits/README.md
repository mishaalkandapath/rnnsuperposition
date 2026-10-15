# copy_circuits: index

Every result of the copy-task circuit analysis (`docs/interp_circuits.md`, "§" below). All runs use the model and
transcoders listed in `features/README.md` and the 1-in-5 traces in `features/copy_transcoder_v2_1of5/`; L = 4 unless
the name says otherwise. Code: `circuit/copy_task/<script>.py` (each docstring has its command). Intervention kinds
(§2): 1 = real GRU, no features; 2 = real GRU edited along feature directions (transcoder error kept); 3 = spliced.

All files are current. Figures made from them: `plots/` (default output of `autointerp_plots.py`, all lengths and
slots) and `docs/figures/circuits/` (the ones the doc shows; index `docs/figures/README.md`).

## Results

| File | Made by | What it holds | § |
|---|---|---|---|
| `circuits.json` | `copy_circuits.py --sections graphs swaps contrast` | L = 3 and 4: single-prompt attribution graphs decomposed by source (identity / presence / biases / error), swap tests (each output's dependence on each input, normalised), contrastive graphs | 1, 2 |
| `absorption_L3.json`, `absorption_L4.json` | `slot_absorption.py` | Shared-write carriers, alignment, and slot binding (h before `<DEL>` from slot-specific vs bag-of-token one-hots: R² 0.99 vs 0.39) | 5 |
| `gates.json` | `gate_mechanism.py` | Per slot and step: the token's trace split into kept / written / gate parts, erase / write / read shares, read-out visibility; gate concentration; gate ablations | 5, 6, 8 |
| `feature_circuits_L4.json` | `feature_circuits.py` | Slot-token selectivity (supernodes), exact contrastive flow (DeepLIFT rescale) with edges, carrier ablations, per-sequence own ablations, steering by token tables with shuffled controls | 3, 4 |
| `token_circuits_L4_g.json` | `token_circuits.py` | Per-sequence circuits for sequences with `g` in slot 2, with short labels | 3 |
| `magnitude_scores_hidden.json`, `_update.json`, `_reset.json` | `magnitude_scores.py` | Additive token-table magnitude models beyond timing (inputs, outputs, all cues), compact token sets, auto labels; candidate: the 70 features firing on ≥ 20% of steps | 4 |
| `circuit_tests_L4.json` | `circuit_tests.py --sections del pop timing carry timeshift match resample` | Kind 1–2 causal tests: `<DEL>` 123 and reset 31, token-general pop features, freezing a step, the carry's second copy, gate time-shifts, the match split (−32.5 = content −32.4 + gate −0.1), mean vs resample ablation | 1, 5, 6, 7, 8 |
| `reset_combos_L4.json`, `reset_combos_L7.json` | `circuit_tests.py --sections reset_combos` | Reset features 31, 434, 312, 76, 147, 322, 477, 271 removed singly, in combinations, all 8 | 8 |
| `circuit_extras.json` | `circuit_extras.py` | Trace cosine over steps, pop split, early feedback, dense-carrier drops (spliced), token-conditional ablation | 4, 5, 6 |
| `rotation_edges_L4.json` | `rotation_edges.py` | Which step-to-step feature→feature edges turn the code (projections on the stored code and the read-out direction), turning / undoing features | 5 |
| `rotation_phases_L4.json` | `rotation_edges.py --phases` | Write angle of each carrier by steps since the write, keeper / flipper groups, the sign test (input vs memory code), the reversal decomposition | 5 |
| `reversal_tests_L4.json` | `rotation_edges.py --reversal_tests rotation_phases_L4.json` | Causal tests of what makes flippers reverse (remove flippers at the write step; remove keepers one step later; random controls) | 5 |
| `gate_features_L4.json` | `gate_features.py` | Gate features' drivers (which candidate features feed them), reset reads by group, update admit / erase | 8 |
| `circuit_sufficiency_L4.json` | `circuit_sufficiency.py` | Minimal circuit (per-sequence top-k features per step by \|flow\|): sufficiency and necessity, free and clamped, against size and random controls | 3 |
| `clamp_tests.json` | `clamp_tests.py` | Per-feature clamp tests of stated effects at L = 4 and 7: 6 important sparse + 40 random explained firing features (on / off), 16 magnitude features (boost / suppress vs random token sets) | 3 |
| `conditional_importance.json` | `conditional_analyses.py --sections importance` | Every candidate feature mean-ablated alone, L = 4 and 7: drop over all sequences, on its active sequences, per output token; repeat rate; sequence rate | 3 |
| `conditional_ablations.json` | `conditional_analyses.py --sections ablations` | Every candidate feature, L = 4 and 7: decoder norm × activation, gated write, zero and time-shift ablations | 3 |
| `conditional_pop.json` | `conditional_analyses.py --sections pop` | All 210 "copy-phase input X" features: on / off clamp results at L = 4 and 7, write size, direct logit effect | 3, 6 |
| `conditional_flows.json` | `conditional_analyses.py --sections flows` | Per feature: mean \|flow\| over all (sequence, slot) pairs vs the pairs it takes part in, write angle, per-step breakdown | 3, 4, 5 |
| `L9/conditional_importance.json`, `L9/conditional_ablations.json` | `conditional_analyses.py --sections importance ablations` on the L = 9 traces | The same two analyses at L = 9 (late-position features) | 3 |
| `verbal_rules_145.json` | `verbal_rules.py` | Feature 145's magnitude explanation as verbal rules with nothing fitted (Spearman, R², per phase) vs the fitted version | 4 |

## Run logs

Each prints the model, then progress and the "wrote" line of the run that produced the current file.

| Log | Run |
|---|---|
| `run.log` | `copy_circuits.py` graphs and swaps (L = 3, 4) |
| `run_contrast.log` | `copy_circuits.py --sections contrast` (added to `circuits.json`) |
| `run_feature_circuits_L4.log` | `feature_circuits.py` |
| `run_circuit_tests_L4.log` | `circuit_tests.py` (after the reset-gate fix, §9) |
| `run_conditional_importance.log`, `run_conditional_ablations.log`, `run_conditional_pop.log`, `run_conditional_flows.log`, `run_conditional_L9.log` | the `conditional_analyses.py` sections |
