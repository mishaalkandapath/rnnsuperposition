# Copy-task figures

Figures for `docs/interpretability_findings.md` (this directory) and `docs/interp_circuits.md` (`circuits/`). All are
made by `python -m circuit.copy_task.autointerp_plots <command>`; inputs are in `features/` (index:
`features/README.md`; circuit results: `features/copy_circuits/README.md`). RL-task figures: `rl/README.md`.

Commands write to a default directory next to their input (e.g. `features/copy_circuits/plots/`, which also holds
the L = 3 and other-slot versions); the figures here are copies of the ones the docs show, or were written here
with `--out`. Three values separated by slashes are for the reset / update / candidate gates.

## `docs/figures/`: autointerp, features vs units (`interpretability_findings.md`)

| Figure | Command | What it shows | Inputs | § |
|---|---|---|---|---|
| `specificity_by_activation.png` | `specificity features/copy_gru_v1_candiff_autointerp` | Share of each activation quartile (weakest to strongest active steps) that the firing predicates cover, per gate, by verdict | `copy_gru_v1_candiff_autointerp/` | 2 |
| `activation_share_by_f1.png` | `coverage_plots features/copy_gru_v1_candiff_autointerp/plots/coverage_by_f1.json --units "units, matched density=features/copy_gru_v1_neuron_baseline_autointerp/coverage_by_f1.json" "units, 1% density=features/copy_gru_v1_neuron_d01_autointerp/coverage_by_f1.json" "units, 5% density=features/copy_gru_v1_neuron_d05_autointerp/coverage_by_f1.json" --out docs/figures` (the JSONs come from `coverage STORE RESULTS --out`) | Share of all of a gate's activations (component × timestep) that comes from components in each held-out F1 band, features vs units at matched / 1% / 5% density (candidate units: high and low end) | `copy_gru_v1_candiff_autointerp/plots/coverage_by_f1.json`, the units' `coverage_by_f1.json` | 3 |
| `step_coverage_by_f1.png` | as above (one command makes both) | Share of timesteps covered by features in each F1 band: saturated by a few dense features | as above | 3 |
| `fidelity_by_selection.png` | `fidelity features/explained_fidelity/fidelity.json` | Spliced: top-1 agreement with the original GRU when only components with F1 ≥ 0.8 / 0.5 / 0.2 (or the core set) are kept and the rest mean-ablated; features vs units at three densities, random sets of the same count as dots | `explained_fidelity/fidelity.json` | 4 |
| `core_sets_by_f1.png` | as above | The core set of each gate (fewest components for 90% of the full set's variance explained), split by held-out F1 band; features vs units | as above | 4 |
| `candidate_drop_tests.png` | as above (`{gate}_drop_tests.png`) | Candidate: top-1 agreement without the dense set (162, 214, 284, 388, 487), with only it, F1 selections plus it, and each dense feature dropped alone | as above | 4 |
| `dense_feature_tokens.png` | `probe features/dense_probe/probe.json` | Token tables of the 5 essential dense candidate features | `dense_probe/probe.json` | 6 |
| `dense_feature_{162,214,284,388,487}_pairs.png` | as above | Each dense feature's activation by token pair in its two slots | as above | 6 |

## `docs/figures/circuits/`: circuits (`interp_circuits.md`)

| Figure | Command | What it shows | Inputs (`features/copy_circuits/`) | § |
|---|---|---|---|---|
| `swap_L4.png`, `swap_norm_L4.png` | `circuits circuits.json` | Swap tests: how much each output changes when each input is swapped (raw and normalised); each output depends only on its own token | `circuits.json` | 1 |
| `queue_freeze_L4.png` | `circuit_summary features/copy_circuits` | Freezing one copy step delays the rest by one, never skips (free-running) | `circuit_tests_L4.json` (timing) | 1 |
| `identity_attribution_L4.png` | `circuits circuits.json` | Single-prompt graphs: credit to each input token per output, with partner credit off the diagonal | `circuits.json` | 2 |
| `contrast_share_L4.png` | as above | Contrastive graphs: share of the logit difference from each source | `circuits.json` | 2 |
| `feature_circuit_L4_slot2.png` | `feature_circuit feature_circuits_L4.json --gates gates.json --circuits circuits.json` | The feature circuit carrying slot 2 (supernodes by step, averaged contrastive flow) | `feature_circuits_L4.json`, `gates.json`, `circuits.json` | 3 |
| `token_circuit_L4_slot2_g.png` | `token_circuit token_circuits_L4_g.json` | The same for sequences with `g` in slot 2, with short labels | `token_circuits_L4_g.json` | 3 |
| `minimal_circuit_L4.png` | `circuit_summary` | Minimal circuit: sufficiency and necessity, free and clamped, against circuit size and random controls | `circuit_sufficiency_L4.json` | 3 |
| `feature_importance_conditional.png` | `conditional features/copy_circuits --out docs/figures/circuits` | Single-feature ablation drop over all sequences vs on the sequences where the feature is active (top) and on its worst output token (bottom), L = 4 and 7 | `conditional_importance.json` | 3 |
| `pop_features_by_token_conditional.png` | as above | Per token: the "fed-back X" feature with the largest effect (off → X repeated; on early → X's logit drops), and the inert ones with the same label | `conditional_pop.json` | 3 |
| `carrier_flow_conditional_L4.png` | as above | Each feature's mean \|flow\| over all (sequence, slot) pairs vs only the pairs it takes part in | `conditional_flows.json` | 3 |
| `rotation_wheel_conditional_L4.png` | as above | Write angles, weighted over all pairs vs the pairs each feature takes part in (sparse write features at 0°) | `conditional_flows.json` | 3 |
| `dense_carriers_needed.png` | `circuit_summary` | Spliced: accuracy with the earlier "5", the 15 other dense carriers, random dense and random sparse sets removed | `circuit_extras.json` (dense_drop) | 4 |
| `trace_turning_L4.png` | `circuit_summary` | The stored token's code: cosine with its written code, and its visibility to the output layer, by steps since the write | `circuit_extras.json`, `gates.json` | 5 |
| `rotation_wheel_L4.png` | `rotation_wheel rotation_phases_L4.json` | Angle wheels by steps since the write: keepers vs flippers, and the trace | `rotation_phases_L4.json` | 5 |
| `reversal_tests_L4.png` | `circuit_summary` | Causal tests of what makes the flippers reverse | `reversal_tests_L4.json` | 5 |
| `carry_second_copy_L4.png` | `circuit_summary` | Steering at a later relay step with and without the carried copy (z = 1) | `circuit_tests_L4.json` (carry) | 5 |
| `pop_cancellation_L4.png` | `circuit_summary` | At pop steps: kept part and the candidate's write of the head, memory only / matched / mismatched token fed back | `circuit_extras.json` (pop_split) | 6 |
| `early_feedback_L4.png` | `circuit_summary` | Feeding back a token before its turn cancels that token's own write | `circuit_extras.json` (early_feedback) | 6 |
| `gate_timeshift_L4.png` | `circuit_summary` | Gate features given their value from another step (717 on in the copy phase, 650 off, …) | `circuit_tests_L4.json` (timeshift) | 8 |
| `reset_features_by_length.png` | `circuit_summary` | Reset features removed singly and together at L = 4 vs 7 | `reset_combos_L4.json`, `reset_combos_L7.json` | 8 |
