# rl_features: index

Everything the RL (two-step task) interpretability analysis reads and writes: 6.9 GB, untracked. Findings:
`docs/rl_interpretability_findings.md`. How each run was made: the RUNS files below. Superseded results:
`archive/README.md`. Code: `circuit/rl_task/` (each script's docstring has its command).

Status: **current** = used by the findings doc or a current figure; **input** = data the analyses read;
**abandoned** = from an analysis that was dropped (kept for the record; figures in `docs/figures/rl/archived/`,
scripts in `circuit/rl_task/archived/`). Every result file below names the script that wrote it.

## Run logs

| File | What it records |
|---|---|
| `rl_magnitude_RUNS.md` | Magnitude-explanation runs (features and units): workflow run ids, item lists, revisit round |
| `rl_simulation_RUNS.md` | Blind words-only simulation runs (own windows, shared windows, refined, clock) |
| `rl_refined_RUNS.md` | Refined explanations, their tests, and the clock round (what was adopted) |
| `rl_neuron_runs/RUNS.md` | Unit firing-explanation runs at 1% / matched / 5% density; their item lists are `rl_neuron_runs/items_neuron_{d01,matched,d05}.json` |
| `rl_autointerp/RUNS.md` | Feature firing-explanation runs (one workflow per gate), the sparse features recorded without agents, the pilot |

## Data (input)

| Directory | Size | What it holds | Made by |
|---|---|---|---|
| `rl_trace_sequences/` | 1.9 GB | Recorded trace windows of the agent (5 trials × 3 steps each): inputs, outputs, previous and new hidden states, reset and update gate values, the core trial's pattern; 8 files = 2 start conditions (high- / low-first) × 4 core-trial patterns (`commonp` common rewarded, `common_p` common unrewarded, `uncommonp` / `uncommon_p` the rare versions). All 8 are loaded (`*_rl.pt`) | `datasets/transcoder_rl_datasets.py` |
| `rl_chosen/chosen_rl_v1_r128_store/` | 0.8 GB | The chosen transcoders' feature store: activations per gate (`reset/`, `update/`, `hidden/` = candidate), plus the step table `autointerp_steps/` (one row per window step, with the task variables, `belief` and `surprisal`). Every analysis reads it | `circuit/rl_find_features.py`; step table `circuit.rl_task.autointerp build` + `belief` |
| `rl_chosen/*.p` | 2.4 GB | Pickled feature, metadata and sequence caches of the same store (`chosen_rl_v1_r128_{features,metadata,sequences}.p`) | `circuit/rl_find_features.py` |
| `rl_unit_values/` | 1.2 GB | Raw GRU gate values per step for all 48 units (`reset.npy` r_t, `update.npy` z_t, `candidate.npy` n_t), aligned with the step table; `meta.json` records the source store, traces, row count and which trace array each file came from | `circuit.rl_task.neuron_baseline --raw` |
| `rl_neuron_d01_store/`, `rl_neuron_matched_store/`, `rl_neuron_d05_store/` | 0.6 GB | Unit "stores": each unit binarised at 1%, at the features' density (matched) and at 5%; candidate units at both ends (`neuron_cand_pos/neg`). Each has its own step table (`autointerp_steps/`) and `thresholds.json` (the cut-off per unit) | `circuit.rl_task.neuron_baseline` |

Every store (the feature store and the unit stores) has the layout of `circuit/feature_store.py`: `store.json` (gates,
firing counts, settings), `sequences.pkl` (the trace-window keys, in row order), `metadata.pkl` (provenance: for the
feature store the transcoder checkpoints and their fingerprints and the trace files; for a unit store the source
store and density rule) and one directory per gate with one `<id>.npz` of sparse activations per component.

## Explanations (current)

| Directory | Count | What it holds | Why it matters |
|---|---|---|---|
| `rl_autointerp/` | 640 | Firing explanation of every feature: predicate, label, verdict, test / dev scores (skeptic-corrected; the 155 whose predicate changed keep the original under `original_*`; the 31 features that matter whose rule used model outputs now hold their observed-only rule from `rl_observed/`, the earlier one under `with_model_outputs`). Also `summary.csv` (one row per feature) and `scorecards/` (376 fitted 40-cue scorecards: abandoned, they use model-output cues) | The 58 important features explained by a firing rule alone; features vs units firing comparison |
| `rl_observed/` | 31 | Observed-variables-only firing rules for the 31 features that matter whose first rule used the model's outputs or upcoming choice: label, predicate, verdict, test / dev scores, the earlier rule (`previous_predicate`, `previous_test_f1`); `skeptic/` holds the skeptic verdicts for the 22 re-explained by agents | findings doc §3 (observed variables only) |
| `rl_neuron_d01_autointerp/`, `rl_neuron_matched_autointerp/`, `rl_neuron_d05_autointerp/` | 192 / 144 / 192 | Firing explanations of units at each density (matched has no update units: same as 1%), each with a `summary.csv` | The unit side of the firing comparison |
| `rl_magnitude/` | 175 | Magnitude explanations of features: short form (at most 3 terms) + label, fitted on top of the timing baseline | The only magnitude explanation for 151 features; the "first explanations" baseline for the other 24 |
| `rl_magnitude_units/` | 144 | Magnitude explanations of all units | Unit magnitude comparison, unit substitution, timing shares |
| `rl_refined/` | 24 | The current explanations of the important features that needed more than a firing rule: kind (firing + modifier, just under 0.8, discrete magnitude), words, predicate, terms, scores | The current breakdown and every "current explanations" result |
| `rl_magnitude_targets.json`, `rl_magnitude_v2_targets.json`, `rl_observed_only_targets.json` | | Item lists for the magnitude runs (v1, v2, observed-only) | Reproducing those runs |

## Blind simulations

Each record holds the words its simulator was given, the predictions and the scores; `chunks/` holds the
simulators' per-chunk predictions before they were combined.

| Directory | What it holds | Status |
|---|---|---|
| `rl_simulation/` | Each member of the behaviour-matched sets (270 features, 75 units) on its own held-out windows, with the words it was explained by | current (§6.1 own-window numbers) |
| `rl_simulation_shared/` | The same 345 members, from those words, on one set of 100 shared held-out windows | current; the story test's base, and the units' shared-window scores |
| `rl_simulation_refined/` | The 24 refined features' words on the shared windows | current (superseded for the 8 below) |
| `rl_simulation_clock/` | Clock-round words of 10 features (with the shared clock definition) | current |
| `rl_simulation_clock_adopted/` | The 8 adopted clock-round results (not 130, 77) | current; read first by the story test |
| `rl_simulation_pilot/`, `rl_simulation_shared_pilot/` | Pilots of the two simulation set-ups | pilots |

## Tests and analyses: `rl_fidelity/`

| File | What it holds | Made by | Status |
|---|---|---|---|
| `conditional_tests.json` | Sections: `importance` (each component ablated alone; overall / active windows / worst pattern), `clamp` and `diagnose` (clamp tests), `substitute` (first-round per-component substitution) | `conditional_tests.py` | current (importance, substitute; diagnose read by `revisit_evidence.py`); clamp abandoned |
| `matter_sets.json` | The components that matter (87 features, 35 units) | `matter_sets.py` | current |
| `matched_sets.json` | Behaviour-matched sets (fewest features / units for 90% recovery, phase-mean ablation) | `matched_sets.py` | current |
| `refined_substitution_before.json`, `refined_substitution_after.json` (+ `run_*.log`) | Per-component substitution of the 24 refined features with their first / refined explanations | `conditional_tests.py --components` | current |
| `story_test.json`, `story_test_refined.json`, `story_test_clock.json` (+ `run_*.log`) | Whole matched set replaced by its words: first words / first refined words / current (clock round) | `story_test.py` | current (`story_test_clock` is the result in the doc) |
| `story_screen_candidate.json` | Value and switch gain if one candidate feature at a time had a perfect simulator | `story_screen.py` | current (clock diagnosis) |
| `readout_test.json` | Decision variables read out from features vs units vs randomly rotated units | `readout_test.py` | current |
| `firing_coverage.json` | Firing rules on the fair sets: explained rates and on/off R² of the whole activity, features vs units | `firing_coverage.py` | current |
| `discrete_levels_units.json`, `discrete_levels_features.json` | Magnitude short forms vs their discrete-only and 3 / 4-level tables | `discrete_levels.py` | current |
| `firing_vs_magnitude.json` | Behaviour with firing-explained (or firing / magnitude-explained) features kept or replaced by explanations | `firing_vs_magnitude.py` | current (§5, first-round explanations) |
| `simulation_items.json`, `shared_items.json` | Item lists of the simulation runs (matched-set members; with their words) | `simulation_items.py` | current (reproducing runs) |
| `revisit_evidence.json` | Evidence for the 6 components re-explained in the revisit round (a snapshot of that time) | `revisit_evidence.py` | current (history of those explanations) |
| `fidelity.json`, `fidelity_units.json`, `fidelity_sweep.json` | Explained-feature fidelity (features, units, sweep over F1 thresholds) | `explained_fidelity.py` | abandoned (global-mean ablation) |
| `ranked_tiers.json` | Budget-matched ranked fidelity by explanation tier | `ranked_fidelity.py --tiers` | abandoned (global-mean ablation) |
| `single_drops.json`, `reset_sets.json`, `value_vs_switch.json`, `backward_selection.json`, `split_selection.json`, `joint_selection.json`, `set_tests.json` | Single-drop costs, behaviour-ranked sets and cores, greedy selections | `archived/rl_*.py` (`circuit/rl_task/archived/README.md`) | abandoned (global-mean ablation; replaced by `matched_sets.py`) |
| `substitutes.json` | Nearest substitutes for the core features | `archived/rl_substitutes.py` | abandoned |
| `substitution.json` | Whole-set substitution by explanations (mean baseline, model-output terms) | `explanation_substitution.py` | abandoned |
| `recompute_v2.log`, `recompute_v3.log` | Logs of recomputing the fidelity results after rule changes | — | abandoned |

## Other

| Entry | What it holds | Status |
|---|---|---|
| `rl_circuits/` | Circuit analysis: contrastive attribution graphs, credit factorial, reversal switches, minimal-circuit and steering checks (`rl_circuits/README.md`; doc `docs/rl_circuits.md`) | current |
| `rl_probe/` | Dense-feature probe results (`update_probe.json`, `candidate_probe.json`), from `circuit/rl_task/feature_probe.py` | abandoned (conditions used the model's policy / value bins; replaced by magnitude explanations) |
| `archive/` | Superseded explanation runs and words (`archive/README.md`) | archived |