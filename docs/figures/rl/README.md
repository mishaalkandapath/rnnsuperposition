# RL figures

Figures for `docs/rl_interpretability_findings.md`. All except `specificity_by_activation.png` are made by
`python -m circuit.rl_task.plots <command>` (`plots all` remakes them all); inputs are in `rl_features/` (index:
`rl_features/README.md`). Figures of abandoned analyses: `archived/README.md`.

Three values separated by slashes are for the reset / update / candidate gates.

| Figure | Command | What it shows | Inputs | Doc section |
|---|---|---|---|---|
| `explanation_breakdown.png` | `breakdown` | Left: how the 83 features that matter are explained (firing rule only / + one modifier / just under 0.8 + modifier / discrete magnitude), per gate. Middle: median share of the whole activity (R² beyond timing) an on/off firing rule covers, features vs units at 5% density. Right: where both need magnitude, the 10 magnitude-only features vs the 35 units that matter (timing share of variance, short form, 4-level table) | `rl_fidelity/matter_sets.json`, `firing_coverage.json`, `discrete_levels_*.json`, `rl_refined/`, `rl_autointerp/`, `rl_magnitude*/` | §3, §5, §7 |
| `clock.png` | `clock` | Left: the value head's output over the 100-trial episode. Right: the 18 candidate features active from trial 50 on (behaviour-matched set), each scaled to its maximum and coloured by onset, against the value's fall and the clock profile given to simulators | step table, feature store, `rl_fidelity/matched_sets.json` | §2.3, §6.3 |
| `explanation_tests.png` | `tests` | Features (first explanations / current explanations) vs units. Left: blind words-only simulation on the 100 shared windows (median correlation beyond timing, components that matter). Middle: per-component substitution pass rates. Right: story-test recovery of switch / GO / value per gate | `rl_simulation_shared/`, `rl_simulation_refined/`, `rl_simulation_clock_adopted/`, `rl_fidelity/conditional_tests.json`, `refined_substitution_after.json`, `story_test.json`, `story_test_clock.json` | §6 |
| `conditional_importance.png` | `conditional` | Each component ablated alone (phase × episode-decile mean): GO-choice agreement lost after its worst previous-trial pattern vs averaged over all windows, features vs units; "useful" = at least 5% lost on the worst pattern (features 5 / 6 / 26, units 1 / 4 / 15) | `rl_fidelity/conditional_tests.json` (importance) | §5 |
| `component_substitution.png` | `conditional` | First-round per-component substitution: does each useful component's explanation recover what the model uses from it, where it is used (features vs units) | `rl_fidelity/conditional_tests.json` (substitute) | §6.2 |
| `interp_f1_vs_rate.png` | `interp` | Held-out F1 of each feature's / unit's firing predicate against its firing rate (units at 1%, matched and 5% density) | `rl_autointerp/`, `rl_neuron_*_autointerp/`, feature and unit stores | §3 |
| `interp_f1_bands.png` | `interp` | Share of features / units in each held-out F1 band (≥ 0.8, 0.5–0.8, 0.2–0.5, < 0.2), per gate and density; candidate units' high and low ends shown separately | as above | §3 |
| `interp_verdicts.png` | `interp` | Final verdicts of the feature firing explanations (after the skeptic), per gate | `rl_autointerp/` | §3 |
| `magnitude_features_vs_units.png` | `magnitude` | Magnitude explanations, features vs units: R² of timing alone, of the short form (the recorded explanation), and of the free fit (evidence only), per gate | `rl_magnitude/`, `rl_magnitude_units/` | §4, §7 |
| `specificity_by_activation.png` | `python -m circuit.copy_task.autointerp_plots specificity rl_features/rl_autointerp --out docs/figures/rl/specificity_by_activation.png` | Share of each activation quartile (weakest to strongest active steps) that the firing predicates cover, per gate, split by verdict (explained / partially explained / no explanation): are the weak activations explained too? | `rl_autointerp/` | §6.1 |
