# Archived RL results

Superseded RL explanation runs and analyses, kept for the record. Nothing current reads from here.
Current results: `rl_features/` (see the RUNS files there and `docs/rl_interpretability_findings.md`).

| Entry | What it is | Why archived | Current version |
|---|---|---|---|
| `rl_autointerp_pilot/` | First firing-explanation pilot (12 features) | Pilot | `rl_autointerp/` |
| `rl_autointerp_before_skeptic_predicates/` | Firing explanations of all features before the skeptic's corrected predicates were adopted (`autointerp.py apply_skeptic`) | Superseded by the skeptic-corrected predicates (the 155 current records whose predicate changed keep the original under `original_*`) | `rl_autointerp/` |
| `rl_neuron_{d01,matched,d05}_autointerp_before_skeptic_predicates/` | The same for GRU units at 1%, matched and 5% density | As above | `rl_neuron_*_autointerp/` |
| `rl_magnitude_v1/`, `rl_magnitude_units_v1/` | First magnitude explanations of features (100) and units (144): free-form terms, up to 8 | Curve-fitting allowed as the explanation; some terms read the model's own value or policy (e.g. units' `tanh(choice)`) | `rl_magnitude/`, `rl_magnitude_units/` (short forms, observed-only) |
| `rl_magnitude_observed_v1/`, `rl_magnitude_units_observed_v1/` | v1 re-explanations without model-output terms (may use belief) | Still free-form (lookup tables, fitted constants) | As above |
| `rl_magnitude_pilot_v1/`, `rl_magnitude_v2_pilot/` | Pilots of v1 and of the v2 short-form protocol | Pilots | As above |
| `compactness_v1_free_fits.json` | How compact the v1 free fits were (features vs units, with and without model outputs) | Built on v1 | Short-form rules in `autointerp_magnitude.py` (`check_short`) |
| `revisit_before/` | Previous magnitude records of the 3 components re-explained in the revisit round: units candidate 0 and 3, feature update 18 (`rl_fidelity/revisit_evidence.json`) | Replaced by the revisited explanations | `rl_magnitude_units/`, `rl_magnitude/` |
| `ranked_v1.json` | First budget-matched ranked fidelity results | Global-mean ablation; first-round tiers (figures in `docs/figures/rl/archived/`) | Behaviour-matched sets and the story test (`rl_fidelity/`) |
| `rl_refined_words/` | Words given to the simulators in the refined-explanation round (24 features) | Duplicated in each result record of `rl_simulation_refined/`; out of date for the 7 features changed in the clock round | `rl_refined/` (current words), `rl_simulation_refined/` |
| `rl_clock_words/` | Words for the clock round (10 features), with the old wording of 182, 130 and 77 | Duplicated in each result record of `rl_simulation_clock/` | `rl_refined/`, `rl_simulation_clock/` |

To resume the two simulation workflows of the last two rows, pass `explanations: "rl_features/archive/<dir>"`
(run ids in `rl_features/rl_simulation_RUNS.md`).
