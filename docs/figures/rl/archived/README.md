# Archived RL figures

Figures from RL analyses that were abandoned or superseded. They are kept for the record only; none is used in
`docs/rl_interpretability_findings.md`. They can be regenerated on request with `python -m circuit.rl_task.plots
<command>`, which writes them here (`plots all` skips them). Inputs are in `rl_features/rl_fidelity/` and
`rl_features/rl_probe/`.

Why they were dropped, in short:
- **Global-mean ablation.** Components left out were set to one global mean. That puts phase-structured components
  off-distribution at every step, and the importance tests used a different (timing-mean) ablation. Every
  behaviour test now uses phase-mean ablation (`circuit/rl_task/matched_sets.py`, `conditional_tests.py`).
- **Model-output variables.** Explanations or cues that read the model's own value or policy leak the decision
  being explained. They are banned now; the ideal-observer `belief` stands in.
- **Flawed or replaced test designs.**

| Figure | Plot command | What it showed | Why archived | Superseded by |
|---|---|---|---|---|
| `fidelity_by_selection.png` | `fidelity` | Switch agreement, GO agreement and value error with only F1-selected, explained, random or core features kept (`fidelity.json`) | Global-mean ablation; explained labels from before refinement | `explanation_tests.png` (story test), `conditional_importance.png` |
| `fidelity_features_vs_units.png` | `fidelity` | The same for features vs thresholded units (`fidelity_units.json`) | Global-mean ablation | `explanation_breakdown.png`, `explanation_tests.png` |
| `ranked_fidelity_keep.png`, `ranked_fidelity_drop.png` | `ranked` | Budget-matched fidelity: keep or drop the top-k components by explanation tier, against an oracle and random orders (`ranked.json`, `ranked_tiers.json`) | Global-mean ablation; tiers from the first explanations | Behaviour-matched sets (`matched_sets.json`) and the story test |
| `sets_behaviour.png`, `sets_overlap.png` | `sets` | Behaviour-ranked top-k sets, choice and value "cores", backward elimination (`reset_sets.json`, `value_vs_switch.json`, `backward_selection.json`, `split_selection.json`) | Single-drop rankings under global-mean ablation | `matched_sets.py` (phase-mean, 90% target) |
| `substitutes.png` | `substitutes` | Whether the choice-only and value-only core features can be swapped for their nearest substitutes (`substitutes.json`) | Built on the abandoned cores | — |
| `time_profiles.png` | `time` | Activation by episode trial for the features in the choice and value cores | Built on the abandoned cores | `clock.png` |
| `explanation_substitution.png` | `substitution` | Whole-set substitution: explained components replaced by their explanations, against mean and timing baselines (`substitution.json`) | Mean-ablation baseline; explanations that read model outputs | Per-component substitution (`component_substitution.png`) and the story test (`explanation_tests.png`) |
| `probe_update_ablation.png`, `probe_update_tables.png`, `probe_candidate_ablation.png`, `probe_candidate_tables.png` | `probe` | Dense-feature probe: conditional ablation, magnitude regression and residual tables for important features without a firing predicate (`rl_features/rl_probe/`) | Conditions included the model's policy and value bins; replaced by magnitude explanations | Magnitude and refined explanations (`magnitude_features_vs_units.png`, `explanation_breakdown.png`) |
| `scorecards_vs_agent.png` | `scorecards` | Agent predicate F1 against fitted 40-cue scorecards | Scorecards used model-output cues (value and policy bins), and a 40-cue scorecard is not a short explanation | — |
| `clamp_effects.png` | `conditional` (clamp section) | Clamp a component high where its explanation says low, and low where it says high; does behaviour move with the situation's natural contrast? | The prediction was flawed: components push consistently, but several signal one situation with opposite weights, so the natural contrast does not predict each one's push | Per-component substitution (`component_substitution.png`) |
