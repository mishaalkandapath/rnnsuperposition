# RL magnitude-explanation runs

Tool: `circuit/rl_task/autointerp_magnitude.py`; workflow: `circuit/rl_task/autointerp_workflows/interpret_magnitude.js`.
Store: `rl_features/rl_chosen/chosen_rl_v1_r128_store`. Unit raw values: `rl_features/rl_unit_values`
(`python -m circuit.rl_task.neuron_baseline ... --raw rl_features/rl_unit_values`).
## Current (v2): one-sentence label + short form, observed-only

The recorded explanation is a one-sentence label plus a short form (at most 3 simple terms, no hand-set constants or
lookup tables; enforced by the tool), which is what gets scored. The free fit (up to 8 terms) is kept as evidence
only. Model-output terms (value, choice, model_act) are forbidden; the ideal-observer `belief`
(`python -m circuit.rl_task.autointerp belief STORE`) is available instead. Skeptic corrections are adopted
(`python -m circuit.rl_task.autointerp_magnitude apply_skeptic STORE DIR`, after `merge`).
Targets (`rl_features/rl_magnitude_v2_targets.json`): every feature with an earlier magnitude explanation (172) and
all GRU units' raw r / z / n (144).

| Items | Run id | Results |
|---|---|---|
| 172 features | wf_3939a8c0-8cc | rl_features/rl_magnitude |
| 96 units (reset, update) | wf_66d25187-996 | rl_features/rl_magnitude_units |
| 48 units (candidate) | wf_e666e342-529 | rl_features/rl_magnitude_units |
| pilot (6) | wf_dd68cc1d-f44 | rl_features/archive/rl_magnitude_v2_pilot |
| re-explained with intervention evidence: update 5, 18, hidden 8, 204 | wf_aeabfddb-766 | rl_features/rl_magnitude |
| re-explained with intervention evidence: unit_candidate 0, 3 | wf_dc1201a4-883 | rl_features/rl_magnitude_units |

The re-explained components are the explained, useful ones whose explanation failed the per-component substitution
test (`python -m circuit.rl_task.conditional_tests --sections substitute`); evidence passed to the agents is in
`rl_features/rl_fidelity/revisit_evidence.json`, their previous records in `rl_features/archive/revisit_before/`.

## Archived (v1, free-form terms; superseded): rl_features/archive/*_v1

| Items | Run id |
|---|---|
| 100 features (behaviour-critical, non-explained) | wf_808faa6a-90a |
| 96 + 48 units | wf_40b31b76-c09, wf_11f35d1a-56a |
| observed-only re-explanations, 128 features / 78 units | wf_f68d779d-811, wf_cf8c78b0-1ea |
| pilot (3) | wf_d7b26578-e6f |

Journals: ~/.claude/projects/-Users-mishaalk-personal-rnnsuperposition/46be5337-94ab-452e-8e5a-9c354e3b2283/subagents/workflows/<run id>/journal.jsonl
After each run: `python -m circuit.copy_task.autointerp_runs merge <results> <journal>`;
`python -m circuit.rl_task.autointerp_magnitude report <results>`.
