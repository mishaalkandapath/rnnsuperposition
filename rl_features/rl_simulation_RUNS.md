# RL simulation-scoring runs (words-only explanations, blind simulation)

Tool: `circuit/rl_task/simulation_scoring.py`; workflow: `circuit/rl_task/autointerp_workflows/simulation_scoring.js`.
Components: the behaviour-matched sets of `circuit/rl_task/matched_sets.py` (`rl_features/rl_fidelity/matched_sets.json`:
fewest features / fewest GRU units per gate that recover 90% of the way from nothing kept to the full feature
transcoder on switch, GO and value; phase-mean ablation); item lists in `rl_features/rl_fidelity/simulation_items.json`.
Results: `rl_features/rl_simulation/<gate>_<id>.json` (explanation + scores).

| Items | Run id |
|---|---|
| 168 features (reset 55, update 113) | wf_90586973-d37 |
| 102 candidate features | wf_2429efbf-22d |
| 75 units (reset 33, update 12, candidate 30) | wf_1b65827f-a70 |
| pilot (6) | wf_9d13f032-e83 (rl_features/rl_simulation_pilot) |

## Shared held-out windows (the same 100 windows for every component)

Workflow: `circuit/rl_task/autointerp_workflows/simulate_shared.js` (each simulator reads only its component's recorded
explanation, via `show_explanation`); results: `rl_features/rl_simulation_shared/<gate>_<id>.json` (scores and
per-step predictions), used by `circuit/rl_task/story_test.py` (the words-implied activations replace the sets).

| Items | Run id |
|---|---|
| 168 features (reset, update) | wf_a65409a6-76b |
| 102 candidate features | wf_d10ef73e-29f |
| 75 units | wf_b32f93d8-ca3 |
| pilot (2) | wf_29ceccc2-e50 (rl_features/rl_simulation_shared_pilot) |
| 24 refined explanations (rl_features/archive/rl_refined_words, from rl_features/rl_refined) | wf_45dcd83f-d64 (rl_features/rl_simulation_refined) |
| 10 clock-round explanations (rl_features/archive/rl_clock_words; args.context = refined_explanations.CLOCK) | wf_67eb7750-8bd (rl_features/rl_simulation_clock) |

The words directories were moved to rl_features/archive/ after these runs; to resume either run, pass
`explanations: "rl_features/archive/<dir>"` (the run ids above recorded the old paths). Each result record also
stores the exact words its simulator was given.

Journals: ~/.claude/projects/-Users-mishaalk-personal-rnnsuperposition/46be5337-94ab-452e-8e5a-9c354e3b2283/subagents/workflows/<run id>/journal.jsonl
