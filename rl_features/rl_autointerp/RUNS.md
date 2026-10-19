# RL autointerp runs (firing pass)

Store: `rl_features/rl_chosen/chosen_rl_v1_r128_store` (step table built with
`python -m circuit.rl_task.autointerp build STORE --traces rl_features/rl_trace_sequences`).
Workflow: `circuit/rl_task/autointerp_workflows/interpret_features.js`, results in this directory.
Sparse / never-firing features (42) were recorded without agents (`autointerp_runs sparse`).

| Gate | Features | Run id | Journal |
|---|---|---|---|
| reset | 128 | wf_c4a97f13-1e7 | ~/.claude/projects/-Users-mishaalk-personal-rnnsuperposition/46be5337-94ab-452e-8e5a-9c354e3b2283/subagents/workflows/wf_c4a97f13-1e7/journal.jsonl |
| update | 223 | wf_7229c695-48b | .../workflows/wf_7229c695-48b/journal.jsonl |
| hidden | 247 | wf_3a07449f-bc2 | .../workflows/wf_3a07449f-bc2/journal.jsonl |

Resume an interrupted run (finished agents replay from cache):
`Workflow({scriptPath: "circuit/rl_task/autointerp_workflows/interpret_features.js", resumeFromRunId: "<run id>", args: <same args>})`.

After each run: `python -m circuit.copy_task.autointerp_runs merge rl_features/rl_autointerp <journal>`,
then `python -m circuit.rl_task.autointerp report rl_features/rl_autointerp`.

Pilot (12 features, earlier prompt without the episode-trial table): `rl_features/rl_autointerp_pilot`, run wf_0b7946f3-dd5.

Observed-only round (2026-10-18): the 31 features that matter whose recorded rule used model outputs (value, choice,
model_act) or the upcoming choice were re-explained with observed variables only (`autointerp score --observed`):
22 by one Agent-tool subagent each plus 5 skeptic subagents (not a saved workflow), 8 by swapping the window-start
choice stand-in for `belief`, update 235 in the main session. Records in `rl_features/rl_observed/`; merged here
(each keeps its earlier rule under `with_model_outputs`). Findings doc §3.
