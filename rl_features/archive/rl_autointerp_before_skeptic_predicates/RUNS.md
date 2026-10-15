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
