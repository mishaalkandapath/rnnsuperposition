# RL neuron-baseline autointerp runs

Stores from `python -m circuit.rl_task.neuron_baseline --feature_store rl_features/rl_chosen/chosen_rl_v1_r128_store
--traces rl_features/rl_trace_sequences --out rl_features/rl_neuron_matched_store
--fixed 0.01=rl_features/rl_neuron_d01_store 0.05=rl_features/rl_neuron_d05_store`.
Matched densities: reset 3.57%, update 1.00%, candidate 2.86%. The matched store's update units are not
interpreted (same density as the 1% store; use rl_neuron_d01_autointerp for them).
Workflow: `circuit/rl_task/autointerp_workflows/interpret_units.js`; items in this directory.

| Store | Units | Run id | Results |
|---|---|---|---|
| rl_neuron_d01_store | 192 | wf_5d835d2c-981 | rl_features/rl_neuron_d01_autointerp |
| rl_neuron_matched_store | 144 (no update) | wf_c42783cf-4c4 | rl_features/rl_neuron_matched_autointerp |
| rl_neuron_d05_store | 192 | wf_81b702e8-5a2 | rl_features/rl_neuron_d05_autointerp |

Journals: ~/.claude/projects/-Users-mishaalk-personal-rnnsuperposition/46be5337-94ab-452e-8e5a-9c354e3b2283/subagents/workflows/<run id>/journal.jsonl
Resume: `Workflow({scriptPath: "circuit/rl_task/autointerp_workflows/interpret_units.js", resumeFromRunId: "<run id>", args: {repo, python, store, out, items: <items_neuron_*.json>}})`.
After each run: `python -m circuit.copy_task.autointerp_runs merge <results> <journal>`, then `python -m circuit.rl_task.autointerp report <results>`.
