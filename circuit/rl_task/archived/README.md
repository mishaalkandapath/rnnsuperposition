# Archived RL analysis scripts

Scripts of abandoned RL analyses, recovered verbatim from the analysis session's scratch directory (each file is the
exact code that wrote its result: last modified before the result file). They are kept so every file in
`rl_features/rl_fidelity/` can be traced to its code; they are not maintained.

Why the analyses were abandoned: they set the features left out to one global mean (`Ablated` from
`circuit/copy_task/explained_fidelity.py`). That puts phase-structured components off-distribution at every step, and
disagreed with the timing-mean ablation of the importance tests. Every current behaviour test uses phase-mean ablation
instead; the behaviour-matched sets of `circuit/rl_task/matched_sets.py` replace the sets and cores below. Figures:
`docs/figures/rl/archived/README.md`.

They hard-code the repo path (`sys.path.insert`) and the model / transcoder checkpoints; run from the repo root.

| Script | Writes (`rl_features/rl_fidelity/`) | Reads | What it did |
|---|---|---|---|
| `rl_single_drops.py` | `single_drops.json` | — | Update and candidate: behaviour cost of dropping each feature alone (selection sample) |
| `rl_set_tests.py` | `set_tests.json` | `single_drops.json` | Explained features kept, plus the top unexplained "essential" features from the single drops |
| `rl_joint_select.py` | `joint_selection.json` | `single_drops.json` | Greedy forward selection on switch + GO + value together, from the top 80 by single drop |
| `rl_backward.py` | `backward_selection.json` | — | Backward elimination on the combined objective |
| `rl_backward_split.py` | `split_selection.json` | — | Backward elimination separately for the choice and for the value |
| `rl_reset_sets.py` | `reset_sets.json` | `fidelity.json` | Reset gate: single drops, top-k sets and behaviour cores (switch vs value vs variance explained), joint backward elimination |
| `rl_value_vs_switch.py` | `value_vs_switch.json` | `fidelity.json`, `single_drops.json` | Top-k and core sets for the value vs for switching |
| `rl_substitutes.py` | `substitutes.json` | `split_selection.json` | Whether the choice-only and value-only features are substitutes for each other (representational R², nearest-feature swaps) |

The other abandoned results have maintained scripts: `fidelity.json`, `fidelity_units.json` and `fidelity_sweep.json`
(`circuit/rl_task/explained_fidelity.py`), `ranked_tiers.json` (`ranked_fidelity.py --tiers`), `substitution.json`
(`explanation_substitution.py`).
