# Refined explanations of the features that matter and need more than a firing rule

Tool: `circuit/rl_task/refined_explanations.py` (the explanations are the `SPECS` list there; the script re-scores
them and writes `rl_features/rl_refined/<gate>_<id>.json`). Originals are untouched in `rl_features/rl_autointerp`
and `rl_features/rl_magnitude` (each record keeps their label and test score under `firing_record` /
`magnitude_record`).

    python -m circuit.rl_task.refined_explanations --out rl_features/rl_refined [--only update/18 ...]

Kinds: `firing+modifier` (firing predicate, test F1 >= 0.8, plus at most one small magnitude modifier),
`firing~+modifier` (the same with F1 just under 0.8, recall capped by window-first steps whose previous trial lies
outside the window), `magnitude` (no clean firing boundary; a discrete magnitude explanation).

Scores: `firing` (F1 on the full dev / test splits), `magnitude` (standard fit: terms on top of the phase x
episode-decile timing baseline), `gated` (as the words read: zero outside the predicate, fitted terms inside it,
no timing baseline added; still R^2 beyond timing), and the `*_onoff` versions of both (predicate alone).

Tests of the refined explanations (all 24):
- Per-component substitution (`circuit.rl_task.conditional_tests --sections substitute --components ... --importance_from
  rl_features/rl_fidelity/conditional_tests.json`, 6000 held-out windows, seed 0): old explanations
  `rl_features/rl_fidelity/refined_substitution_before.json`, refined (`--feature_refined rl_features/rl_refined`, used as
  worded) `rl_features/rl_fidelity/refined_substitution_after.json`.
- Blind words-only simulation on the 100 shared windows: words in `rl_features/archive/rl_refined_words` (each
  result record also stores its words), results in
  `rl_features/rl_simulation_refined` (run id in `rl_features/rl_simulation_RUNS.md`); old scores in
  `rl_features/rl_simulation_shared`.

Clock round (value in the story test). A perfect-simulator screen (each candidate set member given its true
activation, one at a time) showed the candidate value loss sits in the time-dependent explanations (204, 215, 8, 182,
60, 130, 24, 77 the largest). One shared profile, "the clock" (`CLOCK` in refined_explanations.py: the shape of the
value head's late fall), is given once to every simulator (`simulate_shared.js` `args.context`), and explanations say
only "follows the clock (from trial X)"; level errors were fixed with one clause each ("about a quarter as strong").
- Words: `rl_features/archive/rl_clock_words` (8, 204, 60, 66, 24, 215, 250 = the current rl_refined words; 182, 130, 77 =
  edited old words). Simulations: `rl_features/rl_simulation_clock` (run id in rl_simulation_RUNS.md). For those 7
  refined features the rl_simulation_refined records are of their earlier words.
- Adopted (`rl_features/rl_simulation_clock_adopted`): 8, 204, 60, 66, 24, 215, 250, 182. Not adopted: 130 (value
  recovery -0.04) and 77 (switch -0.08), which keep their old words.
- Story test with the adopted set: `rl_features/rl_fidelity/story_test_clock.json` (earlier: `story_test.json` old
  words, `story_test_refined.json` first refined words).
- Substitution after the term changes (66: rare-route level + reward surprisal; 250: belief level; 215: late rise):
  `rl_features/rl_fidelity/refined_substitution_after.json` (re-run, all 24).

Development: firing rules for reset 18, update 150 / 181, candidate 11 / 60 / 183 / 212 came from one refinement agent
per feature (dev only, test scored once; this session's agents, not a saved workflow); candidate 36, 208, 66 and
update 18 were worked out in the main session. Candidate 66 uses the `surprisal` step-table variable added for it
(`circuit.rl_task.autointerp belief STORE` builds `surprisal`, `surprisal_transition`, `surprisal_reward`).
