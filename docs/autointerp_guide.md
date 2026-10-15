# Guide: auto-interpreting transcoder features (any task)

Written for a Claude Code session that has to interpret transcoder features of a new model or task. It sums up
what worked on the copy-task GRU (`docs/interpretability_findings.md`, `docs/interp_circuits.md`) and what we
learned the hard way. The copy-task code is in `circuit/copy_task/`; the general pieces are in `circuit/`.

There are two kinds of explanation, and choosing the right one per feature is the first decision:

- **Firing explanations** say *when a feature is on* (a yes/no predicate). Right for features that are active in few
  examples.
- **Magnitude explanations** say *what drives how strongly it fires*. Needed for features that are active in most
  examples: when they fire tells you little; their information is in their size.

---

## 1. Before any agent runs

1. **Feature store.** Activations per feature, loaded lazily: `circuit/feature_store.py` (convert an extractor pickle
   with `circuit/convert_feature_cache.py`). Store the transcoder checkpoint fingerprints with it so you never
   interpret one checkpoint with another's activations.
2. **Dev / test split by example, not by step.** Copy task: even sequence ids = dev (agents look and fit here), odd =
   test (every reported number). Never let an agent see test-split examples.
3. **A step table: one row per (example, step) with the variables an explanation may use.** This is the main
   task-specific piece. Include:
   - time / position (absolute step, steps to go, phase of the task);
   - the current input and the target / output at the step;
   - history lookups (input d steps back, output d steps ahead, the k-th item of the example, "seen X before");
   - any task variables that plausibly matter (copy task: sequence length, correct-so-far, token counts;
     an RL task: state, last action, reward, trial phase, …).
   The copy version is `build_step_table` and the helper namespace in `circuit/copy_task/autointerp.py`. For a new task,
   write a new table + helpers but **keep the same command-line interface** (`variables`, `profile`, `score`,
   `misses`, `record`, `report`), so the workflow prompts carry over.
4. **The agent's only window on the data is that tool.** Agents must not read store files directly; every number they
   report must come from the tool, computed on the test split.

## 2. Decide firing vs magnitude, per feature

Measure **density across examples**: the fraction of examples (sequences / episodes) in which the feature is active.
Firing on many steps *within* an example is normal and does not make a feature dense.

| feature | explanation to use |
|---|---|
| active in a minority of examples | firing explanation (§3) |
| active in most examples (copy task: ≥ 90% of sequences for 65 of 70 such features) | magnitude explanation (§4) |
| firing label scores well but only says "phase X" / "everywhere" | it is probably dense: also do §4 |

Warning sign: a trivial predicate (`t >= 0`, "copy phase") reaching F1 0.9+. That label is true and useless. In the
copy task these "explained" dense features turned out to carry most of the memory.

## 3. Firing explanations (agents + skeptic)

Tools: `circuit/copy_task/autointerp.py` (CLI), `circuit/copy_task/autointerp_workflows/interpret_features.js`
(workflow), `circuit/copy_task/autointerp_runs.py` (batches, merging), `circuit/copy_task/scorecards.py`.
See `circuit/copy_task/autointerp_workflows/README.md` for exact commands.

Per feature, one agent:
1. reads `variables` and the feature's `profile`: rate, lift and activation mass by every variable, top examples and
   random examples from each activation quartile (dev split);
2. scores predicate hypotheses on dev, uses `misses` on the best one, at most ~12 tries, prefers the simplest
   predicate within a few points of the best;
3. picks a verdict: explained (F1 ≥ 0.8, or mass recall ≥ 0.8 with high precision) / partially explained /
   polysemantic / too sparse (< ~20 examples) / dense uninterpretable / no explanation;
4. `record`s it: this scores the predicate on the **test** split.

Then a **skeptic agent** tries to refute every "explained" verdict: checks each claim with targeted predicates, runs
`misses` on test, looks for systematic (not scattered) failures, proposes a better predicate. Default to "not upheld"
when the evidence is mixed. Fold verdicts in with `autointerp_runs.py merge`.

For features with no short predicate, fit **scorecards** (L1 logistic models over one-hot cues; full / top-k / two-slot
forms). They show how much a readable label loses.

Practical limits: a Workflow runs at most min(16, CPUs − 2) agents at once and at most 1000 agents in total, so batch
~600 features per workflow (skeptics add more). Features too sparse for an agent can be recorded without one
(`autointerp_runs.py sparse`).

Always add a **baseline**: interpret the model's own units the same way (`neuron_baseline.py`,
`interpret_units.js`), at matched density, to check whether the features are actually more interpretable.

## 4. Magnitude explanations (for features dense across examples)

### Why they are needed

A feature active in nearly every example carries information only in how strongly it fires. In the copy task the
token held in memory was carried almost entirely by such features; their firing labels ("copy phase") were right
and said nothing about it. Removing 15 of them dropped copy accuracy to 16% (random sparse sets: 86–88%).

### How to score them

Explain the activation *beyond a baseline*, on held-out examples:

- **Baseline** = the mean activation at the same position / time (copy task: per (length, step)). Timing alone is the
  trivial part.
- **Score** = R² beyond baseline = 1 − SSE(explanation) / SSE(baseline): the share of what timing leaves unexplained
  that the explanation captures. Fit any weights on dev, score on test.
- **Reference models** (`circuit/copy_task/magnitude_scores.py`, fast: one-hot design, X'X from co-occurrence counts,
  one pass for all features):
  - additive tables over cues (current input, inputs 1–7 back, outputs 0–3 ahead, first item), phase-specific: the
    ceiling for additive explanations;
  - a compact form: one set of items that raises the feature, one that lowers it, a weight per cue: what a short
    label can reach.

Copy-task numbers (70 features, median R² beyond timing): last 2 inputs 0.27, last 8 inputs 0.57, full tables 0.79,
compact sets 0.53, agent-written 0.65.

### The agent pass

Tool: `circuit/copy_task/autointerp_magnitude.py` (`profile`, `score`, `misses`, `record`, `report`). An explanation is
**up to 8 numeric terms** (numpy expressions over the step variables, e.g. "item in set S at 1–3 steps back, input
phase only"); the tool fits one weight per term on dev plus the baseline and scores on test. The profile shows the
timing profile, the fitted tables, the compact-set fit, both reference scores, and examples of activation minus
baseline. Verdicts: explained (test R² beyond baseline ≥ 0.6, or ≥ 0.45 and ≥ 80% of the full-table ceiling),
partially explained (≥ 0.3), no explanation. A skeptic checks every "explained" one, as in §3. Workflow:
`circuit/copy_task/autointerp_workflows/interpret_magnitude.js` (args: `repo`, `python`, `store`, `out`, `features`;
run `magnitude_scores.py` for the gate first, since the profile reads its reference scores). For a new task, also
change its `TASK` text and the variable hints in `TOOLS`. Write records to a separate directory (we used
`features/<run>_magnitude/`), never over the firing results.

### What magnitude codes looked like (copy task)

- **item sets over a history window**: "high when the last ~4 inputs are in {e, b, g, h}", weights fading with lag;
- **signed axes**: one set raises it, another lowers it;
- **sign flips between phases**: the same item raises the feature as an input and lowers it as a stored item;
- **slot memory**: depends on what is stored in a fixed slot (the k-th item), not just recent inputs;
- **queue reads**: depends on the item being output now and the next ones waiting.

### Check that a magnitude explanation is real

- **Steer by the tables**: move the feature's activation by (mean given item A − mean given item B) and check the
  model's output changes to A's behaviour. Control: the same edit with the table rows shuffled (same size of push).
  In the copy task this flipped outputs 86–100%; shuffled: 0–3%.
- **Item-conditional ablation**: remove the feature and split the damage by item. A real carrier hurts mostly the
  items in its raising set (copy task: 69–83% on its items, ~95% on others).

### "Explained" is not the same as "said in words"

A fitted magnitude explanation (weights per term, token tables copied from fits) can score well and still not be a
description a person could use. Anthropic's standard is a verbal description from which a reader predicts the
activations (simulation scoring by rank correlation). To check: write the explanation as a rule in words, compute it
using only the numbers the words state (nothing fitted), and score its rank correlation with activation minus the
baseline on test data, next to the fitted version scored the same way (`circuit/copy_task/verbal_rules.py`). In the copy
task a one-sentence rule for the most important dense feature kept ~80% of the fitted rank correlation (0.59 vs 0.75),
and most other important dense features fit the same sentence template. Report the two separately: "accurately
modelled" and "can be said in words".

### Merged features are a transcoder issue

Many dense features combined two variables: a timing / phase signal (on for every item) plus an item set (2–7× higher
for its items). The model uses the item part. This is the dictionary failing to factor the variables. Report it and
fix it by training the transcoder better; don't relabel features as sparse to paper over it.

## 5. Validate with interventions

Labels are hypotheses. Test the important ones causally, and say which kind of intervention each result is:

1. **real model, no features** (change inputs or whole gate vectors): says what the model does;
2. **real model, edited along a feature direction** (compute the real value, then change only that feature's share:
   value − (a_f − replacement)·d_f, keeping the transcoder error): isolates the feature; baseline = the exact model;
3. **spliced** (the transcoder replaces the layer): adds the transcoder's own error, which compounds over a recurrence.

Always include random controls of the same size (and for directions, random directions of the same size).

**Test each important label's effect, not only its activations** (`circuit/copy_task/clamp_tests.py`). Derive the
predicted effect from the label and clamp the feature in the situation it describes:
- "input X" → switch it on where the input is not X: the model should behave as if X were there (and off where it
  is X: X should be lost);
- a magnitude label with raising / lowering sets → boost it: outputs should move towards the raising set;
- control: the rank of the predicted item among all items, or random sets of the same size.

In the copy task 15 of 16 magnitude labels and 13 of 29 sampled token labels passed cleanly; a third of the
"explained" token features did nothing or the opposite.

**Measure a feature where it takes part** (`circuit/copy_task/conditional_analyses.py`): its effect on the examples
where it is active and on the output item it hurts most, not averaged over every example. Averaged rankings pick
dense features only, because a feature used for one item in thirty is diluted thirty-fold. Use mean, zero and time-shift
ablations together, and test at the lengths / conditions a label refers to (late-position features exist only in long
sequences).

## 6. Pitfalls we hit

- **Selection effects.** Keeping "features with F1 ≥ τ" silently drops features whose *firing* is unpredictable,
  however important; adding back just 5 dense features changed copy fidelity from 55% to 98.5%. Check what a
  threshold excludes.
- **Coverage is a weak metric.** A few dense features cover every step.
- **Mean ablation does nothing to a feature that is constant at a given step** (e.g. a delimiter detector): its mean
  is its value. Use zero, or a time-shift (give it its value from another step). Same-step resampling cannot remove
  a timing signal either.
- **Make sure edits reach downstream computation.** We once edited the GRU's reset gate *after* the candidate had
  been computed from it, so every reset intervention silently did nothing. Re-derive anything that depends on an
  edited quantity, and sanity-check one edit against an independent implementation.
- **Linear attribution gives "partner credit" to AND-like features.** A feature that needs A and B gets credit split
  over both, so single-example attribution can point at inputs that do not matter. Use contrastive comparisons and
  counterfactual tests before reading credit as dependence.
- **Averaging over all examples hides item-specific features.** In the copy task the averaged ranking found 22
  important features; measured where each takes part, 190 cost ≥ 20%.
- **Many features share a label with a used feature and do nothing.** 44% of the copy-task candidate features had no
  measurable effect; most were low-norm near-duplicates (decoder norm × activation median 0.15–0.23, against 1.1–2.0 for
  used sparse features). A label cannot tell them apart: check the write size and test the effect.
- **Quote the tool, not the agent.** Every number in a write-up should come from a recorded tool output on test data.

## 7. Adapting to a new task

Change:
- the step table and helper namespace (§1.3), the variable docs, and the phase names;
- the task description text in the workflow prompts (`TASK` in the `.js` files);
- the baseline for magnitude scoring (what "position / time" means), and the cues in `magnitude_scores.py`
  (e.g. last actions / rewards / states instead of tokens).

Reuse as is: `feature_store.py`, the CLI shape, the workflow structure (interpreter + skeptic, test-split scoring),
the scorecard idea, the magnitude scoring design (baseline, R² beyond baseline, additive tables, compact sets,
term-based agent explanations), and the intervention checks.
