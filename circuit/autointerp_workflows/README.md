# Agent-based feature interpretation

Activation-based interpretation of transcoder features (and of GRU units, as a
baseline), in the style of Anthropic's *Towards Monosemanticity*: one agent per
feature proposes what the feature fires on as an exact predicate, which is
scored on held-out sequences. A skeptic agent then tries to refute every
"explained" verdict.

## Pieces

| File | Role |
|---|---|
| `circuit/feature_store.py` | Per-feature on-disk activation store, loaded lazily |
| `circuit/convert_feature_cache.py` | Convert an extractor pickle into a store |
| `circuit/neuron_baseline.py` | Build a store from thresholded GRU gate units (baseline) |
| `circuit/autointerp.py` | The agents' tool: `variables`, `profile`, `score`, `misses`, `record`, plus `report` |
| `circuit/autointerp_workflows/interpret_features.js` | Workflow: one interpreter per transcoder feature, skeptic on "explained" |
| `circuit/autointerp_workflows/interpret_units.js` | Same protocol, for GRU units |
| `circuit/autointerp_runs.py` | Build workflow inputs (`items`), record sparse features (`sparse`), merge skeptic verdicts (`merge`) |
| `circuit/scorecards.py` | Fitted scorecards for features no short predicate explains |
| `circuit/autointerp_plots.py` | Figures (specificity by activation level) |

## Protocol

Each interpreter agent:
1. reads `variables` and the feature's `profile` (rates, lifts and activation mass by
   phase, position, tokens and token history; top and per-quartile examples), on the
   dev split (even sequence ids);
2. scores predicate hypotheses on the dev split and inspects `misses`, at most ~12;
3. picks a verdict: explained / partially_explained / polysemantic / too_sparse /
   dense_uninterpretable / no_explanation;
4. `record`s it, which scores the predicate on the held-out test split (odd ids).

Agents can only read the data through `circuit.autointerp`, so every reported
number is computed by the tool, on sequences the agent never fitted to.

## Running a batch

The `.js` files are Claude Code Workflow scripts: run them from Claude Code with
the Workflow tool. A workflow runs at most min(16, CPUs - 2) agents at once and
at most 1000 agents in total, so use batches of up to about 600 features (each
"explained" verdict adds one skeptic). Independent batches can run as parallel
workflows.

```bash
# 1. features too sparse for an agent (no agent needed)
python -m circuit.autointerp_runs sparse STORE RESULTS --widths reset=512 update=1024 hidden=512

# 2. remaining features, one JSON line per batch
python -m circuit.autointerp_runs items STORE RESULTS --batch 600
```

Then, in Claude Code, for each batch:

```
Workflow({scriptPath: "circuit/autointerp_workflows/interpret_features.js",
          args: {repo: "/abs/path/to/rnnsuperposition", python: "/path/to/python",
                 store: "features/<run>_store", out: "features/<run>_autointerp",
                 items: <one batch from step 2>}})
```

(`interpret_units.js` takes the same arguments, with unit stores from
`circuit/neuron_baseline.py`.) A stopped or interrupted run resumes with
`Workflow({scriptPath, args, resumeFromRunId})`; finished agents replay from cache.

```bash
# 3. after each batch: fold skeptic verdicts into the results (journal path printed at launch)
python -m circuit.autointerp_runs merge RESULTS <transcript dir>/journal.jsonl

# 4. summary table and CSV
python -m circuit.autointerp report RESULTS

# 5. optional: scorecards for partial / unexplained features, and figures
python -m circuit.scorecards STORE RESULTS --workers 5
python -m circuit.autointerp_plots specificity RESULTS
```

## Results layout

`RESULTS/<gate>_<feature>.json` holds the label, predicate, explanation, dev and
test scores (precision, recall, F1, activation-mass recall, coverage by activation
quartile), the skeptic review where there was one, `final_verdict`, and a
`scorecard` summary if fitted (full weights in `RESULTS/scorecards/`).
`RESULTS/summary.csv` collects all features.
