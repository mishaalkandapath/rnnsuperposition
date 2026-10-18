# features: index (copy task)

Everything the copy-task interpretability analysis reads and writes: about 36 GB. Findings:
`docs/interpretability_findings.md` (autointerp: features vs units, fidelity, the dense-feature probe) and
`docs/interp_circuits.md` (circuits). How to run the autointerp: `docs/autointerp_guide.md` and
`circuit/copy_task/autointerp_workflows/README.md`. Code: `circuit/copy_task/` (each script's docstring has its
command); general pieces in `circuit/`. Figures: `docs/figures/README.md`. RL task: `rl_features/README.md`.

Status: **current** = used by a findings doc or a current figure; **input** = data the analyses read.
Tracked = committed to git; the large inputs are untracked (the store has a Git LFS rule in
`.gitattributes`, unused so far).

## Model and transcoders (input)

| Path | What it is |
|---|---|
| `data/models/copy_train/copy_128_high/copy_128_high.ckpt` | The model: GRU, hidden size 128, copy task with lengths 3–9 (30 tokens a–z, 0–3, plus `<DEL>`) |
| `copy_transcoders/reset_512_sparse_e100.ckpt` (untracked) | Reset-gate transcoder, 512 JumpReLU features |
| `copy_transcoders/candidate_1024_sparse_dpi_final.ckpt` (untracked) | **Update**-gate transcoder, 1024 features (the file name says "candidate"; it is the update gate) |
| `copy_transcoders/candidate_512_updatelike_e100.ckpt` (untracked) | Candidate-gate transcoder, 512 features (the "hidden" gate in stores and records) |

The store's `metadata.pkl` records the checkpoint fingerprints these were matched against.

## Data (input, untracked)

| Directory | Size | What it holds | Made by |
|---|---|---|---|
| `copy_transcoder_v2_1of5/` | 5.4 GB | Gate traces of the model, one file per length (`1M_128_high_seq{3..9}.pt`): inputs, outputs, hidden states and gate values; a 1-in-5 subsample of `data/train_data/copy_transcoder_v2/` (`subsample_manifest.json`: divisor 5, seed 0, kept sequences per file). Every circuit script reads these | subsampling script (manifest) |
| `copy_gru_v1_candiff/` | 22 GB | Pickled feature, metadata and sequence caches of the three transcoders on 1M sequences (`copy_gru_v1_diffcand_r512_{features,metadata,sequences}.p`) | `circuit/copy_task/copy_find_features.py`; converted into the store below |
| `copy_gru_v1_candiff_store/` | 7.0 GB | The feature store: activations per gate (`reset/`, `update/`, `hidden/` = candidate), 1M sequences, plus the step table `autointerp_steps/` (one row per sequence step with the task variables). Every autointerp analysis reads it | `circuit/convert_feature_cache.py`; step table `circuit.copy_task.autointerp` |
| `copy_gru_v1_neuron_baseline_store/` | 226 MB | GRU units as a baseline "store": each unit binarised at the features' firing rate (**matched** density: reset 0.03%, update 0.26%, candidate 0.98%); candidate units at both ends (`neuron_cand_pos/neg`); 200k sequences; `thresholds.json` | `circuit/copy_task/neuron_baseline.py` |
| `copy_gru_v1_neuron_d01_store/`, `copy_gru_v1_neuron_d05_store/` | 311 MB, 948 MB | The same at fixed 1% and 5% density | `circuit/copy_task/neuron_baseline.py --fixed` |

Every store has the layout of `circuit/feature_store.py`: `store.json` (gates, firing counts), `sequences.pkl`,
`metadata.pkl` (provenance) and one directory per gate with one sparse activation file per component.

## Explanations (current, tracked)

| Directory | Count | What it holds | Used in |
|---|---|---|---|
| `copy_gru_v1_candiff_autointerp/` | 512 reset + 1024 update + 512 candidate (`hidden_*`) | Firing explanation of every feature: predicate, label, verdict, test / dev scores, skeptic review, `final_verdict`. Also `summary.csv` (one row per feature), `scorecards/` (fitted scorecards for partial / unexplained features) and `plots/` (the coverage and specificity figures and `coverage_by_f1.json`) | `interpretability_findings.md` §2–4; every label in `interp_circuits.md`; `clamp_tests.py`, `conditional_analyses.py` |
| `copy_gru_v1_neuron_baseline_autointerp/` | 128 × 4 unit groups | Firing explanations of units at matched density, `coverage_by_f1.json`, `scorecards/` | `interpretability_findings.md` §2–3 |
| `copy_gru_v1_neuron_d01_autointerp/`, `copy_gru_v1_neuron_d05_autointerp/` | 128 × 4 each | The same at 1% and 5% density | as above |
| `copy_gru_v1_candiff_magnitude/` (untracked) | 70 | Magnitude explanations of the candidate features firing on ≥ 20% of steps: up to 8 terms fitted on the timing baseline, label, test R² beyond timing, skeptic verdict | `interp_circuits.md` §4; labels in the circuit diagrams |

## Results

| Directory | What it holds | Made by | Used in |
|---|---|---|---|
| `copy_circuits/` | Every circuit analysis result (attribution graphs, swaps, gates, flows, causal tests, minimal circuit, clamp and conditional tests) and their run logs: index in `copy_circuits/README.md` | the scripts in `circuit/copy_task/` | `interp_circuits.md` |
| `explained_fidelity/fidelity.json` (+ `run.log`) | Does the model still work with only the explained features (spliced, mean ablation), core sets by F1, the candidate drop tests | `circuit/copy_task/explained_fidelity.py` | `interpretability_findings.md` §4 |
| `explained_fidelity/dense_set_vs_units.json` (+ `run_dense_set_vs_units.log`) | Candidate: F1 selections of features, with and without the dense 5, vs the same selections of units at three densities; 95% bootstrap intervals and paired-bootstrap differences | `explained_fidelity.py --compare_set` | `interpretability_findings.md` §4 |
| `dense_probe/probe.json` (+ `run.log`) | The 5 essential dense candidate features (162, 214, 284, 388, 487): token tables, two-slot pairs, conditional ablations | `circuit/copy_task/feature_probe.py` | `interpretability_findings.md` §6 (superseded in scope by the magnitude explanations, `interp_circuits.md` §4) |
