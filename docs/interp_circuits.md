# Copy circuits in the GRU: how a token at position i is output at step L + i

Circuit analysis of the copy-task GRU (hidden size 128) with its three transcoders (reset, update, candidate), in the
style of Ameisen et al. (2025), *Circuit Tracing*. Mostly length 4 (L = 4), some length 3 and 7.
Feature labels come from `docs/interpretability_findings.md` (firing labels) and from the magnitude pass below.

**Task layout.** A sequence is L tokens, then `<DEL>` at step L, then the model's own outputs fed back. At step L + j the
model outputs token j. Step names: `in0 … in(L-1)` (inputs), `DEL`, `out1 … out(L-1)`. At `out k` the input is the
token output one step earlier (token k-1) and the model outputs token k. Token j is written at `in j` and output at
`L + j` (output 0 at `DEL`).

## Summary

1. **The model is a queue, not a position counter.** Each output depends only on its own input token. The model
   outputs the oldest token still in memory, and a token leaves memory only when it is fed back ("pop"). Skipping a
   copy step delays the rest of the output by one; it never skips.
2. **The circuit has four stages**, all in candidate features:
   - **write:** a sparse "input g"-type feature plus dense token-set features pick the token up;
   - **relay:** dense features hold it and hand it on each step;
   - **read-out:** dense features write it into the logit at its output step;
   - **pop:** the fed-back token cancels its own stored copy in the candidate's next write.

   About 10 candidate features per step (of 512) are both sufficient and necessary to carry a token from input i to
   output L + i, even with the rest of the model held fixed (§3, minimal circuit).
3. **The token's code turns about 180° between write and read-out.** Two groups of dense features do this: "keepers"
   hold the token in roughly its input orientation, and "flippers" read the keepers with the opposite sign. The stored
   code ends up opposite to the input code, and that is exactly why feeding the token back cancels it.
4. **The gates only do timing.** Update-gate features clock how fast memory turns over. Reset-gate features clock how
   much memory the candidate reads, and matter mainly for longer sequences. Both are computed from candidate features
   (mainly the `<DEL>` feature 123 and the positional part of the dense relay features). Neither carries token content.
5. **Most of the circuit runs on dense candidate features** (active in almost every sequence). Their information is in
   their magnitude, so they needed magnitude explanations. Many of them merge a timing signal with a token set, which
   is a transcoder-training issue.

---

## 0. Setup and code

Models: `data/models/copy_train/copy_128_high/copy_128_high.ckpt`; transcoders in `copy_transcoders/`
(`reset_512_sparse_e100.ckpt`, `candidate_1024_sparse_dpi_final.ckpt` = **update**, `candidate_512_updatelike_e100.ckpt`
= **candidate**). Data: `features/copy_transcoder_v2_1of5/1M_128_high_seq{3,4,7}.pt` (the 1/5 subsample). Unless
stated, results use sequences with distinct tokens and the real GRU. Outputs are in `features/copy_circuits/`.

| file | what it does | output |
|---|---|---|
| `circuit/edge_att.py` | attribution graphs (frozen gates, hidden state folded into edges), gate-feature decompositions; `gate_scores` split out of `gate_feature_ranking` | used by the scripts below |
| `circuit/copy_task/copy_circuits.py` | per-output attribution graphs, swap tests, contrastive graphs (`--sections graphs swaps contrast`) | `circuits.json` |
| `circuit/copy_task/slot_absorption.py` | is "slot filled" its own feature or absorbed into token features | `absorption_L{3,4}.json` |
| `circuit/copy_task/gate_mechanism.py` | the token's trace in h: kept / written / gate split, gate features vs gate units | `gates.json` |
| `circuit/copy_task/feature_circuits.py` | slot-token tables, exact contrastive flow (DeepLIFT rescale), carrier ablations, steering by token tables | `feature_circuits_L4.json` |
| `circuit/copy_task/token_circuits.py` | the circuit for one real sequence and slot, with labels | `token_circuits_L4_g.json` |
| `circuit/copy_task/magnitude_scores.py` | additive token-table magnitude explanations for dense features, held-out R² | `magnitude_scores_{hidden,update,reset}.json` |
| `circuit/copy_task/autointerp_magnitude.py` + workflow | agent-written magnitude explanations (one agent per feature + skeptic) | `features/copy_gru_v1_candiff_magnitude/` |
| `circuit/copy_task/circuit_tests.py` | interventions: `<DEL>`, pop, freezing, carry, gate time-shifts, pop matching, resample, reset combinations | `circuit_tests_L4.json`, `reset_combos_L{4,7}.json` |
| `circuit/copy_task/rotation_edges.py` | rotation by edges, feature angles (keepers / flippers), sign test, reversal decomposition, reversal interventions | `rotation_edges_L4.json`, `rotation_phases_L4.json`, `reversal_tests_L4.json` |
| `circuit/copy_task/gate_features.py` | what drives the gate features, which reads / writes they open | `gate_features_L4.json` |
| `circuit/copy_task/circuit_sufficiency.py` | minimal circuit: sufficiency and necessity patching (free and clamped) against circuit size | `circuit_sufficiency_L4.json` |
| `circuit/copy_task/circuit_extras.py` | trace cosine, pop split, early feedback, dense-carrier drops, token-conditional ablation | `circuit_extras.json` |
| `circuit/copy_task/autointerp_plots.py` | figures: subcommands `circuits`, `feature_circuit`, `token_circuit`, `rotation_wheel`, `circuit_summary` | `features/copy_circuits/plots/` |

Typical call (all scripts take the same model arguments):

```
PYTHONPATH=. python -m circuit.copy_task.circuit_tests --rnn data/models/copy_train/copy_128_high/copy_128_high.ckpt \
    --traces features/copy_transcoder_v2_1of5/1M_128_high_seq4.pt \
    --transcoders reset=copy_transcoders/reset_512_sparse_e100.ckpt \
        update=copy_transcoders/candidate_1024_sparse_dpi_final.ckpt \
        candidate=copy_transcoders/candidate_512_updatelike_e100.ckpt \
    --out features/copy_circuits/circuit_tests_L4.json
PYTHONPATH=. python -m circuit.copy_task.autointerp_plots circuit_summary features/copy_circuits
```

---

## 1. What the model does (behaviour, real GRU)

**Each output depends only on its own token.** Swap the token at position i (and the fed-back copy) and only output i
changes (logit of the original token −24 to −27); all other outputs change by at most 0.12.

**Memory and feedback have to agree.** Swap token i but leave the fed-back token unchanged: the next output is the
*stored* token at i in 82–93% of cases (the correct next token only 9–15%). The fed-back token pops whatever it
matches; if it matches nothing, the oldest token stays and is output again.

![swap test](figures/circuits/swap_L4.png)
![swap test, normalised](figures/circuits/swap_norm_L4.png)

**Skipping a step delays, never skips.** Freeze the hidden state for one copy step (update gate z = 0). Running
free (the model's own outputs fed back), every later output is the previous token: the output is delayed by one.
Teacher-forced, the fed-back tokens no longer match the head, so nothing is popped and later outputs fail.

![freeze test](figures/circuits/queue_freeze_L4.png)

*Kind 1 (real GRU, no features). Code:* `copy_circuits.py` (`swaps`), `circuit_tests.py` (`timing`).

---

## 2. How the circuit was traced (method)

**Attribution graphs.** Gates are frozen at their real values (as the paper freezes attention), so the GRU is linear
in what gets written into h. Nodes are candidate features at each step, inputs, error nodes and logits; edges are
exact linear effects through the carry `(1 - z)` and the reset gate. Every decomposition sums exactly to the real
logit (residual ≤ 3e-4). Error nodes carry only 0.8–1.7% of the attribution (the analogue of the paper's replacement
score).

**Problem: AND-like features give "partner credit".** A single graph credits output 3 with +57 from token 0, but
swapping token 0 changes output 3 by 0.0. The read-out features fire only for "this token AND it is time for this
slot", and a linear graph splits credit over both inputs. See `identity_attribution_L4.png` (raw credit) next to the
swap test.

![single-graph credit](figures/circuits/identity_attribution_L4.png)

**Fix 1: contrastive graphs.** Subtract the graph of a run where token i is swapped. Off-diagonal totals drop to
≤ 0.33 (≤ ~1.8 at any one step). But on the diagonal, unchanged inputs still get credit when a read-out feature
switches off.

![contrastive graphs](figures/circuits/contrast_share_L4.png)

**Fix 2: exact contrastive flow (DeepLIFT rescale).** Feature *pre*activations are linear in what feeds them. Between
the two runs, each feature's preactivation change splits exactly into upstream activation changes, gate changes and
input changes. Each JumpReLU is replaced by its slope between the runs, Δa / Δp. Unchanged features and inputs then
carry exactly zero flow, and flows still sum exactly to the logit change (residual ≤ 5e-5). This is what the circuit
diagrams below use. *Code:* `feature_circuits.py` (`change_flow`).

**Slot-token tables.** For every candidate feature, step and slot, the mean activation given the token in that slot.
They give each carrier a magnitude code (which tokens raise / lower it) and let us *steer* features by magnitude.

### Three kinds of intervention

Results in this doc come from three kinds of experiment. Each result below says which, or the code does.

| kind | what changes | used for |
|---|---|---|
| **1. Real GRU, no features** | inputs, or whole gate vectors (all 128 coordinates) | swap tests, feedback mismatch, early feedback, freezing a step (z = 0), the carry test's z = 1, the pop split and match test, the trace / rotation measurements in h, gate-*unit* ablations |
| **2. Real GRU, edited along feature directions** | the GRU computes its real gate or candidate value, then only the chosen features' share is changed: value − (a_f − replacement)·d_f. Everything else stays real, including the transcoder error. | `<DEL>` 123 and reset 31, pop-feature ablations, gate time-shifts, reset combinations, resample vs mean, reversal tests, own-carrier ablations, steering by token tables, gate feature vs unit ablations |
| **3. Spliced** | the transcoder *replaces* the candidate (some features mean-ablated); its error is not kept | dense-carrier drops, token-conditional ablations (`circuit_extras.py` via `explained_fidelity.py`), the earlier fidelity tests |

Why it matters: kind 2 isolates the feature (baseline = the exact model, 100%). Kind 3 adds the transcoder's own
error, which compounds over the recurrence (baseline 98.5–98.9%), so its numbers are a little harsher. Kind 1 says
what the model does, not which features do it. Most feature claims here rest on kind 2.

### How much the attribution graphs helped

The graph *machinery* was essential; the paper-style use (read off one pruned graph for one prompt) helped little and
at first misled us.

- **Direct help.** The graphs gave the first clues: the logit is written by read-out features at the output step; the
  fed-back token pushes its own logit down by 30–41 (first sign of the pop); the `<DEL>` feature 123 and the gate
  features (717, 650, 434, 31) first appeared in the graph's feature effects and gate rankings. The ~1% error share
  told us the features account for nearly all of the computation.
- **Help as infrastructure.** Freezing the gates makes the GRU exactly linear in what it writes. That exactness is what
  the contrastive flow (DeepLIFT rescale), the gate-feature splits (`gate_scores`, `gate_features.py`), and the
  rotation / reversal decompositions are built on.
- **Where they misled or fell short.** Single graphs gave partner credit (+57 from a token that has no effect;
  feature effects of ±60–110 cancelling each other); this needed swap tests, contrast and the rescale to fix. Pruned
  graphs were never used; the diagrams come from flows averaged over 300 sequences. The main mechanisms came from
  elsewhere: the queue and pop from behavioural tests (kind 1), the rotation and rewrite from the trace in h,
  keepers / flippers from the angle analysis, the cancellation from splitting the GRU step, and the gates' roles from
  interventions (kind 2).

In short: the graphs gave an exact account of *where* effects flow once used contrastively; *how* copying works came
mostly from counterfactuals and interventions, with the graph decompositions attributing those findings to features.

---

## 3. The circuit at a high level

For slot i the token's effect flows: **input at in_i → write features → relay features (each step) → read-out
features at L + i → logit**, then **pop** one step later. Mean over 300 sequences, L = 4:

| | flow |
|---|---|
| enters at the input of in_i | +21 to +22 of a total change of +25 to +28 |
| through each relay step's features | +12 to +17 (straight to the next step, plus skips through the update-gate carry) |
| read-out features → logit | +23 to +26 |
| gate changes, bias / error writes | 1–2 in total |

Features carrying 80% of the flow per sequence: 4–5 at the write step, 6–10 per relay step, about 6 at read-out.

![feature circuit, slot 2](figures/circuits/feature_circuit_L4_slot2.png)

**One real sequence** (`p t g 2 <DEL> p t g`, slot 2 = `g`), with labels:

| step | what happens | features (how much `g` moves each, against `b`) |
|---|---|---|
| in2 (input g) | write | 120 "input g in the first five positions" (sparse) +2.00; 79 "{g,0,2,y,i} up" +0.82; 274 "{l,x,g,i} down" −1.32 |
| in3, DEL | relay | 79 +1.6; 349 "{g,r,j,o} up over the last ~6 inputs" +1.6; 274 −1.4; 8 +0.7 |
| out1 (g is next) | relay → read-out | 8 "token value read from the queue head (g > q > e > u)" +0.85; 212 "{0 ≫ g, r, i, 2, p}" +0.56 |
| out2 (output g) | read-out | 8 +1.82; 212 +1.46; 214 "o (also g, 3, u)" +1.15; 77 "c, d, f, 0 high; a, e, w low" −0.62 |

Every label's sign agrees with how `g` actually moves the feature.

![token circuit, g in slot 2](figures/circuits/token_circuit_L4_slot2_g.png)

**Causal checks** (L = 4; kind 2, real GRU with feature edits):

| test | result |
|---|---|
| remove each sequence's own top-5 carriers at the **write** step | output i accuracy 19–41% (random 5: 92–100%); earlier outputs unaffected |
| same at the **read-out** step | 64–72% |
| same at a single **relay** step | 87–100% (the token is spread over many features, and the carry keeps a second copy) |
| **steer** the top-20 carriers' magnitudes by the token tables, as if the slot held another token | output switches to that token: 59–97% one step after the write, 100% at read-out |
| same with each feature's token rows shuffled (same size of push) | 0–3% |

*Code:* `feature_circuits.py`, `token_circuits.py`; plots via `autointerp_plots.py feature_circuit / token_circuit`.

### Is the circuit enough, and is it needed? (minimal circuit, activation patching)

For each sequence and slot i, take the original run (token v) and the run with v swapped for another token b
(feedback matched). The circuit is that sequence's top-k candidate features per step, at steps i … L+i, ranked by
|contrastive flow| (so features driving either token count). Patching sets a feature to its value in the other run, in
the real GRU (gates and transcoder error left as they are). Four modes:

| mode | run | circuit features | every other feature | question |
|---|---|---|---|---|
| sufficiency, free | swapped | set to original values | recomputed | does output i become v again? |
| sufficiency, clamped | original | held at original values | held at swapped values | does output i stay v? |
| necessity, free | original | set to swapped values | recomputed | does output i stop being v? |
| necessity, clamped | original | set to swapped values | held at original values | does output i stop being v? |

"Free" lets the rest of the model react to the patched circuit; "clamped" holds the rest fixed, so only the circuit's
own writes can change the result. Patching happens at every step from i to L + i, each step with its own top-k
features; the values patched in come from a clean run of the other sequence (e.g. in necessity, the circuit takes
the values it had in the clean swapped run). The update and reset gates and the transcoder error are never patched:
they are computed live from the patched state.

| features per step | share of \|flow\| | sufficiency free: output = v | sufficiency clamped: output = v | necessity free: output ≠ v | necessity clamped: output ≠ v |
|---|---|---|---|---|---|
| 1 | 29% | 39–51% | 1–2% | 86–93% | 16–37% |
| 3 | 55% | 91–97% | 31–41% | 100% | 76–82% |
| 5 | 68% | 97–100% | 75–84% | 100% | 96–97% |
| **10** | **84%** | **100%** | **97–100%** | **100%** | **100%** |
| 20 | 95% | 100% | 100% | 100% | 100% |
| random features outside the circuit (any k) | | 0% | 0% | 0–2% | 0–2% |

(Ranges over slots 0–3, 300 sequences each; about 38–48 features per step carry any flow, out of 512.)

- **About 10 features per step are both sufficient and necessary**, even with the rest of the model held fixed:
  clamped, they keep output i correct 97–100% of the time, and corrupting them changes it 100% of the time (to the
  swapped token b in 93–97%). Random features outside the circuit do nothing in any mode.
- **When the rest can react, far fewer are needed**: 3 per step written in restore the token 91–97% of the time, and
  corrupting just the top 1 per step changes the output 86–93% of the time. The other features follow the circuit.
- Corrupting slot i's circuit also breaks later outputs (other outputs 14–57% for slots 0–2): the stored token no
  longer matches the fed-back one, so it is not popped (§1).
- So the circuit is small relative to the dictionary (about 10 of 512 per step) but not a handful; the token is spread
  over a few dozen carriers per step, matching §5.

An earlier version of this test ("complement") let the circuit features themselves be recomputed while the rest was
held at swapped values; the corrupted rest then dragged the circuit along, so it needed about 20 features per step.
The clamped modes above hold the circuit at its values too.

![minimal circuit](figures/circuits/minimal_circuit_L4.png)

*Code:* `circuit_sufficiency.py`; output `circuit_sufficiency_L4.json`.

---

## 4. The dense carriers

**Most carriers are dense.** In the `g` example, 18 of 20 carriers are active in 97–100% of sequences; only 120
("input g") and 411 are sparse (10–15% of sequences). Firing labels ("copy phase", F1 0.96) are right about *when* they
fire but say nothing about what they carry.

**Magnitude explanations.** For the 70 candidate features that fire on ≥ 20% of steps (65 of them active in ≥ 90% of
sequences), activation ≈ timing (mean per L, t) + additive token tables. Held-out R² beyond timing (median):

| explanation | R² beyond timing |
|---|---|
| last 2 inputs | 0.27 |
| last 8 inputs | 0.57 |
| outputs: current … 3 ahead (the queue in memory) | 0.36 |
| all cues (full tables) | 0.79 |
| compact token sets (one raising set, one lowering set, a weight per cue) | 0.53 |
| **agent-written explanations** (≤ 8 terms, skeptic-checked) | **0.65** — 45 explained, 24 partly, 1 not |

The dense gate features are purely positional (timing alone R² ≈ 1.0). *Code:* `magnitude_scores.py`,
`autointerp_magnitude.py`; records in `features/copy_gru_v1_candiff_magnitude/`.

**They are needed.** Kind 3: candidate transcoder spliced into the GRU, features mean-ablated, lengths 3–9:

| removed | accuracy |
|---|---|
| nothing | 98.9% |
| the "5" dense features from the earlier report (162, 214, 284, 388, 487) | 57.7% |
| **the 15 other dense circuit carriers** (8, 32, 54, 77, 79, 91, 212, 231, 266, 274, 349, 368, 387, 440, 500) | **15.6%** |
| 15 random other dense features | 26–34% |
| 15 random sparse features | 86–88% |
| one carrier alone | 77% (91) to 96% (368) |

The earlier "5" were not the only important dense features; they were the ones whose *firing* could not be predicted.

![dense carriers needed](figures/circuits/dense_carriers_needed.png)

**Each carrier serves its own tokens** (kind 3). Removing one hurts mostly outputs of the tokens in its raising set:

| feature | its raising set | outputs in that set | other outputs |
|---|---|---|---|
| 77 | d, c, f, 0 | 69% | 95% |
| 387 | u, d, w, 2 | 71% | 95% |
| 368 | y, s, 1 | 81% | 96% |
| 212 | 0, g, r, i | 83% | 94% |
| 231 | j, 1, p, m | 83% | 95% |
| 349 | g, i, r, o | 91% | 94% |

**Merged variables (a transcoder issue).** These features fire for every token (the timing / phase part) and run 2–7×
higher for their own token set (the part the model uses). The transcoder merged a phase feature and a token-set
feature into one. The same happens at slots 0–1: no feature carries "a token arrived" there; it is spread over the
token features (37–49% of the shared write; `slot_absorption.py`). At slots 2–3 and `<DEL>`, dedicated positional
features carry it (43–95%). The fix is better transcoder training, not relabelling.

*Code:* `circuit_extras.py` (`dense_drop`, `token_cond`), `slot_absorption.py`.

---

## 5. How a token is stored: rewrite, rotation, keepers and flippers

**Leaky rewrite.** The token's trace in h (original minus swapped run) splits exactly into kept, written and gate
parts. Only 30–60% survives each input step; the candidate rewrites the rest. 96–99% of the candidate's change comes
from reading the stored trace through the reset gate. Gate changes add 0–5%. (`gate_mechanism.py`)

**The code turns about 180°.** Cosine with the written code falls to −0.70 … −0.85 at the output step. Visibility to the
output layer goes from −30 … −46 when written to +50 … +55 at the output step, then drops after the pop.

![trace turning](figures/circuits/trace_turning_L4.png)

**The candidate writes a turned copy each step.** Projecting each step's pieces on the stored code (e1) and the
direction towards the read-out (e2): the carried part stays on e1; the written part has e1 > 0 and a large e2 (turned
35–85°). Nothing actively undoes the old code; it fades through the update gate. Half of the turning needs 117–239
feature→feature edges, so the turn is spread over the whole dense population. (`rotation_edges.py`)

**Two groups of features.** Each carrier writes the token at a characteristic angle (0° = as written, 180° = read-out):

- **keepers** (487, 283, 145, 274, 488, 474): write at +13° … +36° at every step;
- **flippers** (426, 8, 348, 387, 440, 328, 212, 214, 368, 388, 336, 315, 500, …, and the copy-phase read-out 77, 34,
  163, 177): write at +150° … +163°.

Most flippers already respond at the write step (about −20°) and reverse 2 steps later. The trace turns smoothly
because it is a blend that shifts from keepers to flippers.

![angle wheel](figures/circuits/rotation_wheel_L4.png)

**Flippers store the token as the opposite of how they read it as input.** Correlation between a flipper's input code
(encoder weights for the token) and its memory code (how the stored token moves it), by steps since the write:

| | +1 | +2 | +3 | +4 (output) |
|---|---|---|---|---|
| flippers (median) | +0.09 | −0.63 | −0.77 | −0.86 (100% negative) |
| keepers (median) | +0.03 | +0.03 | +0.02 | −0.14 |

**What causes the reversal.** The flippers' correlation splits exactly by what in h they read: keepers' write −0.40 /
−0.49 (at +1 / +2), flippers' own input-driven write +0.47 / −0.02, gates ≈ 0. So flippers read the keepers with the
opposite sign; at the write step their own input-driven write hides this; one step later the input has moved on and
the reversal shows. Causal tests (kind 2; `rotation_edges.py --reversal_tests`):

| intervention | effect on the flippers' correlation | outputs |
|---|---|---|
| B: remove the flippers at the write step (no input drive) | reverses one step early: +1 goes +0.27 → −0.06, +0.41 → 0.00, +0.02 → −0.46 | unchanged |
| A: remove 10 keepers at +1 | reversal weaker at +2: −0.84 → −0.62, −0.83 → −0.55, −0.72 → −0.36 (slot 3: −0.13 → +0.18) | 90–99% |
| random controls | no change | unchanged |

![reversal tests](figures/circuits/reversal_tests_L4.png)

**The carry keeps a second copy** (kinds 1 and 2). Steering at a single later relay step often fails (33–78%). Dropping the carried
copy at that step (z = 1) lets steering work (63–93%), and the candidate's rewrite alone keeps that output at 78–100%.

![carry](figures/circuits/carry_second_copy_L4.png)

**The token is stored per slot.** h before `<DEL>` is predicted with R² 0.99 from slot-specific token one-hots, against
0.39 from a bag of tokens. (`slot_absorption.py`)

---

## 6. The pop: cancellation in the candidate

At a pop step the update gate (≈ 0.70) replaces most of memory either way; the head survives only if the candidate
writes it again.

| pop step | kept from memory | candidate write, memory only | write, matched token fed back | write, other token fed back |
|---|---|---|---|---|
| out1 | +2.4 | +29.5 | −0.5 | +30.5 |
| out2 | +1.7 | +30.5 | −1.0 | +32.8 |
| out3 | +2.2 | +31.2 | +0.2 | +33.2 |

The matched-vs-mismatched difference (−32.5) is −32.4 from what the candidate writes (−28.8 through the input
weights, −3.7 through the reset gate) and −0.1 from the update gate. The fed-back token's input term and the stored
token's memory term are added inside the candidate's tanh and cancel. This is the flippers' reversed code at work.

![pop cancellation](figures/circuits/pop_cancellation_L4.png)

**Cancellation follows the token, not the slot.** Feeding back the token due now or next turns *its* write negative
(−14, −26) while the head stays and is output again.

![early feedback](figures/circuits/early_feedback_L4.png)

**No separate pop features** (kind 2). The features with the largest pop flow (177, 77, 163, 34, 347, …) are the read-out
features. Removing 10 of them breaks outputs (31–48% with resampling) but causes no repeats (≤ 2%). They make up about
a third of the cancellation; the rest is spread out.

*Kind 1 for the pop split, match test and early feedback; kind 2 for the pop-feature ablations. Code:* `circuit_extras.py` (`pop_split`, `early_feedback`), `circuit_tests.py` (`match`, `pop`).

---

## 7. `<DEL>`

Candidate feature **123** ("`<DEL>` delimiter step") fires at `<DEL>` with nearly the same value in every sequence.

- Zeroing it at `<DEL>` (kind 2): output 0 still 100%, outputs 1–3 58% / 0% / 0%. Together with reset 31 zeroed: 14% / 0% / 0%.
- It drives the copy-phase gate features (§8): update 177 (+13.1 of its preactivation), reset 434, 322, 477, update 901,
  through its write decaying in memory ("time since `<DEL>`").
- Mean ablation shows nothing (its mean at `<DEL>` is its value), so the right removal is zero.

---

## 8. Update-gate vs reset-gate features

The two gates enter the GRU step differently: **update** `h = (1 - z) h + z n` (how much memory changes), **reset**
`n = tanh(W_in x + W_hn (r ⊙ h) + b)` (what the candidate reads).

| | update-gate features | reset-gate features |
|---|---|---|
| what they control | how much of memory survives and how much of the new write gets in (one knob for both) | what the candidate sees of memory, i.e. the *content* of the rewrite |
| main features | 650 (steps 1–3), 580 (first step), 717 (t ≥ 3 → `<DEL>`, off in copy), 297 (`<DEL>`), 177 (first copy step), 901 (end of copy) | 31 (`<DEL>`), 434 (copy phase), 312 (input position ≥ 3), 76 (early), 147 (position 2), 322 / 477 (later copy steps) |
| share of their gate's action | 10–35% of erasing; the bias does 57–90%, including the pop's erase | the bias carries 60–110% of the reading; features add phase-specific opening (76 +0.22 at the flippers' reversal step; 434 +0.21–0.27 at the pop; 31 at `<DEL>`) |
| what drives them | built from candidate features: 717 and 312 from the relay features' positional part (336, 440, 315, 500, 91); 177, 901 from `<DEL>` 123; 650 from any token feature one step back | 434, 322, 477 from `<DEL>` 123 decaying in memory; 312 from the relay features; 31 from the `<DEL>` input |
| mistimed (L = 4; kind 2) | timing errors: 717 on in the copy phase → outputs 2–3 at 0%; 650 off early → 69–76% | corrupted contents: 434 on during input → output 3 at 17%, no repeats |
| removed | (all-steps mean of top features breaks late outputs) | L = 4: little (all 8 off → 83%); L = 7: 31 alone → 60%, 312 → 71%, 434 → 72%, all 8 → 26% |
| token content / matching | none (≤ 2 logits) | none (≤ 6%; −3.7 of −32 at the pop) |

So: **update features clock how fast memory turns over** (get them wrong → delays, repeats, leaks); **reset features
clock what the rewrite is computed from** (get them wrong → corrupted contents), and are needed mainly under heavier
memory load.

![gate time-shift](figures/circuits/gate_timeshift_L4.png)
![reset features by length](figures/circuits/reset_features_by_length.png)

*Drivers / reads / writes are exact decompositions; time-shifts and removals are kind 2. Code:* `gate_features.py` (drivers, reads, writes), `circuit_tests.py` (`timeshift`, `reset_combos`),
`gate_mechanism.py` (kept / erased shares).

**Features vs GRU units for the gates.** 1–2 gate features cover 80% of each gate term (beyond the bias), against
49–56 units. Removing the top features does targeted damage where removing the same number of units does nothing;
at an equal size of push the damage is similar, and against random directions of equal size the feature directions are
more damaging in 2 of 4 gate × length cases. (`gate_mechanism.py`)

---

## 9. Mistakes and pitfalls we hit

- **Partner credit** in single graphs (off-diagonal +57 from token 0): fixed with contrast, then the DeepLIFT-rescale
  flow (§2).
- **Reset-gate edits were silently ignored** in an early version of `circuit_tests.run` (the candidate was computed
  before the hook changed r). Fixed; all reset results here are from the fixed version.
- **Mean or same-step resample ablation cannot remove a timing feature** (it already has that value at that step). Use
  zero or a time-shift instead.
- **"Dense"** means active across many sequences; firing on many steps within a sequence is expected.

## 10. Open questions

- How exactly each read-out / flipper feature combines the stored and the fed-back token (opposite signs per feature
  is the hypothesis; shown in aggregate, not per feature).
- Whether the pop permanently erases a token fed back early (teacher forcing stalls the queue, so this was not testable).
- Update-gate time-shifts at L = 7 (reset features mattered much more there).
- Lengths beyond 4 for the full feature circuit; L = 3 was checked for graphs and swaps only.
- A better-trained candidate transcoder that separates timing from token sets.
