# RL agent: examples of well-explained transcoder features and GRU units

Five transcoder features and five GRU units per gate (reset, update, candidate), chosen as the best explained by
the fairest test we have, with their explanations exactly as written.

## How these were chosen and scored

- **Pool.** The behaviour-matched sets (`circuit/rl_task/matched_sets.py`, `rl_features/rl_fidelity/matched_sets.json`):
  per gate, the fewest features and the fewest units that, with everything else of that gate at its phase mean,
  recover 90% of the way from nothing kept to the full transcoder on switch agreement, GO agreement and value error.
  Reset: 55 features / 33 units; update: 113 / 12; candidate: 102 / 30.
- **Explanation.** Words only, at most 2 sentences, written by a blind explainer agent that saw only the
  component's activation (rescaled 0-10 over its own range; raw values, no thresholds) on 30 example windows of the
  dev split (`circuit/rl_task/simulation_scoring.py explain_sample`). No predicates, fitted terms or earlier labels.
- **Score.** A separate simulator agent saw only the explanation and the same 100 random held-out windows used for
  every component (1,500 steps, no values) and predicted every step (`simulate_shared.js`). Columns:
  - *Correlation*: simulated vs true activation over all 1,500 steps;
  - *Beyond timing*: the same after removing each (phase, episode decile) mean from both; whether the words predict
    more than "high at GO" or "higher late in the episode". **Examples are ranked by this.**
  - *Variance explained*: 1 - SSE / SST of the simulated values;
  - *Timing share*: how much of the true activation phase x episode decile alone accounts for;
  - *Active steps*: share of steps above 0.5 on the 0-10 scale;
  - *Matters*: whether the component is in the set that matters individually (conditional or averaged ablation
    effect on choice or value, `rl_features/rl_fidelity/matter_sets.json`); *worst-situation choice loss*: the GO
    choices changed by mean-ablating it alone, after the previous-trial pattern where that is largest
    (`circuit/rl_task/conditional_tests.py`). Choice × state × reward: e.g. `a1 S2 r` = chose a1, reached S2
    (rare), rewarded.

## What the examples show

- **The best-explained features are sparse event detectors.** They are active on 5-16% of steps, mostly a single
  step of a single trial type (e.g. D2 after a rare transition from a particular choice, or D1 after a reward in a
  particular state), with a magnitude that grows or shrinks with recent history or time in the episode. Timing
  accounts for only 13-37% of them. Several of the best-explained features are also among the ones the model
  needs most (candidate 219, 210, 147, 167, 10; reset 59).
- **The best-explained units are dense.** They are active on 84-99% of steps and their explanations are a
  description per phase ("always high at GO; at D2 shows the state reached; at D1 low after a reward"). For many,
  timing alone accounts for 58-87% of the activation; the beyond-timing score is what their words add. Several
  track the agent's current preference over many trials (reset 15, candidate 19, 5).
- **Being well explained and mattering are different things.** The best-explained candidate features also matter
  a lot (19-43% of GO choices lost after their critical trial type for 219, 210, 147, 167), but none of the five
  best-explained reset units and few of the best-explained update features matter individually.
- **Caution on very rare components.** Update features 119 and 188 are active on almost none of the 1,500 shared
  steps (0% and 1%), so their high scores rest on a handful of events.
- **Both reach very high scores here**: these are the top five of each, not typical cases. Across the whole sets,
  on these shared windows, units score as well as or better than features (median correlation beyond timing:
  reset 0.79 vs 0.77, update 0.80 vs 0.59, candidate 0.80 vs 0.78), because sparse features lose correlation on
  every missed or false spike in random windows.

## Reset gate

### Transcoder features

| # | Correlation (shared) | Beyond timing | Variance explained | Timing share | Active steps | Matters | Worst-situation choice loss |
|---|---|---|---|---|---|---|---|
| 41 | 0.99 | 0.99 | 0.98 | 0.13 | 5% | yes | 2.3% after a2 S1 r |
| 59 | 0.99 | 0.98 | 0.97 | 0.25 | 10% | yes | 17.9% after a1 S1 u |
| 37 | 0.97 | 0.97 | 0.94 | 0.13 | 6% | no | 0.9% after a1 S2 r |
| 106 | 0.97 | 0.97 | 0.95 | 0.25 | 12% | no | 0.4% after a1 S2 r |
| 10 | 0.97 | 0.97 | 0.94 | 0.37 | 16% | no | 0.8% after a2 S1 r |

- **41**: The value is 0 everywhere except at D2 when the agent chose a2 and a rare transition took it to S1. There it is about 7-8 early in the episode (before trial 50), about 9 between trials 50 and 85, and 10 in the last trials; a rare transition after a1 (landing in S2) stays at 0, and rewards and earlier trials do not matter.
- **59**: At D1 it is high (9-10) only when the previous trial was rewarded and that trial D2 state was S2, whichever action was chosen and whether the transition was common or rare. It is 0 at every other step, including D1 after an unrewarded trial or after a reward in S1, and at every GO and D2 step.
- **37**: This component fires only at D2, and only when the agent chose a1 but landed in S2 by a rare transition: then it is about 9-10, highest (10) when the trial before was an a2 choice that rarely reached S1, and about 8 on a repeat of the same a1-to-S2 rare event. Every other step is 0: all D1 and GO steps, every a2 choice (including a2 rarely reaching S1), and common a1-to-S1 trials, which reach at most 1 right after an a1-to-S2 rare trial.
- **106**: Nonzero only at GO steps where the agent chooses a1; it is always 0 at D1 and D2 and whenever a2 is chosen. If the previous trial was a rewarded a1 to S1 common trial, it is 8-10 (6-8 right after an unrewarded trial, 10 deep in a long rewarded a1 streak). If the previous trial was unrewarded after a common transition, it falls to about 2-3, and it is about 0 when the previous trial was a rare transition, or when there were two misses in a row.
- **10**: It is 0 at D1 and GO, and 0 at D2 whenever the agent chose a2 (whichever state it reached); it only turns on at D2 after an a1 choice, about 9-10 when a1 led to S1 (common) and the previous trial or two were rewarded, about 7-8 when the previous trial was unrewarded, and about 3-6 when a1 led to S2 (rare), lowest after an unrewarded trial. It runs weaker early (trials 1-5, about 6-8) and late in the episode (trials 75 and later, about 4-7 for common, 0-2 for rare), and is strongest in the middle of the episode, around trials 30-60.

### GRU units (raw r)

| # | Correlation (shared) | Beyond timing | Variance explained | Timing share | Active steps | Matters | Worst-situation choice loss |
|---|---|---|---|---|---|---|---|
| 8 | 0.98 | 0.93 | 0.96 | 0.83 | 84% | no | 1.1% after a1 S1 u |
| 44 | 0.98 | 0.93 | 0.96 | 0.78 | 96% | no | 1.5% after a1 S1 u |
| 15 | 0.93 | 0.92 | 0.85 | 0.12 | 97% | no | 2.2% after a2 S2 u |
| 18 | 0.96 | 0.92 | 0.92 | 0.72 | 93% | no | 0.8% after a2 S2 u |
| 31 | 0.92 | 0.89 | 0.83 | 0.58 | 99% | no | 1.5% after a2 S2 u |

- **8**: At GO it is always high, 9-10 if the previous trial went unrewarded and 7-9 if it was rewarded (lowest, about 7, after several rewards in a row). At D2 it shows the state reached, about 7 for S1 and 3 for S2 whatever the choice, and at D1 it is low: 0-1 after a reward, 2-4 after a loss, rising to 5-6 after several losses in a row.
- **44**: At D1 and GO it is always high, about 9-10 (GO is 10 after choosing a1 and about 9 after a2; D1 slips to 8 late in the episode, after about trial 80), whatever the reward. At D2 it depends on the state reached and the choice made: about 8 for a1 reaching S2 (rare), about 5 for a2 reaching S2 (common), about 3 for a1 reaching S1 (common, dropping to 1-2 after about trial 80), and 0-1 for a2 reaching S1 (rare).
- **15**: This unit tracks the agent current preference: it is high while the agent keeps choosing a1 (GO about 8-9; D2 about 9-10 on reaching S1, about 7 on reaching S2) and low while it keeps choosing a2 (GO about 3-4; D2 about 1-2 on reaching S2, about 4 on reaching S1). At D1 it is 9-10 after an unrewarded trial and about 5 after a rewarded one during a1 runs, and about 3 after an unrewarded trial and 0-1 after a rewarded one during a2 runs, with values in between (5-7) for the first trials after a switch.
- **18**: At D2 the value depends only on the state reached: about 0-1 when S2 is shown and about 4-6 when S1 is shown. At D1 and GO it is high and rises after a reward and when the agent has lately been choosing a2: D1 is 10 after a reward (8-9 after no reward) when the agent has been choosing a2, and 7-8 after a reward (5-6 after no reward) when it has been choosing a1; GO is 1-2 lower than D1 (8 for a2 after a reward, 6-7 for a1 after a reward, 4-6 after no reward, as low as 3-4 after several losses in a row).
- **31**: At GO it is nearly maxed (10 early in the episode, 9 by mid-episode, 6-7 after about trial 85); at D1 it is 8-10 after an unrewarded trial, but after a rewarded trial it is about 6 while the agent has been choosing a1, about 2 while it has been choosing a2, and 0-1 late in the episode. At D2 it is about 7-8 after a common transition (dropping to 4-5 late in the episode), 9 after a rare a1-to-S2 transition, and low (5 mid-episode, 2-3 late) after a rare a2-to-S1 transition.

## Update gate

### Transcoder features

| # | Correlation (shared) | Beyond timing | Variance explained | Timing share | Active steps | Matters | Worst-situation choice loss |
|---|---|---|---|---|---|---|---|
| 31 | 0.97 | 0.97 | 0.94 | 0.32 | 13% | no | 1.0% after a2 S2 u |
| 201 | 0.96 | 0.96 | 0.92 | 0.14 | 5% | no | 0.3% after a2 S1 r |
| 119 | 0.95 | 0.95 | 0.79 | 0.06 | 0% | no | 0.0% after a2 S2 u |
| 188 | 0.94 | 0.93 | 0.82 | 0.12 | 1% | no | 0.1% after a2 S2 u |
| 108 | 0.97 | 0.93 | 0.94 | 0.73 | 22% | yes | 0.9% after a2 S1 r |

- **31**: The value is 0 at every D1 and GO step and is nonzero only at D2 when the agent chose a2 and reached S2 by the common transition, where it is usually 9-10 if the agent has kept choosing a2 for several trials. It is 0 at D2 for every a1 choice and every rare transition, it is often 0 or low (0-6) when a2 is the first choice right after a run of a1, and it drops to about 1-5 late in the episode (after about trial 85).
- **201**: It is 0 everywhere except at D2 on a rare transition after choosing a2 (reaching S1), where it is usually 9-10, regardless of reward, previous trials or how many rare transitions came before. Rare transitions after choosing a1 (reaching S2), all common transitions, and every D1 and GO step stay at 0; very late in the episode (around trial 85 and later) the a2-to-S1 spike can be weak (0-2).
- **119**: It fires only at the GO step, only very late in the episode (about trial 90 onward), when the agent chooses a2 right after a trial where a2 led to S1 by a rare transition and got no reward: about 6-8 the first time, rising to 9-10 if the trial before that was also an unrewarded rare trial (about 3 if it is before trial 93). Every other case is 0: all D1 and D2 steps, any a1 choice, any trial before about 90, and a GO that follows a rewarded trial or an unrewarded common trial (at most 1).
- **188**: Zero everywhere except at D2 when the agent chose a2 and landed rarely in S1 late in the episode: about 9-10 from trial ~89 on, about 7 near trial 87, about 3 near trial 82, and 0 before trial ~80 (D1 and GO are always 0, and so are all a1 choices). Within that late window it is highest (9-10) when the D1 of the same trial showed a reward. It drops to about 5 when that D1 showed no reward or the previous trial was also a rare S1 landing, and to about 0-1 when both are true.
- **108**: It is 0 at every D1 and GO step and only turns on at D2, where the value depends on this trial choice and grows with trial number. After choosing a1, D2 is about 4 near trials 20-40, 5 near 50, 6 near 75 and 9-10 from trial 90 on; after choosing a2 it is about 0-1 up to trial 70, 2-4 around trials 75-90 and 7-8 from trial 95 on, and whether the state reached was common or rare, or whether a reward came, barely matters.

### GRU units (raw z)

| # | Correlation (shared) | Beyond timing | Variance explained | Timing share | Active steps | Matters | Worst-situation choice loss |
|---|---|---|---|---|---|---|---|
| 32 | 0.97 | 0.95 | 0.94 | 0.46 | 97% | no | 2.2% after a2 S1 r |
| 40 | 0.96 | 0.91 | 0.91 | 0.61 | 82% | yes | 11.5% after a1 S2 r |
| 5 | 0.93 | 0.88 | 0.85 | 0.41 | 18% | yes | 20.1% after a1 S2 r |
| 19 | 0.93 | 0.88 | 0.86 | 0.42 | 17% | yes | 31.0% after a1 S2 r |
| 38 | 0.96 | 0.85 | 0.90 | 0.76 | 44% | yes | 9.5% after a1 S2 r |

- **32**: At D1 and GO the unit sits at 10 no matter what happened (now and then 9 at D1), and at D2 its value depends mainly on the choice just made: about 9-10 after choosing a2 (around 8-9 after a2 leads to a rare S1), but about 0-1 after a1 leads to a common S1 (up to 2-3 if a1 was chosen several trials in a row). After a1 leads to a rare S2 it is in the middle, usually 3-5 and sometimes up to 8, while rewards and time in the episode barely matter.
- **40**: It is always 10 at D2. At GO it mostly tracks the choice: about 9 for a2 (10 if the previous trial was unrewarded), about 0 for a1 after a reward and 1-2 for a1 after no reward, but 4-6 when the agent has just switched from a2 to a1; at D1 it is about 5 while the agent is on an a2 streak (about 7 right after an unrewarded rare trial, falling to 0-2 after two or more unrewarded a2 trials in a row), and 0-2 while it is choosing a1.
- **5**: This unit is about 0 at every GO and D2 step, and at D1 it depends only on whether the previous trial ended in S2 (reward does not matter): if it ended in S1 the value is 0. If it ended in S2, the value is 10 when that S2 came from a rare transition after choosing a1, or was the first S2 after S1 visits, and about 6-8 (usually 7) during a steady run of choosing a2 and reaching S2.
- **19**: Only D1 is ever above zero: at D1 it is high when the previous trial ended in state S2, and 0 when it ended in S1, whatever the reward was. It is about 10 if that S2 was a rare transition (the agent chose a1) and about 6-9 if it was common (the agent chose a2), with about 6 in the first ~15 trials and 7-9 later; GO and D2 are always 0.
- **38**: Only D1 matters, and its value depends on which state the previous trial reached: 8-10 if it was S2 (10 if that trial went unrewarded, about 8-9 if rewarded), but if it was S1 it is about 2-4 when rewarded and 4-6 when unrewarded. GO is 0-2 and D2 is almost always 0, rising to 1-4 only during runs of unrewarded a2 choices.

## Candidate

### Transcoder features

| # | Correlation (shared) | Beyond timing | Variance explained | Timing share | Active steps | Matters | Worst-situation choice loss |
|---|---|---|---|---|---|---|---|
| 219 | 1.00 | 1.00 | 1.00 | 0.26 | 11% | yes | 19.3% after a1 S1 u |
| 210 | 1.00 | 1.00 | 1.00 | 0.13 | 5% | yes | 19.7% after a2 S1 r |
| 147 | 1.00 | 1.00 | 0.99 | 0.12 | 5% | yes | 42.5% after a1 S2 r |
| 167 | 1.00 | 1.00 | 0.99 | 0.32 | 13% | yes | 24.7% after a2 S1 r |
| 10 | 0.99 | 0.99 | 0.99 | 0.27 | 8% | yes | 4.9% after a2 S1 r |

- **219**: This fires only at D2, when the agent chose a1 and reached S1 by a common transition: 10 late in the episode (around trial 85 and later) and about 9 earlier (8 to 9 before trial 25). It is 0 at every D1 and GO step, and 0 at D2 for every other outcome (a1 rare to S2, or any a2 choice, common or rare), whatever the reward history.
- **210**: This component fires only at D2 when the agent chose a2 but rarely ended up in S1: that step gets about 9-10 (10 more often earlier in the episode, about 9 later on), whatever the previous rewards were. Every other step is 0, including all D1 and GO steps and every D2 where a1 led to S1, a2 led to S2, or a1 rarely led to S2.
- **147**: This fires only at D2 on trials where the agent chose a1 and got the rare transition to S2: it is about 9 the first time that happens, and 10 when the previous trial was also a rare a1-to-S2 trial or the episode is late (after trial 85). At every other step it is 0: all D1 and GO steps, every common transition, and every trial where the agent chose a2, including a2 trials that rarely reach S1.
- **167**: The value is 0 at every D1 and GO step, and at D2 too unless the agent chose a2 and reached S2 by the common transition; reaching S2 by a rare transition after a1, or reaching S1, gives 0. On those a2-to-S2 common D2 steps it is about 7 early in the episode (trials up to about 40), 8 in the middle (about 40 to 85) and 9 late (about 85 to 99), plus 1 more (up to 10) when the previous trial was unrewarded.
- **10**: Zero at D1 and GO, and zero at D2 unless the agent chose a1 and reached S1 by the common transition. In that case the value falls with time in the episode: about 10 in trials 1-15 (9 if the previous trial was unrewarded, 7-8 just after a rare trial), about 9 around trials 20-30, 7 around trials 40-45, 2-3 around trials 70-73, and 0 from about trial 75 on.

### GRU units (raw n)

| # | Correlation (shared) | Beyond timing | Variance explained | Timing share | Active steps | Matters | Worst-situation choice loss |
|---|---|---|---|---|---|---|---|
| 31 | 0.96 | 0.97 | 0.91 | 0.08 | 89% | yes | 11.4% after a2 S2 u |
| 19 | 0.93 | 0.93 | 0.87 | 0.16 | 94% | yes | 32.2% after a1 S2 r |
| 13 | 0.99 | 0.93 | 0.97 | 0.87 | 63% | yes | 8.4% after a2 S2 u |
| 0 | 0.96 | 0.93 | 0.91 | 0.44 | 40% | yes | 5.8% after a1 S2 r |
| 5 | 0.92 | 0.93 | 0.83 | 0.10 | 88% | yes | 42.1% after a1 S2 r |

- **31**: At D2 it tracks the state shown: about 10 when S1 is reached and 0-2 when S2 is reached, whichever action was chosen. At D1 it reflects the last trial state and reward: about 2-3 after a rewarded S1, 4-5 after a rewarded rare S2 (chose a1), 5-6 after an unrewarded S1, about 7 after a rewarded common S2 (chose a2), and 8-10 after an unrewarded S2 or a run of losses; at GO it sits at 6-7, falling to 5 after an unrewarded trial and rising to 8 when the agent switches to a2 or right after a rewarded rare trial.
- **19**: This unit tracks the agent current preference for a2/S2: while the agent keeps choosing a2 it stays at 9-10 at D1 and GO and about 9 at a D2 reaching S2 (about 7 after a rare S1, 4-6 if that follows a loss), dipping to 6-9 at D1 after an unrewarded trial (lower, 2-5, after repeated losses or early in an a2 run). While the agent keeps choosing a1 it is low: D2 is 0-1 (S1) or 1-4 (rare S2), GO is 2-3 early in the episode and 4-7 late, D1 is 7-9 if the previous trial was rewarded and 0-1 if it was not.
- **13**: GO is always 0, and D1 is high: 10 while the agent has been choosing a2 steadily, 8-9 during runs of a1 choices, and about 5 at trial 0. D2 depends on the state reached: 7-10 for S1 reached rarely after choosing a2, 3-6 for S1 reached commonly after a1 (5-6 right after switching to a1 or after no reward, 2-3 during steady rewarded a1 runs), 1-2 for S2 reached commonly after a2, and 0 for S2 reached rarely after a1.
- **0**: This unit tracks choosing a1: at GO it is 10 when the agent picks a1 (9 if the previous trial went unrewarded, 7-8 once a1 has lost several times in a row), and 0-1 when it picks a2 (3-5 on the first a2 choice right after a run of a1). At D2 it is always 0, and at D1 it is 0-1 after a rewarded trial or during an a2 run, but 3-7 after an unrewarded trial during an a1 run, rising with each additional loss in a row.
- **5**: This unit tracks how strongly the agent currently favors a2. While it keeps choosing a2 it stays at 8-10 at every step (D1 is 10 after a reward and 9 after no reward), but while it keeps choosing a1 it sits at 0-1 during GO and D2, rising only at D1 to about 3-5 after a reward (up to 8 after a long rewarded a1 streak) and 0 after no reward; it slides toward the middle (about 4-7) during switches or after a run of unrewarded a2 trials, and at D2 it is about 2 higher when a rare transition (reaching S2) follows choosing a1.

