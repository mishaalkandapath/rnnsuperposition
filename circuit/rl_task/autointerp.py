"""Activation-based feature interpretation for the RL (two-step task) transcoders.

Same commands and protocol as circuit/copy_task/autointerp.py, so the workflow
prompts carry over. Works on a feature store (circuit/feature_store.py) whose
sequences are RL trace-window keys (circuit/rl_find_features.py). Every
(window, step) gets a row of derived variables; a feature's activations are a
dense vector over those rows, and explanations are predicates over the
variables (numpy expressions), scored exactly.

Windows are up to 5 trials (3 context trials, the core trial, the next trial)
x 3 phases. Overlapping windows of one episode share trials, so the split is by
EPISODE: even episode ids = dev (profiles, iteration), odd = test (reported).

Commands:
    python -m circuit.rl_task.autointerp build   STORE --traces DIR     (once: step table from the trace windows)
    python -m circuit.rl_task.autointerp profile STORE GATE FEATURE
    python -m circuit.rl_task.autointerp score   STORE GATE FEATURE "PREDICATE" [--split dev|test]
    python -m circuit.rl_task.autointerp misses  STORE GATE FEATURE "PREDICATE" [--split dev|test]
    python -m circuit.rl_task.autointerp record  STORE GATE FEATURE RESULT_JSON --out DIR
    python -m circuit.rl_task.autointerp report  DIR
    python -m circuit.rl_task.autointerp variables

Task: each trial is DELAY_1 (fixate; see the previous trial's reward), GO
(choose a1 or a2), DELAY_2 (see the state reached: a1 -> S1 and a2 -> S2 with
p = 0.8 "common", otherwise "rare"). The reached state pays 1 with p = 0.9 for
the currently good state, 0.1 for the other; the reward is seen at the next
trial's DELAY_1. The good state reverses a few times per episode.
"""
import argparse
import glob
import json
import os
import pickle
import shutil
from collections import defaultdict
from typing import Dict

import numpy as np

from circuit.copy_task.autointerp import report, score  # task-agnostic
from circuit.feature_store import open_feature_store

PHASES = {0: "D1", 1: "GO", 2: "D2"}
D1, GO, D2 = 0, 1, 2
FIX, S1, S2 = 0, 1, 2           # states (and a1 = 1, a2 = 2 for actions; 0 = fixation)
PAD = 5
STARTS = ["high_first", "low_first"]
PATTERNS = ["common rewarded", "common unrewarded", "rare rewarded", "rare unrewarded"]

# (description, when it is known to the agent). "future" = not yet observable at that step.
VARIABLE_DOCS = {
    "step": "step index in the window (3 * trial + phase)",
    "trial": "trial index in the window (0 .. n_trials-1)",
    "phase": "0 = D1 (DELAY_1: fixate, previous reward shown), 1 = GO (choose), 2 = D2 (DELAY_2: reached state shown)",
    "n_trials": "real trials in the window (almost always 5)",
    "rel": "trial relative to the window's core trial: -3 .. -1 context, 0 core, +1 next trial",
    "is_core": "1 on the core trial's steps",
    "in_state": "state INPUT at this step: 0 fixation (D1, GO), 1 S1, 2 S2 (D2)",
    "in_action": "last-action INPUT: 0 fixation, 1 a1, 2 a2 (at D2: this trial's GO choice; at GO: the D1 action)",
    "in_rew": "reward INPUT: 1 rewarded, 0 not, -1 penalty (at D1: the previous trial's outcome)",
    "go": "go-signal input (1 at GO)",
    "action": "this trial's GO choice (1 a1, 2 a2, 0 fixation = invalid). Known from D2; at D1/GO it is the UPCOMING choice (future)",
    "state": "state this trial reaches (1/2; 0 if invalid). Known from D2; at D1/GO it is FUTURE",
    "common": "1 if this trial's transition is common (state == action), 0 rare, -1 invalid. Known from D2; FUTURE at D1/GO",
    "reward": "this trial's outcome (1/0, -1 penalty, -2 unknown: no next trial in the window). FUTURE on all of this trial's steps (seen at the next D1)",
    "prev_reward": "previous trial's outcome (= the reward input at this trial's D1; 0 on the episode's first trial)",
    "prev_action": "previous trial's choice (-1 on the window's first trial)",
    "prev_state": "previous trial's reached state (-1 on the first trial)",
    "prev_common": "previous trial's transition common (1) / rare (0); -1 unknown",
    "stay": "1 if this trial's choice repeats the previous trial's, 0 switch, -1 unknown. FUTURE at D1/GO",
    "next_action": "next trial's choice (-1 if not in the window). FUTURE",
    "next_stay": "1 if the next trial repeats this trial's choice, 0 switch, -1 unknown. FUTURE",
    "d1_err": "1 if the agent did not fixate at this trial's D1 (seen as in_action at GO)",
    "valid": "1 unless the GO choice was fixation or the agent acted at D2",
    "ep_trial": "trial index in the episode (0 .. 99)",
    "ep_bucket": "ep_trial bucketed: 0, 1, 2, 3 = 3..9, 4 = 10..49, 5 = 50..99",
    "start_high": "1 if the episode started with S1 good (0.9 / 0.1), 0 if S2 good",
    "core_pattern": "the window core trial's pattern: 0 common rewarded, 1 common unrewarded, 2 rare rewarded, 3 rare unrewarded",
    "p_s1": "TRUE reward probability of S1 on this trial (0.9 or 0.1; hidden from the agent); -1 unknown (between a reversal)",
    "good": "the currently good state (1 S1, 2 S2; hidden from the agent), -1 unknown",
    "chose_good": "1 if this trial's choice is the action whose common state is currently good, 0 if not, -1 unknown. FUTURE at D1/GO",
    "since_rev": "trials since the last reversal of the good state (capped at 20; -1 unknown); hidden from the agent",
    "belief": "ideal-observer P(S1 is the good state) on this trial, from the episode's observed choices, states and "
              "rewards before this trial (known from this trial's D1; float 0-1). Computed from inputs only (not the "
              "model), with a reversal hazard of 0.025 per trial; across trials missing from the data it only decays "
              "toward 0.5, so it is weaker where belief_cover is low",
    "belief_cover": "how many of the previous 10 trials of the episode the belief could use (0-10)",
    "surprisal": "how unexpected the previous trial's outcome was (seen at this trial's D1), in nats: "
                 "surprisal_transition + surprisal_reward; 0 where the previous trial is not in the data",
    "surprisal_transition": "-log P(state reached | choice) of the previous trial: 1.61 rare, 0.22 common",
    "surprisal_reward": "-log P(reward outcome | state reached, the belief on that trial) of the previous trial: "
                        "small for an expected outcome (0.1 for a reward at a confidently-good state), up to 2.3",
    "value": "the model's value-head output at this step (float)",
    "choice": "the model's policy logit(a1) - logit(a2) at this step (float; meaningful at GO)",
    "model_act": "argmax of the model's 3 policy logits at this step (0 fixate, 1 a1, 2 a2)",
}
HELPERS_DOC = ("Helpers: tv('col', d) -> trial-level column col (action, state, common, reward, prev_reward, stay, "
               "good, chose_good, p_s1, ...) of the trial d trials away in the same window (d < 0 back, d > 0 ahead; "
               "-9 outside the window); isin(col, [v, ...]); D1, GO, D2 (phase codes); S1, S2; numpy as np. "
               "Combine with & | ~ and parentheses, e.g. "
               "\"(phase == D1) & (prev_reward == 1) & (prev_common == 0)\" or "
               "\"(phase == GO) & (tv('action', -1) == 2) & (tv('reward', -1) == 1)\". "
               "Variables marked FUTURE are not yet observable to the agent at that step: if one explains the "
               "feature, check whether an observed variable does as well (e.g. the agent's own upcoming choice is "
               "computed; a future reward or transition cannot be known).")
TRIAL_COLS = ("action", "state", "common", "reward", "prev_reward", "prev_action", "prev_state", "prev_common",
              "stay", "next_action", "next_stay", "d1_err", "valid", "ep_trial", "p_s1", "good", "chose_good",
              "since_rev", "belief", "belief_cover", "surprisal", "surprisal_transition", "surprisal_reward")
FLOAT_COLS = ("value", "choice", "p_s1")

# --observed: variables the agent could not use at that step. Model outputs (value, choice, model_act), hidden task
# variables (good, p_s1, since_rev, chose_good, start_high), data-construction labels (core_pattern, is_core,
# n_trials) and this or a later trial's choice / state / transition / reward (use in_action / in_state at D2, and
# prev_* or tv(col, d < 0) for earlier trials).
UNOBSERVED = ("value", "choice", "model_act", "good", "p_s1", "since_rev", "chose_good", "start_high", "core_pattern",
              "is_core", "n_trials", "action", "state", "common", "reward", "stay", "next_action", "next_stay")
TV_UNOBSERVED = ("good", "p_s1", "since_rev", "chose_good")


def check_observed(expr: str):
    """Raise ValueError if the predicate uses a variable the agent could not observe at that step."""
    import re
    for col, d in re.findall(r"tv\(\s*['\"](\w+)['\"]\s*,\s*([-+]?\d+)\s*\)", expr):
        if int(d) >= 0 or col in TV_UNOBSERVED or col.startswith("next"):
            raise ValueError(f"--observed: tv('{col}', {d}) is not observable at that step")
    bare = re.sub(r"tv\([^)]*\)", " ", expr)
    bare = re.sub(r"['\"][^'\"]*['\"]", " ", bare)
    used = set(re.findall(r"(?<![\w.])([A-Za-z_]\w*)", bare)) & set(UNOBSERVED)
    if used:
        raise ValueError(f"--observed: {sorted(used)} not allowed (model outputs, hidden or future variables; "
                         f"use in_action / in_state at D2, prev_* or tv(col, d < 0) for earlier trials)")


# ----------------------------------------------------------------------------- step table

TRACE_KEYS = ("inputs", "outputs", "seq_id", "core_trial", "trial_in_episode", "start_high", "pattern", "p_s1")


def _load_traces(trace_dir, keys=TRACE_KEYS):
    """Trace-window files (datasets/transcoder_rl_datasets.py), keeping only `keys`."""
    import torch
    from torch.utils.data import StackDataset
    torch.serialization.add_safe_globals([StackDataset])
    paths = sorted(glob.glob(os.path.join(trace_dir, "*_rl.pt")))
    if not paths:
        raise SystemExit(f"no *_rl.pt trace-window files in {trace_dir}")
    for path in paths:
        cols = torch.load(path, map_location="cpu").datasets
        yield path, {k: cols[k].numpy() for k in keys}
        del cols


def match_windows(sequences, windows):
    """(file index, row) of one trace window per store key (identical windows share a key; keep the first)."""
    import torch
    from circuit.rl_find_features import RLFeatureActivationAnalyzer
    key_index = {k: i for i, k in enumerate(sequences)}
    analyzer = RLFeatureActivationAnalyzer(None, None, None, "cpu")
    chosen = [None] * len(sequences)
    for wi, d in enumerate(windows):
        for i in range(len(d["inputs"])):
            w = {k: d[k][i] for k in ("inputs", "outputs", "core_trial", "start_high", "pattern")}
            key = analyzer.window_key({k: torch.as_tensor(v) for k, v in w.items()})
            j = key_index.get(key)
            if j is not None and chosen[j] is None:
                chosen[j] = (wi, i)
    missing = sum(c is None for c in chosen)
    if missing:
        raise ValueError(f"{missing} store sequences have no matching trace window; wrong --traces?")
    return chosen


def n_padding(x):
    """Leading padding trials of one window's inputs (trials, phases, features)."""
    return int((x == PAD).reshape(x.shape[0], -1).all(-1).sum())


def _allocation_by_trial(windows) -> Dict[int, Dict[int, float]]:
    """{episode: {episode trial: p_s1}} from the windows' core trials, gaps filled
    where the nearest known trials on both sides agree (the allocation only
    changes at reversals)."""
    known = defaultdict(dict)
    for d in windows:
        for s, t, p in zip(d["seq_id"], d["trial_in_episode"], d["p_s1"]):
            known[int(s)][int(t)] = round(float(p), 2)
    filled = {}
    for ep, kt in known.items():
        ts = sorted(kt)
        full = {}
        for a, b in zip(ts[:-1], ts[1:]):
            full[a] = kt[a]
            if kt[a] == kt[b]:
                for t in range(a + 1, b):
                    full[t] = kt[a]
        full[ts[-1]] = kt[ts[-1]]
        filled[ep] = full
    return filled


def build_step_table(store_path: str, trace_dir: str) -> Dict[str, np.ndarray]:
    """Step table over the store's sequences (keys), from the trace windows."""
    _, sequences, _ = open_feature_store(store_path)
    windows = [d for _, d in _load_traces(trace_dir)]
    alloc = _allocation_by_trial(windows)
    chosen = match_windows(sequences, windows)

    rows = defaultdict(list)
    for j, (wi, i) in enumerate(chosen):
        d = windows[wi]
        x, out = d["inputs"][i], d["outputs"][i]
        n_pad = n_padding(x)
        x, out = x[n_pad:], out[n_pad:]
        n = x.shape[0]
        core = int(d["core_trial"][i]) - n_pad
        ep, ep_core = int(d["seq_id"][i]), int(d["trial_in_episode"][i])
        state_in, act_in, rew_in = x[..., 0:3].argmax(-1), x[..., 3:6].argmax(-1), x[..., 6]
        rew_code = np.where(rew_in > 0.5, 1, np.where(rew_in < -0.05, -1, 0))
        action, state = act_in[:, D2], state_in[:, D2]
        d2_act = np.full(n, 0)                                     # action taken at D2, seen at next D1
        d2_act[:-1] = act_in[1:, D1]
        valid = (action != FIX) & (d2_act == FIX)
        common = np.where(action == FIX, -1, (state == action).astype(int))
        reward = np.full(n, -2); reward[:-1] = rew_code[1:, D1]
        prev = lambda arr, fill: np.concatenate([[fill], arr[:-1]])
        nxt = lambda arr, fill: np.concatenate([arr[1:], [fill]])
        prev_reward = rew_code[:, D1]                               # observed input, also on the window's first trial
        prev_action, prev_state, prev_common = prev(action, -1), prev(state, -1), prev(common, -1)
        stay = np.where(prev_action < 0, -1, (action == prev_action).astype(int))
        next_action = nxt(action, -1)
        next_stay = np.where(next_action < 0, -1, (next_action == action).astype(int))
        ep_trial = ep_core - core + np.arange(n)
        p_s1 = np.array([alloc[ep].get(int(t), -1.0) for t in ep_trial], dtype=np.float32)
        good = np.where(p_s1 < 0, -1, np.where(p_s1 > 0.5, S1, S2))
        chose_good = np.where((good < 0) | (action == FIX), -1, (action == good).astype(int))
        # trials since the good state last changed (within the known stretch of the episode)
        ep_alloc = alloc[ep]
        since = np.full(n, -1)
        for k, t in enumerate(ep_trial):
            if p_s1[k] < 0:
                continue
            back = 0
            while back < 20 and ep_alloc.get(int(t) - back - 1) == p_s1[k]:
                back += 1
            since[k] = back if (back == 20 or ep_alloc.get(int(t) - back - 1) is not None or t - back == 0) else -1
        trial_vals = dict(action=action, state=state, common=common, reward=reward, prev_reward=prev_reward,
                          prev_action=prev_action, prev_state=prev_state, prev_common=prev_common, stay=stay,
                          next_action=next_action, next_stay=next_stay, d1_err=(act_in[:, GO] != FIX).astype(int),
                          valid=valid.astype(int), ep_trial=ep_trial, p_s1=p_s1, good=good,
                          chose_good=chose_good, since_rev=since)
        for name, vals in trial_vals.items():
            rows[name].append(np.repeat(vals, 3))
        trial = np.repeat(np.arange(n), 3)
        rows["seq"].append(np.full(3 * n, j)); rows["step"].append(np.arange(3 * n))
        rows["trial"].append(trial); rows["phase"].append(np.tile([D1, GO, D2], n))
        rows["n_trials"].append(np.full(3 * n, n)); rows["rel"].append(trial - core)
        rows["is_core"].append((trial == core).astype(int))
        rows["in_state"].append(state_in.reshape(-1)); rows["in_action"].append(act_in.reshape(-1))
        rows["in_rew"].append(rew_code.reshape(-1)); rows["go"].append(x[..., 7].reshape(-1).round().astype(int))
        rows["start_high"].append(np.full(3 * n, int(bool(d["start_high"][i]))))
        rows["core_pattern"].append(np.full(3 * n, int(d["pattern"][i])))
        rows["episode"].append(np.full(3 * n, ep))
        rows["value"].append(out[..., 3].reshape(-1)); rows["choice"].append((out[..., 1] - out[..., 2]).reshape(-1))
        rows["model_act"].append(out[..., :3].argmax(-1).reshape(-1))
    cols = {}
    for name, parts in rows.items():
        arr = np.concatenate(parts)
        cols[name] = (arr.astype(np.float32) if name in FLOAT_COLS
                      else arr.astype(np.int64) if name == "episode"
                      else arr.astype(np.int32) if name == "seq" else arr.astype(np.int16))
    et = cols["ep_trial"]
    cols["ep_bucket"] = np.select([et == 0, et == 1, et == 2, et < 10, et < 50], [0, 1, 2, 3, 4], 5).astype(np.int16)
    lengths = np.bincount(cols["seq"].astype(np.int64), minlength=len(sequences))
    cols["row_start"] = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return cols


HAZARD = 0.025          # reversals: 1-4 per 100-trial episode, uniform over trials
P_GOOD, P_BAD = 0.9, 0.1


def belief_columns(cols: Dict[str, np.ndarray], n_trials: int = 100):
    """Ideal-observer belief P(S1 good) per row, from every trial of the episode present in the table.

    A trial's outcome (state reached, reward) is used once it is observed (from the next trial's D1);
    across trials missing from the table, and invalid trials, only the reversal hazard applies.
    """
    ep, t = np.asarray(cols["episode"]).astype(np.int64), np.asarray(cols["ep_trial"]).astype(np.int64)
    st, rw, va = np.asarray(cols["state"]), np.asarray(cols["reward"]), np.asarray(cols["valid"])
    eps, ep_idx = np.unique(ep, return_inverse=True)
    known = (st > 0) & (rw >= 0) & (va == 1)
    state = np.zeros((len(eps), n_trials), dtype=np.int8)               # 0 = not observed
    reward = np.zeros((len(eps), n_trials), dtype=np.int8)
    state[ep_idx[known], t[known]] = st[known]
    reward[ep_idx[known], t[known]] = rw[known]
    b = np.full((len(eps), n_trials), 0.5)
    cover = np.zeros((len(eps), n_trials), dtype=np.int16)
    cur = np.full(len(eps), 0.5)
    for k in range(1, n_trials):
        s, r = state[:, k - 1], reward[:, k - 1]
        obs = s > 0
        like_s1 = np.where(s == 1, np.where(r == 1, P_GOOD, 1 - P_GOOD), np.where(r == 1, P_BAD, 1 - P_BAD))
        like_s2 = np.where(s == 2, np.where(r == 1, P_GOOD, 1 - P_GOOD), np.where(r == 1, P_BAD, 1 - P_BAD))
        post = np.where(obs, cur * like_s1 / (cur * like_s1 + (1 - cur) * like_s2), cur)
        cur = post * (1 - HAZARD) + (1 - post) * HAZARD
        b[:, k] = cur
        cover[:, k] = (state[:, max(0, k - 10):k] > 0).sum(1)
    return b[ep_idx, t].astype(np.float32), cover[ep_idx, t]


def surprisal_columns(cols: Dict[str, np.ndarray], belief: np.ndarray):
    """How unexpected the previous trial's outcome was (shown at this trial's D1), in nats, per row:
    transition part -log P(state reached | choice) (rare 1.61, common 0.22) and reward part
    -log P(reward | state reached, the observer's belief on that trial). Uses the episode's trials in the
    table, so it is known at a window's first trial too; 0 where the previous trial is not in the table."""
    ep, t = np.asarray(cols["episode"]).astype(np.int64), np.asarray(cols["ep_trial"]).astype(np.int64)
    st, rw, ac, va = (np.asarray(cols[k]) for k in ("state", "reward", "action", "valid"))
    eps, ep_idx = np.unique(ep, return_inverse=True)
    known = (st > 0) & (rw >= 0) & (ac > 0) & (va == 1)
    n_trials = 100
    s_tr, s_rw = np.zeros((len(eps), n_trials)), np.zeros((len(eps), n_trials))
    b = np.full((len(eps), n_trials), 0.5)
    b[ep_idx, t] = belief
    k = t[known]
    p_good_state = np.where(st[known] == S1, b[ep_idx[known], k], 1 - b[ep_idx[known], k])
    p_rew = p_good_state * P_GOOD + (1 - p_good_state) * P_BAD
    nxt = k + 1 < n_trials                                   # stored on the next trial, where it is seen
    s_tr[ep_idx[known][nxt], k[nxt] + 1] = -np.log(np.where(st[known] == ac[known], 0.8, 0.2))[nxt]
    s_rw[ep_idx[known][nxt], k[nxt] + 1] = -np.log(np.where(rw[known] == 1, p_rew, 1 - p_rew))[nxt]
    return s_tr[ep_idx, t].astype(np.float32), s_rw[ep_idx, t].astype(np.float32)


def add_belief(store_path: str):
    cache = os.path.join(store_path, "autointerp_steps")
    cols = {k: np.load(os.path.join(cache, f"{k}.npy"), mmap_mode="r")
            for k in ("episode", "ep_trial", "state", "reward", "action", "valid")}
    belief, cover = belief_columns(cols)
    s_tr, s_rw = surprisal_columns(cols, belief)
    np.save(os.path.join(cache, "belief.npy"), belief)
    np.save(os.path.join(cache, "belief_cover.npy"), cover)
    np.save(os.path.join(cache, "surprisal_transition.npy"), s_tr)
    np.save(os.path.join(cache, "surprisal_reward.npy"), s_rw)
    np.save(os.path.join(cache, "surprisal.npy"), s_tr + s_rw)
    return belief, cover


class Interp:
    """Step table + activations for one store (same interface as the copy-task Interp)."""

    def __init__(self, store_path: str):
        self.store_path = store_path
        self._sequences = None
        cache = os.path.join(store_path, "autointerp_steps")
        if not os.path.exists(os.path.join(cache, "DONE")):
            raise SystemExit(f"No step table in {store_path}: run "
                             f"`python -m circuit.rl_task.autointerp build {store_path} --traces DIR` first")
        self.cols = {name[:-4]: np.load(os.path.join(cache, name), mmap_mode="r")
                     for name in os.listdir(cache) if name.endswith(".npy")}
        self.n_rows = len(self.cols["step"])
        # Split by episode: overlapping windows of one episode share trials.
        self.test = (self.cols["episode"] % 2).astype(bool)
        tr = self.cols["trial"].astype(np.int64)
        self._trial_start = np.arange(self.n_rows) - 3 * tr - self.cols["phase"]   # row of trial 0, D1
        self._n_trials = self.cols["n_trials"].astype(np.int64)

    @staticmethod
    def build(store_path: str, trace_dir: str):
        cache = os.path.join(store_path, "autointerp_steps")
        tmp = f"{cache}.tmp{os.getpid()}"
        os.makedirs(tmp, exist_ok=True)
        cols = build_step_table(store_path, trace_dir)
        for name, col in cols.items():
            np.save(os.path.join(tmp, f"{name}.npy"), np.ascontiguousarray(col))
        open(os.path.join(tmp, "DONE"), "w").close()
        if os.path.exists(cache):
            shutil.rmtree(cache)
        os.rename(tmp, cache)
        return cols

    @property
    def sequences(self):
        if self._sequences is None:
            with open(os.path.join(self.store_path, "sequences.pkl"), "rb") as f:
                self._sequences = pickle.load(f)
        return self._sequences

    def tv(self, col: str, d: int) -> np.ndarray:
        """Trial-level column `col` of the trial d trials away, same window, same phase row (-9 outside)."""
        if col not in TRIAL_COLS:
            raise ValueError(f"tv() takes a trial-level column, one of {TRIAL_COLS}")
        target = self.cols["trial"].astype(np.int64) + d
        ok = (target >= 0) & (target < self._n_trials)
        rows = np.where(ok, self._trial_start + 3 * np.clip(target, 0, None), 0)
        vals = np.asarray(self.cols[col])[rows]
        return np.where(ok, vals, -9).astype(vals.dtype)

    def activations(self, gate: str, feature: int) -> np.ndarray:
        act = np.zeros(self.n_rows, dtype=np.float32)
        path = os.path.join(self.store_path, gate, f"{feature}.npz")
        if not os.path.exists(path):
            return act
        d = np.load(path)
        lengths = np.diff(d["offsets"])
        rows = np.repeat(self.cols["row_start"][d["seq_ids"]], lengths) + d["positions"].astype(np.int64)
        act[rows] = d["magnitudes"]
        return act

    def namespace(self):
        ns = {k: v for k, v in self.cols.items() if k not in ("row_start", "episode")}
        ns.update(np=np, isin=lambda col, vals: np.isin(col, vals), tv=self.tv,
                  D1=D1, GO=GO, D2=D2, S1=S1, S2=S2)
        return ns

    def predicate(self, expr: str) -> np.ndarray:
        out = eval(expr, {"__builtins__": {}}, self.namespace())
        out = np.broadcast_to(np.asarray(out), (self.n_rows,))
        if out.dtype != bool:
            raise ValueError("predicate must evaluate to a boolean array")
        return out

    # ------------------------------------------------------------------ rendering
    def render(self, s: int, act: np.ndarray, max_mag: float, pred=None) -> str:
        """One window: per trial [rel] D1 r<prev reward> | GO a<choice> | D2 S<state><c common / R rare>,
        each cell with its activation (relative to max) when active; *cell* where the predicate is true."""
        c = self.cols
        lo, hi = int(c["row_start"][s]), int(c["row_start"][s + 1])
        rew = lambda v: {1: "1", 0: "0", -1: "-", -2: "?"}.get(int(v), "?")
        parts = []
        for r0 in range(lo, hi, 3):
            core = "*" if c["is_core"][r0] else ""
            cells = [f"r{rew(c['prev_reward'][r0])}", f"a{int(c['action'][r0])}",
                     f"S{int(c['state'][r0])}{'c' if c['common'][r0] == 1 else 'R' if c['common'][r0] == 0 else '!'}"]
            out = []
            for k, cell in enumerate(cells):
                r = r0 + k
                txt = f"{cell}:{act[r] / max_mag:.2f}" if act[r] > 0 else cell
                out.append(f"*{txt}*" if pred is not None and pred[r] else txt)
            good = {1: "S1", 2: "S2"}.get(int(c["good"][r0]), "?")
            parts.append(f"[{int(c['rel'][r0]):+d}{core} good={good}] " + " ".join(out))
        head = f"{STARTS[1 - int(c['start_high'][lo])]}, core {PATTERNS[int(c['core_pattern'][lo])]}"
        return f"({head}) " + " | ".join(parts)


# ----------------------------------------------------------------------------- profile / misses

def profile(ip: Interp, gate: str, feature: int, n_examples: int = 10) -> str:
    act = ip.activations(gate, feature)
    dev = ~ip.test
    a = act[dev]
    c = {k: np.asarray(v)[dev] for k, v in ip.cols.items() if k not in ("row_start",)}
    active = a > 0
    lines = [f"# Feature {gate}/{feature} (dev split: even episode ids)"]
    n_act, n_rows = int(active.sum()), len(a)
    n_seqs = len(np.unique(c["seq"][active])) if n_act else 0
    n_eps = len(np.unique(c["episode"][active])) if n_act else 0
    lines.append(f"active on {n_act} of {n_rows} steps ({n_act / n_rows:.4%}), in {n_seqs} of "
                 f"{len(np.unique(c['seq']))} windows and {n_eps} of {len(np.unique(c['episode']))} episodes")
    if n_act == 0:
        lines.append("Feature never fires on the dev split.")
        return "\n".join(lines)
    mags = a[active]
    lines.append(f"magnitude: mean {mags.mean():.3f}, median {np.median(mags):.3f}, max {mags.max():.3f}")
    base = n_act / n_rows

    def table(name, values=None, labels=None):
        vals = c[name] if values is None else values
        lines.append(f"\n## by {name}: value | steps | active | rate | lift | share of activation mass")
        for k in np.unique(vals):
            sel = vals == k
            n = int(sel.sum())
            k_act = int((active & sel).sum())
            rate = k_act / n if n else 0.0
            lines.append(f"  {labels(k) if labels else k}: {n} | {k_act} | {rate:.3f} | {rate / base:.1f}x | "
                         f"{float(a[sel].sum()) / float(a.sum()):.3f}")

    table("phase", labels=lambda k: PHASES[int(k)])
    rel_phase = c["rel"].astype(np.int64) * 3 + c["phase"]
    table("rel x phase", rel_phase, labels=lambda k: f"rel {int(k) // 3:+d} {PHASES[int(k) % 3]}")
    for name in ("in_rew", "in_action", "in_state", "action", "state", "common", "reward", "prev_reward",
                 "prev_action", "prev_common", "stay", "next_action", "next_stay", "good", "chose_good",
                 "since_rev", "start_high", "core_pattern", "ep_bucket", "d1_err", "valid", "model_act"):
        table(name)
    # time in the episode (100 trials): features often switch on / off at a trial count
    table("ep_trial (10-trial bins)", c["ep_trial"].astype(np.int64) // 10,
          labels=lambda k: f"trials {int(k) * 10}-{int(k) * 10 + 9}")
    # model outputs, by quantile bin (dev rows)
    for name in ("value", "choice"):
        q = np.quantile(c[name], [0.1, 0.3, 0.5, 0.7, 0.9])
        bins = np.searchsorted(q, c[name])
        edges = [-np.inf] + list(q) + [np.inf]
        table(f"{name} (quantile bin)", bins, labels=lambda k, e=edges: f"[{e[int(k)]:.2f}, {e[int(k) + 1]:.2f})")

    order = np.argsort(-a)
    dev_index = np.nonzero(dev)[0]
    max_mag = float(mags.max())
    lines.append("\n## top activating windows: [rel(*=core) good=state] D1 r<prev reward> | GO a<choice> | "
                 "D2 S<state><c=common R=rare>, cell:activation/max when active; one line per window")
    seen = set()
    for i in order:
        s = int(c["seq"][i])
        if s in seen or a[i] <= 0:
            continue
        seen.add(s)
        lines.append("  " + ip.render(s, act, max_mag))
        if len(seen) >= n_examples:
            break
    rng = np.random.default_rng(0)
    active_idx = np.nonzero(active)[0]
    qs = np.quantile(mags, [0.0, 0.25, 0.5, 0.75])
    for lo_q, hi_q, name in ((qs[0], qs[1], "lowest quartile"), (qs[1], qs[2], "second quartile"),
                             (qs[2], qs[3], "third quartile")):
        pool = active_idx[(a[active_idx] >= lo_q) & (a[active_idx] < hi_q)]
        if len(pool):
            lines.append(f"\n## random examples from the {name} of activations")
            for i in rng.choice(pool, min(4, len(pool)), replace=False):
                lines.append(f"  (step {int(c['step'][i])}) " + ip.render(int(c["seq"][i]), act, max_mag))
    return "\n".join(lines)


def misses(ip: Interp, gate: str, feature: int, expr: str, split: str = "dev", n: int = 8) -> str:
    act = ip.activations(gate, feature)
    pred = ip.predicate(expr)
    sel = ip.test if split == "test" else ~ip.test
    max_mag = float(act.max()) if act.max() > 0 else 1.0
    rng = np.random.default_rng(0)
    lines = [f"# Counterexamples for {gate}/{feature} ({split} split)", f"predicate: {expr}",
             "Notation: cell:a = active with activation a (relative to the max); *cell* = predicate true there."]
    for title, mask in (("ACTIVE but predicate FALSE (missed activations)", (act > 0) & ~pred & sel),
                        ("predicate TRUE but INACTIVE (false positives)", (act == 0) & pred & sel)):
        rows = np.nonzero(mask)[0]
        lines.append(f"\n## {title}: {len(rows)} steps")
        if title.startswith("ACTIVE") and len(rows):
            lines.append(f"   (these hold {float(act[rows].sum()) / float(act[sel].sum()):.1%} of the activation mass)")
        for r in (rng.choice(rows, min(n, len(rows)), replace=False) if len(rows) else []):
            lines.append(f"  step {int(ip.cols['step'][r])} ({PHASES[int(ip.cols['phase'][r])]}, rel "
                         f"{int(ip.cols['rel'][r]):+d}): " + ip.render(int(ip.cols["seq"][r]), act, max_mag, pred))
    return "\n".join(lines)


def apply_skeptic(ip: Interp, results_dir: str) -> Dict[str, int]:
    """Make the skeptic's better predicate the recorded one, re-scored by the tool on dev and test.

    The skeptic's corrected label is already the final label (autointerp_runs merge); this does the same
    for the predicate, so the scores and every downstream test use the corrected explanation. The
    original predicate and scores are kept under original_*.
    """
    counts = {"adopted": 0, "failed": 0, "no_better_predicate": 0}
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".json") or name == "summary.json":
            continue
        path = os.path.join(results_dir, name)
        rec = json.load(open(path))
        better = (rec.get("skeptic") or {}).get("better_predicate")
        if not better or rec.get("predicate_from") == "skeptic":
            counts["no_better_predicate"] += not better
            continue
        try:
            test, dev = (score(ip, rec["gate"], rec["feature"], better, s) for s in ("test", "dev"))
        except Exception as e:                  # an unparseable suggestion keeps the original
            rec["skeptic_predicate_error"] = str(e)[:200]
            counts["failed"] += 1
        else:
            rec.update(original_predicate=rec.get("predicate"), original_test_scores=rec.get("test_scores"),
                       original_dev_scores=rec.get("dev_scores"), predicate=better, test_scores=test,
                       dev_scores=dev, predicate_from="skeptic")
            counts["adopted"] += 1
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)
    return counts


# ----------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="Activation-based feature interpretation (RL two-step task)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("store"); b.add_argument("--traces", required=True)
    sub.add_parser("belief", help="add the ideal-observer belief columns to a built step table").add_argument("store")
    ak = sub.add_parser("apply_skeptic", help="adopt the skeptic's better predicates in a results directory (re-scored)")
    ak.add_argument("store"); ak.add_argument("results")
    p = sub.add_parser("profile"); p.add_argument("store"); p.add_argument("gate"); p.add_argument("feature", type=int)
    s = sub.add_parser("score"); s.add_argument("store"); s.add_argument("gate"); s.add_argument("feature", type=int)
    s.add_argument("predicate"); s.add_argument("--split", choices=["dev", "test"], default="dev")
    s.add_argument("--observed", action="store_true", help="reject variables the agent could not observe")
    r = sub.add_parser("record"); r.add_argument("store"); r.add_argument("gate"); r.add_argument("feature", type=int)
    r.add_argument("result_json", help='{"label": ..., "predicate": ... or null, "verdict": ..., "notes": ...}')
    r.add_argument("--out", required=True)
    r.add_argument("--observed", action="store_true", help="reject variables the agent could not observe")
    m = sub.add_parser("misses"); m.add_argument("store"); m.add_argument("gate"); m.add_argument("feature", type=int)
    m.add_argument("predicate"); m.add_argument("--split", choices=["dev", "test"], default="dev")
    m.add_argument("--observed", action="store_true", help="reject variables the agent could not observe")
    rp = sub.add_parser("report"); rp.add_argument("out_dir")
    sub.add_parser("variables")
    args = ap.parse_args()

    if args.cmd == "variables":
        for k, v in VARIABLE_DOCS.items():
            print(f"{k:13s} {v}")
        print(HELPERS_DOC)
        return
    if args.cmd == "report":
        print(report(args.out_dir))
        return
    if args.cmd == "belief":
        belief, cover = add_belief(args.store)
        print(f"belief: mean {belief.mean():.3f}; rows with >= 8 of the previous 10 trials observed: "
              f"{(cover >= 8).mean():.1%}, with none: {(cover == 0).mean():.1%}")
        return
    if args.cmd == "build":
        cols = Interp.build(args.store, args.traces)
        print(f"built step table: {len(cols['step'])} rows over {len(cols['row_start']) - 1} windows, "
              f"{len(np.unique(cols['episode']))} episodes; p_s1 known on {(cols['p_s1'] >= 0).mean():.1%} of rows")
        return
    if getattr(args, "observed", False):
        pred = args.predicate if args.cmd != "record" else json.loads(args.result_json).get("predicate")
        if pred:
            try:
                check_observed(pred)
            except ValueError as e:
                raise SystemExit(str(e))
    ip = Interp(args.store)
    if args.cmd == "apply_skeptic":
        print(f"{args.results}: {apply_skeptic(ip, args.results)}")
        return
    if args.cmd == "profile":
        print(profile(ip, args.gate, args.feature))
    elif args.cmd == "score":
        print(json.dumps(score(ip, args.gate, args.feature, args.predicate, args.split), indent=2))
    elif args.cmd == "misses":
        print(misses(ip, args.gate, args.feature, args.predicate, args.split))
    elif args.cmd == "record":
        result = json.loads(args.result_json)
        if result.get("predicate"):
            result["test_scores"] = score(ip, args.gate, args.feature, result["predicate"], "test")
            result["dev_scores"] = score(ip, args.gate, args.feature, result["predicate"], "dev")
        result.update(gate=args.gate, feature=args.feature)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"{args.gate}_{args.feature}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"recorded {path}")
        if "test_scores" in result:
            t = result["test_scores"]
            print(f"test: precision {t['precision']:.3f}, recall {t['recall']:.3f}, f1 {t['f1']:.3f}, "
                  f"mass recall {t['mass_recall']:.3f}")


if __name__ == "__main__":
    main()
