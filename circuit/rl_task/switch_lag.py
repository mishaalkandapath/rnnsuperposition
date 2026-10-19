"""How late does the agent switch after a reversal, compared with the ideal observer on the same evidence?

Closed-loop episodes (reversal_flow.rollout); reversals after which the agent had settled
(reversal_flow.settled_reversals). The ideal observer (reversal_flow.observer) does not know when the reversal
happened: it is the Bayesian belief P(S1 good) from the agent's own states and rewards, knowing only the task's
rules (hazard 0.025, reward probability 0.9). Per reversal, trials from the reversal to the switch, same definition
for both: the first trial on which the choice is the new action and is again on the next trial.
  agent sampled   the agent's actual (sampled) choices
  agent greedy    the agent's preferred action (sign of its choice logit), no sampling noise
  observer        the observer's greedy choice (the action leading to the state it believes good)
Also the first trial on which each prefers the new action (agent logit > 0; observer belief favours the new state).
Paired differences agent minus observer per reversal, overall and split by the old good state (switching away from
S1 / a1 or from S2 / a2).

    python -m circuit.rl_task.switch_lag --out rl_features/rl_circuits/switch_lag.json [--episodes 3000]
"""
import argparse
import json
import os

import numpy as np

from circuit.rl_task.contrast_flow import load_model
from circuit.rl_task.reversal_flow import observer, rollout, settled_reversals


def first(cond, R, after=15, twice=True):
    for S in range(R, R + after - 1):
        if cond[S] and (cond[S + 1] if twice else True):
            return S - R
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rnn, _, _ = load_model()
    _, _, rec = rollout(rnn, args.episodes, args.seed)
    b = observer(rec)
    rows, olds = [], []
    for e, R, old, new in settled_reversals(rec):
        agent = rec["choice"][e] == new
        greedy = rec["logit"][e] * (1 if new == 1 else -1) > 0
        obs = (b[e] > 0.5) == (new == 1)
        r = (first(agent, R), first(greedy, R), first(obs, R), first(greedy, R, twice=False), first(obs, R, twice=False))
        if None not in r:
            rows.append(r); olds.append(old)
    A, olds = np.array(rows), np.array(olds)
    names = ["agent_sampled", "agent_greedy", "observer", "agent_first_prefers_new", "observer_first_prefers_new"]
    out = {"episodes": args.episodes, "reversals": len(A),
           "trials_to_switch": {n: {"median": float(np.median(A[:, i])), "mean": round(float(A[:, i].mean()), 2)}
                                for i, n in enumerate(names)},
           "agent_minus_observer": {}}
    for i, j, n in ((0, 2, "sampled_vs_observer"), (1, 2, "greedy_vs_observer"),
                    (3, 4, "first_preference")):
        d = A[:, i] - A[:, j]
        out["agent_minus_observer"][n] = {
            "median": float(np.median(d)), "mean": round(float(d.mean()), 2),
            "share_agent_later": round(float(np.mean(d > 0)), 3), "share_same": round(float(np.mean(d == 0)), 3),
            "share_agent_earlier": round(float(np.mean(d < 0)), 3),
            "lag_counts": {int(k): int(v) for k, v in zip(*np.unique(d, return_counts=True))}}
    out["by_old_good_state"] = {}
    for old in (1, 2):
        m = olds == old
        d = A[m, 1] - A[m, 2]
        out["by_old_good_state"][f"switching away from S{old}"] = {
            "reversals": int(m.sum()), "agent_greedy_mean": round(float(A[m, 1].mean()), 2),
            "agent_greedy_median": float(np.median(A[m, 1])), "observer_mean": round(float(A[m, 2].mean()), 2),
            "lag_mean": round(float(d.mean()), 2), "share_agent_later": round(float(np.mean(d > 0)), 3),
            "share_same": round(float(np.mean(d == 0)), 3), "share_agent_earlier": round(float(np.mean(d < 0)), 3)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
