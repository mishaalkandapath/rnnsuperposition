"""Does the value head react to an outcome as much as the outcome changes the expected return?

If V(h) = E[G | h] (what training on (G - V)^2 aims at), then for any group of steps picked by what the agent has
observed, the mean of V equals the mean of the actual return G. So comparing V and G group by group tests whether V
is right, with no model of the task needed. Closed-loop episodes (reversal_flow.rollout); G is the discounted
return-to-go (gamma 0.97 per step, value_circuit.returns); both taken beyond their time profile (minus the mean at
that step of the episode). At the next D1 after a trial, rewarded minus unrewarded, separately for trials that
reached the state the ideal observer believed good and the one it believed bad:
  dV   the value head's difference
  dG   the actual return's difference (what a correct value would show)

    python -m circuit.rl_task.value_check --out rl_features/rl_circuits/value_check.json [--episodes 3000]
"""
import argparse
import json
import os

import numpy as np

from circuit.rl_task.contrast_flow import load_model
from circuit.rl_task.reversal_flow import observer, rollout
from circuit.rl_task.value_circuit import returns


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rnn, _, _ = load_model()
    _, _, rec = rollout(rnn, args.episodes, args.seed)
    V, G = rec["value"], returns(rec["step_reward"])
    Vb, Gb = V - V.mean(0), G - G.mean(0)
    b = observer(rec)
    ch, st, rw = rec["choice"], rec["state"], rec["reward"]
    out = {"episodes": args.episodes}
    for name, want_good in (("believed-good state reached", True), ("believed-bad state reached", False)):
        groups = {}
        for r in (1, 0):
            v, g = [], []
            for t in range(1, 98):
                m = (ch[:, t] > 0) & (st[:, t] > 0) & (rw[:, t] == r) & (((b[:, t] > 0.5) == (st[:, t] == 1)) == want_good)
                v += Vb[m, 3 * (t + 1)].tolist(); g += Gb[m, 3 * (t + 1)].tolist()
            groups[r] = (np.array(v), np.array(g))
        (v1, g1), (v0, g0) = groups[1], groups[0]
        se = lambda a, c: float(np.sqrt(a.var() / len(a) + c.var() / len(c)))
        out[name] = {"dV": round(float(v1.mean() - v0.mean()), 3), "dV_se": round(se(v1, v0), 3),
                     "dG": round(float(g1.mean() - g0.mean()), 3), "dG_se": round(se(g1, g0), 3),
                     "n_rewarded": len(v1), "n_unrewarded": len(v0)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
