"""Reward prediction errors of the agent, analysed as in Wang et al. (2018, "Prefrontal cortex as a meta-reinforcement
learning system", Simulation 4), with controls for what that analysis can tell apart.

The agent's RPE at each step is the TD error of its own value output, delta_t = r_t + gamma V_{t+1} - V_t
(gamma = 0.97, the training discount; reward is received at the D2 step). Two steps of a trial carry the comparison:
  reveal   the GO step, whose next value is read after the state reached is shown (stage-2 arrival)
  outcome  the D2 step, which receives the reward (the next value is read at the next D1)
Reference RPEs on the same trials (closed-loop episodes, reversal_flow.rollout), for each step:
  bayesian      Wang et al.'s "model-based" reference: the ideal observer's belief b that S1 is the good state;
                expected reward of a state E(S1) = 0.9 b + 0.1 (1 - b); of a choice 0.8 E(common) + 0.2 E(rare);
                reveal = E(state reached) - E(choice); outcome = r - E(state reached)
  mb_rl         Daw et al.'s (2011) model-based RL: second-stage values Q(S) learned by a delta rule (alpha), first
                stage planned from the transition model (0.8 / 0.2); reveal = Q(state) - plan(choice);
                outcome = r - Q(state)
  model_free    SARSA(0): first-stage Q(a) and second-stage Q(S) by delta rules (alpha);
                reveal = Q(state) - Q(choice); outcome = r - Q(state)
  reward_only   no learning at all: 0 at reveal, r - mean reward at outcome
Learning rates are chosen to maximise each reference's correlation with the agent (the most generous comparison).
Correlations pooled over reveal and outcome steps (as in Wang et al.) and per step; the model-free RPE's residual
after regressing out the Bayesian one (Wang et al.'s test), and the agent's RPE regressed on Bayesian and MB-RL
references together.

    python -m circuit.rl_task.rpe_analysis --out rl_features/rl_circuits/rpe_analysis.json
"""
import argparse
import json
import os
import time

import numpy as np

from circuit.rl_task.contrast_flow import load_model
from circuit.rl_task.reversal_flow import observer, rollout

GAMMA = 0.97
ALPHAS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


def agent_rpe(rec):
    V, R = rec["value"], rec["step_reward"]
    Vn = np.concatenate([V[:, 1:], np.zeros((V.shape[0], 1))], 1)
    d = R + GAMMA * Vn - V
    return d[:, 1::3], d[:, 2::3]                                         # reveal (GO), outcome (D2), per trial


def bayes_rpe(rec):
    b = observer(rec)
    ch, st, rw = rec["choice"], rec["state"], np.clip(rec["reward"], 0, 1)
    E = lambda s: np.where(s == 1, 0.9 * b + 0.1 * (1 - b), 0.9 * (1 - b) + 0.1 * b)
    Ec = 0.8 * E(ch) + 0.2 * E(3 - ch)                                    # action a commonly leads to state a
    return E(st) - Ec, rw - E(st)


def mb_rl_rpe(rec, alpha):
    ch, st, rw = rec["choice"], rec["state"], np.clip(rec["reward"], 0, 1)
    N, T = ch.shape
    Q = np.full((N, 3), 0.5)
    rev, out = np.zeros((N, T)), np.zeros((N, T))
    rows = np.arange(N)
    for t in range(T):
        a, s = ch[:, t], st[:, t]
        ok = (a > 0) & (s > 0)
        plan = 0.8 * Q[rows, np.clip(a, 1, 2)] + 0.2 * Q[rows, 3 - np.clip(a, 1, 2)]
        qs = Q[rows, np.clip(s, 1, 2)]
        rev[:, t] = np.where(ok, qs - plan, 0); out[:, t] = np.where(ok, rw[:, t] - qs, 0)
        Q[rows[ok], s[ok]] += alpha * (rw[ok, t] - qs[ok])
    return rev, out


def mf_rpe(rec, alpha):
    ch, st, rw = rec["choice"], rec["state"], np.clip(rec["reward"], 0, 1)
    N, T = ch.shape
    Q1, Q2 = np.full((N, 3), 0.5), np.full((N, 3), 0.5)
    rev, out = np.zeros((N, T)), np.zeros((N, T))
    rows = np.arange(N)
    for t in range(T):
        a, s = ch[:, t], st[:, t]
        ok = (a > 0) & (s > 0)
        q1, q2 = Q1[rows, np.clip(a, 1, 2)], Q2[rows, np.clip(s, 1, 2)]
        d1, d2 = q2 - q1, rw[:, t] - q2
        rev[:, t] = np.where(ok, d1, 0); out[:, t] = np.where(ok, d2, 0)
        Q1[rows[ok], a[ok]] += alpha * d1[ok]
        Q2[rows[ok], s[ok]] += alpha * d2[ok]
    return rev, out


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rnn, tcs, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    ok = (rec["choice"] > 0) & (rec["state"] > 0) & (rec["reward"] >= 0)
    ok[:, 0] = False; ok[:, -1] = False
    ar, ao = agent_rpe(rec)
    refs = {"bayesian": bayes_rpe(rec)}
    pooled = lambda r_, o_: np.concatenate([r_[ok], o_[ok]])
    agent = pooled(ar, ao)
    best = {}
    for name, fn in (("mb_rl", mb_rl_rpe), ("model_free", mf_rpe)):
        scores = {a: corr(agent, pooled(*fn(rec, a))) for a in ALPHAS}
        a_best = max(scores, key=scores.get)
        refs[name] = fn(rec, a_best)
        best[name] = {"alpha": a_best, "pooled_corr_by_alpha": {str(k): round(v, 3) for k, v in scores.items()}}
    rw = np.clip(rec["reward"], 0, 1)
    refs["reward_only"] = (np.zeros_like(rw), rw - rw[ok].mean())
    out = {"episodes": args.episodes, "gamma": GAMMA, "learning_rates": best, "correlation_with_agent_rpe": {}}
    for name, (r_, o_) in refs.items():
        out["correlation_with_agent_rpe"][name] = {
            "pooled (reveal and outcome)": round(corr(agent, pooled(r_, o_)), 3),
            "reveal steps": round(corr(ar[ok], r_[ok]), 3) if r_[ok].std() > 0 else None,
            "outcome steps": round(corr(ao[ok], o_[ok]), 3)}
    # Wang et al.'s test: the model-free RPE's part orthogonal to the Bayesian one
    by, mf, mb = pooled(*refs["bayesian"]), pooled(*refs["model_free"]), pooled(*refs["mb_rl"])
    A = np.column_stack([np.ones_like(by), by])
    mf_res = mf - A @ np.linalg.lstsq(A, mf, rcond=None)[0]
    mb_res = mb - A @ np.linalg.lstsq(A, mb, rcond=None)[0]
    out["wang_test"] = {"corr_agent_with_model_free_residual": round(corr(agent, mf_res), 3),
                        "corr_agent_with_mb_rl_residual": round(corr(agent, mb_res), 3),
                        "corr_between_references": {"bayesian vs mb_rl": round(corr(by, mb), 3),
                                                    "bayesian vs model_free": round(corr(by, mf), 3),
                                                    "bayesian vs reward_only": round(corr(by, pooled(*refs["reward_only"])), 3)}}
    D = np.column_stack([np.ones_like(by), by, mb, mf])
    coef, *_ = np.linalg.lstsq(D, agent, rcond=None)
    sd = D[:, 1:].std(0)
    out["multiple_regression_standardised"] = {k: round(float(c * s / agent.std()), 3)
                                               for k, c, s in zip(("bayesian", "mb_rl", "model_free"), coef[1:], sd)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
