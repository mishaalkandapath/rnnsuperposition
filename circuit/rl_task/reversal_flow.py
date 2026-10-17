"""How the two-step-task GRU switches its choice after the good state reverses: behaviour around reversals and the
contrastive attribution graph of each switch.

rollout    closed-loop episodes of the real agent (100 trials, reversal hazard 0.025 as in training, actions sampled
           from its policy as in training and in the trace data).
behaviour  around each reversal after which the agent had settled (it chose the old good action, the one commonly
           leading to the good state, on at least 4 of the 5 trials before): P(new good action) by trials since the
           reversal, for the agent and for the ideal observer's greedy choice on the agent's own outcomes, and the
           agent's choice logit oriented towards the new action; the two-step stay table P(stay | previous transition x reward).
switches   the first trial S after such a reversal (within 15 trials) on which the agent chooses the new action and
           also chooses it on the next trial. Window: trials S-K .. S from the carried state. Target: the choice at
           S.GO, oriented towards the new action (logit new - logit old).
           Counterfactual "old world": every reward of trials S-K .. S-1 replaced by the outcome the old belief
           expects (1 in the old good state, 0 in the old bad state); also "last": only trial S-1's reward replaced.
           Exact contrastive flow through the features of all three gates (contrast_flow.full_flow): how much of the
           switch each trial's outcome supplies, and which features carry it from trial to trial.

    python -m circuit.rl_task.reversal_flow --out rl_features/rl_circuits/reversal_flow.json [--episodes 2000]
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from circuit.rl_task.contrast_flow import PHASES, REW, full_flow, gru_run, load_model
from datasets.task_datasets import BatchedTwoStepRLEnvironment

HAZARD, P_GOOD = 0.025, 0.9


@torch.no_grad()
def rollout(rnn, n_episodes, seed, n_trials=100, hook=None):
    """Closed-loop episodes: inputs (N, 3T, 8), state entering each step (N, 3T, H), per trial (N, T): choice
    (0 = invalid), state reached, reward, good state, choice logit a1 - a2 at GO; per step (N, 3T): the value
    output and the reward received (penalties included). hook(phase, h) -> h edits the state after each step."""
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    gru, readout = rnn.layers[0], rnn.layers[-1]
    env = BatchedTwoStepRLEnvironment(n_episodes, reward_switch_prob=HAZARD)
    env.set_batch_size(n_episodes, high_first=True)
    flip = torch.rand(n_episodes) < 0.5                                  # half the episodes start with S2 good
    env.reward_probs[flip] = env.reward_probs[flip][:, [0, 2, 1]]
    h = rnn.initial_states[0].detach().expand(n_episodes, -1).clone()
    X, Hp = [], []
    rec = {k: torch.zeros(n_episodes, n_trials, dtype=torch.long) for k in ("choice", "state", "good")}
    rec["reward"] = torch.zeros(n_episodes, n_trials)
    rec["logit"] = torch.zeros(n_episodes, n_trials)
    rec["value"] = torch.zeros(n_episodes, 3 * n_trials)                 # per step
    rec["step_reward"] = torch.zeros(n_episodes, 3 * n_trials)           # per step, penalties included
    for t in range(n_trials):
        env.reset_trial_batch()
        rec["good"][:, t] = torch.where(env.reward_probs[:, 1] > 0.5, 1, 2)
        for ph in range(3):
            x = env.get_input_vector_batch().float()
            X.append(x); Hp.append(h)
            r = torch.sigmoid(gru.input_to_reset(x) + gru.hidden_to_reset(h))
            z = torch.sigmoid(gru.input_to_update(x) + gru.hidden_to_update(h))
            n = torch.tanh(gru.input_to_new(x) + gru.hidden_to_new(r * h))
            h = (1 - z) * h + z * n
            if hook is not None:
                h = hook(ph, h)
            out = readout(h)
            a = torch.multinomial(torch.softmax(out[:, :3], -1), 1)[:, 0]
            if ph == 1:
                rec["logit"][:, t] = out[:, 1] - out[:, 2]
                rec["choice"][:, t] = torch.where(a > 0, a, torch.zeros_like(a))
            rec["value"][:, 3 * t + ph] = out[:, 3]
            _, rw = env.step_batch(a)
            rec["step_reward"][:, 3 * t + ph] = rw
            if ph == 1:
                rec["state"][:, t] = env.current_state
            if ph == 2:
                rec["reward"][:, t] = rw
    return torch.stack(X, 1), torch.stack(Hp, 1), {k: v.numpy() for k, v in rec.items()}


def observer(rec):
    """Ideal-observer P(S1 good) before each trial, from the agent's own states and rewards."""
    N, T = rec["choice"].shape
    b = np.full((N, T), 0.5)
    cur = np.full(N, 0.5)
    for t in range(1, T):
        s, r, ok = rec["state"][:, t - 1], rec["reward"][:, t - 1], (rec["choice"][:, t - 1] > 0)
        ok = ok & (r >= 0)
        l1 = np.where(s == 1, np.where(r == 1, P_GOOD, 1 - P_GOOD), np.where(r == 1, 1 - P_GOOD, P_GOOD))
        l2 = np.where(s == 2, np.where(r == 1, P_GOOD, 1 - P_GOOD), np.where(r == 1, 1 - P_GOOD, P_GOOD))
        post = np.where(ok, cur * l1 / (cur * l1 + (1 - cur) * l2), cur)
        cur = post * (1 - HAZARD) + (1 - post) * HAZARD
        b[:, t] = cur
    return b


def settled_reversals(rec, before=5, after=15):
    """(episode, reversal trial R, old action, new action) where the agent chose the old good action on >= 4 of
    the 5 trials before R."""
    out = []
    good, ch = rec["good"], rec["choice"]
    for e, R in zip(*np.nonzero(good[:, 1:] != good[:, :-1])):
        R = R + 1
        if R < before or R + after > good.shape[1]:
            continue
        old_a, new_a = int(good[e, R - 1]), int(good[e, R])                # action k commonly leads to state k
        if (good[e, R - before:R] == good[e, R - 1]).all() and (ch[e, R - before:R] == old_a).sum() >= 4:
            out.append((int(e), int(R), old_a, new_a))
    return out


def behaviour(rec, rev, after=15):
    b = observer(rec)
    ideal = np.where(b > 0.5, 1, 2)
    agent_c, ideal_c, logit = defaultdict(list), defaultdict(list), defaultdict(list)
    for e, R, old_a, new_a in rev:
        for k in range(-5, after):
            if rec["choice"][e, R + k] > 0:
                agent_c[k].append(rec["choice"][e, R + k] == new_a)
            ideal_c[k].append(ideal[e, R + k] == new_a)
            logit[k].append(rec["logit"][e, R + k] * (1 if new_a == 1 else -1))
    ch, st, rw = rec["choice"], rec["state"], rec["reward"]
    ok = (ch[:, :-1] > 0) & (ch[:, 1:] > 0) & (rw[:, :-1] >= 0)
    common = st[:, :-1] == ch[:, :-1]
    stay = ch[:, 1:] == ch[:, :-1]
    table = {f"{'common' if c else 'rare'} {'rewarded' if r else 'unrewarded'}":
             round(float(stay[ok & (common == c) & (rw[:, :-1] == r)].mean()), 3) for c in (1, 0) for r in (1, 0)}
    valid = ch > 0
    return {"n_reversals": len(rev),
            "p_new_action_by_trials_since_reversal": {k: {"agent": round(float(np.mean(agent_c[k])), 3),
                                                          "ideal_observer": round(float(np.mean(ideal_c[k])), 3)}
                                                      for k in sorted(agent_c)},
            "choice_logit_towards_new_action": {k: {"mean": round(float(np.mean(logit[k])), 2),
                                                    "q10_q90": [round(float(q), 2) for q in np.quantile(logit[k], [0.1, 0.9])]}
                                                for k in sorted(logit)},
            "stay_table": table,
            "agreement_with_ideal_observer": round(float((ch[valid] == ideal[valid]).mean()), 3)}


def switch_events(rec, rev, after=15):
    ev = []
    for e, R, old_a, new_a in rev:
        for S in range(R, min(R + after, rec["choice"].shape[1] - 1)):
            if rec["choice"][e, S] == new_a and rec["choice"][e, S + 1] == new_a:
                ev.append((e, R, S, old_a, new_a))
                break
    return ev


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--K", type=int, default=5, help="trials before the switch trial in the window")
    ap.add_argument("--max_events", type=int, default=600)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rnn, tcs, tracer = load_model()
    X, Hp, rec = rollout(rnn, args.episodes, args.seed)
    rev = settled_reversals(rec)
    result = {"episodes": args.episodes, "K": args.K, "behaviour": behaviour(rec, rev)}
    print(json.dumps(result["behaviour"]), flush=True)
    ev = switch_events(rec, rev)
    ev = [x for x in ev if x[2] - args.K >= 0][:args.max_events]
    b = observer(rec)
    result["switches"] = {"n": len(ev), "trials_after_reversal": np.bincount([S - R for _, R, S, _, _ in ev]).tolist(),
                          "observer_p_new_at_switch_median": round(float(np.median(
                              [b[e, S] if new_a == 1 else 1 - b[e, S] for e, R, S, _, new_a in ev])), 3)}
    print(f"{len(ev)} switch events ({time.time() - t0:.0f}s)", flush=True)
    K = args.K
    for mode in ("old_world", "last"):
        agg = {"d_target": [], "src": defaultdict(float), "node": defaultdict(float), "count": defaultdict(int),
               "checks": [], "edges": defaultdict(float)}
        for i, (e, R, S, old_a, new_a) in enumerate(ev):
            lo = S - K
            x = X[e, 3 * lo:3 * S + 2].clone()
            x_cf = x.clone()
            ks = range(K) if mode == "old_world" else [K - 1]
            for k in ks:                                             # trial lo + k's reward is shown at lo + k + 1's D1
                t = lo + k
                if rec["choice"][e, t] > 0 and rec["reward"][e, t] >= 0:
                    x_cf[3 * (k + 1), REW] = float(rec["state"][e, t] == old_a)
            if torch.equal(x, x_cf):
                continue
            acts = [tracer.run_forward_pass(gru_run(rnn, xx, Hp[e, 3 * lo])) for xx in (x, x_cf)]
            sign = 1.0 if new_a == 1 else -1.0
            f = full_flow(tracer, acts[0], acts[1], sign * tracer.W_choice, 3 * K + 1)
            agg["d_target"].append(f["d_target"]); agg["checks"].append(max(abs(f["check"]), f["preact_check"]))
            name = lambda t: f"S{'' if t // 3 == K else f'{t // 3 - K:+d}'}.{PHASES[t % 3]}"
            for key, v in f["into"].items():
                for (_, t, _), val in zip(f["nodes"], v.tolist()):
                    agg["src"][f"{key}@{name(t)}"] += val
            for key, v in f["target"].items():
                agg["src"][key] += v
            for (kd, t, fi), fl, da in zip(f["nodes"], f["flow"].tolist(), f["da"].tolist()):
                if da != 0:
                    agg["node"][f"{name(t)}/{kd}:{fi}"] += fl; agg["count"][f"{name(t)}/{kd}:{fi}"] += 1
            E = f["edges"]
            for gi, fi in torch.nonzero(E.abs() > 0.05).tolist():
                (kg, tg, fg), (kf, tf, ff) = f["nodes"][gi], f["nodes"][fi]
                if tf // 3 != tg // 3:                                   # edges that carry evidence across trials
                    agg["edges"][f"{name(tf)}/{kf}:{ff} -> {name(tg)}/{kg}:{fg}"] += float(E[gi, fi])
            if i % 100 == 0:
                print(f"{mode}: {i}/{len(ev)} ({time.time() - t0:.0f}s)", flush=True)
        n = len(agg["d_target"])
        mean = lambda dct: {k: round(v / n, 4) for k, v in sorted(dct.items(), key=lambda kv: -abs(kv[1]))
                            if abs(v / n) >= 1e-3}
        node = mean(agg["node"])
        by_step = defaultdict(list)
        for nm, v in node.items():
            by_step[nm.split("/")[0]].append((nm.split("/")[1], v))
        result[mode] = {"n": n, "d_target_mean": round(float(np.mean(agg["d_target"])), 3),
                        "d_target_sd": round(float(np.std(agg["d_target"])), 3),
                        "max_check": float(np.max(agg["checks"])), "flow_by_source": mean(agg["src"]),
                        "flow_through_step": {s: round(sum(v for _, v in fs), 3) for s, fs in by_step.items()},
                        "top_features_by_step": {s: [{"feature": fi, "flow": v, "windows_changed": agg["count"][f"{s}/{fi}"]}
                                                     for fi, v in fs[:args.top]] for s, fs in by_step.items()},
                        "top_cross_trial_edges": dict(list(mean(agg["edges"]).items())[:40])}
        print(f"{mode}: n={n} d_target={result[mode]['d_target_mean']:+.2f} check={result[mode]['max_check']:.1e}",
              flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
