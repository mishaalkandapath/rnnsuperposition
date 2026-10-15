"""Probe RL transcoder features that matter for behaviour but no firing predicate explains.

RL version of circuit/copy_task/feature_probe.py, for a few features of one
gate's transcoder (e.g. the partially explained candidate features that
circuit/rl_task/explained_fidelity.py found essential):

  profile      firing rate and mean activation by phase; pairwise correlation
  regression   how well one-hot cue groups predict the activation *magnitude*
               (ridge, held-out R^2; windows split by parity): timing (phase x
               episode-trial decile) alone, plus each cue group, all, all minus each
  conditional  which information the model needs from the feature: replace it
  ablation     by its mean conditional on a cue (global, phase, timing, the
               previous reward, the previous trial's choice / state / reward,
               the last two trials, this trial's choice, ...) and run the GRU.
               If conditioning on a cue restores behaviour (switch / GO
               agreement, value error), that cue is what the feature carries.
  tables       activation minus its (phase, episode-trial decile) mean, by
               phase x the previous trial's pattern, and at D1 by the last two
               trials' patterns
  decoder      candidate features only: the direct readout of the decoder
               vector (policy logit(a1) - logit(a2), value) and decoder cosines

Cues are computed from the inputs (what the agent observes) except `reward`
(this trial's outcome, FUTURE until the next D1) and the model-output bins
(the original GRU's policy at this trial's GO and value at this step).

    python -m circuit.rl_task.feature_probe --rnn RNN.ckpt --transcoder C.ckpt --gate candidate \\
        --features 208 66 24 215 --controls 180 --traces rl_features/rl_trace_sequences --out probe.json
"""
import argparse
import json
import os

import numpy as np
import torch

from circuit.copy_task.explained_fidelity import load_transcoder
from circuit.rl_task.explained_fidelity import behaviour_metrics, load_eval, run_all, teacher_forced
from training.splice_eval import build_task_rnn

UNKNOWN = 3                                   # category for "not in the window"
PATTERN = {(1, 1, 1): "a1 S1 r", (1, 1, 0): "a1 S1 u", (1, 2, 1): "a1 S2 r", (1, 2, 0): "a1 S2 u",
           (2, 2, 1): "a2 S2 r", (2, 2, 0): "a2 S2 u", (2, 1, 1): "a2 S1 r", (2, 1, 0): "a2 S1 u"}
PATTERN_ORDER = list(PATTERN.values())        # common then rare, per action


def cues(groups, refs):
    """Per-step cue columns, flattened in teacher_forced order (group, window, step)."""
    cols = {k: [] for k in ("phase", "trial", "ep_dec", "prev_reward", "action", "state", "reward",
                            "prev_action", "prev_state", "prev2_action", "prev2_state", "prev2_reward",
                            "choice_bin", "value", "choice", "window")}
    base = 0
    for g, ref in zip(groups, refs):
        x = g["inputs"]
        n, T, _ = x.shape
        k = T // 3
        state = x[:, 2::3, 0:3].argmax(-1)                                    # (n, k) state reached (D2)
        action = x[:, 2::3, 3:6].argmax(-1)                                   # GO choice, seen at D2
        rew_in = x[:, 0::3, 6]
        prev_reward = torch.where(rew_in > 0.5, 1, 0)                         # reward input at this trial's D1
        unk = torch.full((n, 1), UNKNOWN)
        shift = lambda a, d: torch.cat([unk.expand(n, d), a[:, :k - d]], 1) if d > 0 else \
            torch.cat([a[:, -d:], unk.expand(n, -d)], 1)
        trial_cols = {"prev_reward": prev_reward, "action": action, "state": state,
                      "reward": shift(prev_reward, -1), "prev_action": shift(action, 1),
                      "prev_state": shift(state, 1), "prev2_action": shift(action, 2),
                      "prev2_state": shift(state, 2), "prev2_reward": shift(prev_reward, 1)}
        go_choice = ref[:, 1::3, 1] - ref[:, 1::3, 2]                          # original policy at GO
        trial_cols["choice_go"] = go_choice
        rep = lambda a: a.repeat_interleave(3, 1)
        t = torch.arange(T).expand(n, T)
        ep_trial = g["ep_trial0"][:, None] + t // 3
        add = {"phase": t % 3, "trial": t // 3, "ep_dec": (ep_trial // 10).clamp(0, 9),
               "value": ref[..., 3], "choice": rep(go_choice),
               "window": (torch.arange(n) + base)[:, None].expand(n, T)}
        add.update({c: rep(trial_cols[c]) for c in cols if c in trial_cols})
        for c, v in add.items():
            cols[c].append(v.flatten())
        base += n
    out = {c: torch.cat(v).numpy() for c, v in cols.items() if v}
    q = np.quantile(out["choice"], [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
    out["choice_bin"] = np.searchsorted(q, out["choice"])
    q = np.quantile(out["value"], [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
    out["value_bin"] = np.searchsorted(q, out["value"])
    return out


def codes(c, key):
    out = np.zeros(len(c["phase"]), dtype=np.int64)
    for k in key:
        out = out * 16 + c[k].astype(np.int64)
    return out


def conditional_means(c, y, key):
    """Mean of y within each cell of the cue combination `key`."""
    if not key:
        return np.full_like(y, y.mean())
    _, inv = np.unique(codes(c, key), return_inverse=True)
    return (np.bincount(inv, weights=y) / np.bincount(inv))[inv]


LAST = ("prev_action", "prev_state", "prev_reward")
LAST2 = LAST + ("prev2_action", "prev2_state", "prev2_reward")
CONDITIONS = [("global", ()), ("phase", ("phase",)), ("phase, timing", ("phase", "ep_dec")),
              ("phase, prev reward", ("phase", "prev_reward")),
              ("phase, prev reward x prev state", ("phase", "prev_reward", "prev_state")),
              ("phase, last trial", ("phase",) + LAST),
              ("phase, last trial, timing", ("phase", "ep_dec") + LAST),
              ("phase, last two trials", ("phase",) + LAST2),
              ("phase, this choice", ("phase", "action")),
              ("phase, last trial, this choice", ("phase", "action") + LAST),
              ("phase, policy bin (model)", ("phase", "choice_bin")),
              ("phase, value bin (model)", ("phase", "value_bin"))]
CUE_GROUPS = {"last trial": ("phase",) + LAST, "trial before": ("phase", "prev2_action", "prev2_state", "prev2_reward"),
              "this choice": ("phase", "action"), "this state": ("phase", "state"),
              "this reward (future)": ("phase", "reward"), "policy bin (model)": ("phase", "choice_bin"),
              "value bin (model)": ("phase", "value_bin")}
TIMING = ("phase", "ep_dec")


def regression(c, acts):
    from scipy import sparse
    from sklearn.linear_model import Ridge
    test = c["window"] % 2 == 1

    def onehot(key):
        _, inv = np.unique(codes(c, key), return_inverse=True)
        return sparse.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)), shape=(len(inv), inv.max() + 1))

    blocks = {name: onehot(key) for name, key in [("timing", TIMING)] + list(CUE_GROUPS.items())}

    def r2(names, y):
        X = sparse.hstack([blocks[n] for n in names]).tocsr()
        m = Ridge(alpha=1.0).fit(X[~test], y[~test])
        resid = y[test] - m.predict(X[test])
        return float(1 - (resid ** 2).sum() / ((y[test] - y[test].mean()) ** 2).sum())

    out, groups = {}, list(CUE_GROUPS)
    for fid, y in acts.items():
        res = {"timing": r2(["timing"], y)}
        for gname in groups:
            res[f"timing+{gname}"] = r2(["timing", gname], y)
        observed = [g for g in groups if "future" not in g and "model" not in g]
        res["observed"] = r2(["timing"] + observed, y)
        res["all"] = r2(["timing"] + groups, y)
        for gname in groups:
            res[f"all-{gname}"] = r2(["timing"] + [g for g in groups if g != gname], y)
        out[fid] = res
        print(f"  feature {fid}: " + "  ".join(f"{k} {v:.2f}" for k, v in res.items()), flush=True)
    return out


@torch.no_grad()
def run_override(rnn, tc, x, h0, gate, override):
    """Readouts with the gate replaced by its transcoder, chosen features set to override[f] (n, T)."""
    gru, readout = rnn.layers[0], rnn.layers[-1]
    h, out = h0, []
    ids = list(override)
    for t in range(x.shape[1]):
        xt = x[:, t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        f = tc.act(tc.input_to_features(torch.cat([r * h if gate == "candidate" else h, xt], -1)))
        if ids:
            f[:, ids] = torch.stack([override[i][:, t] for i in ids], 1)
        val = tc.features_to_outputs(f)
        if gate == "reset":
            r = val
        if gate == "update":
            z = val
        n_t = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        if gate == "candidate":
            n_t = val
        h = (1 - z) * h + z * n_t
        out.append(readout(h))
    return torch.stack(out, 1)


def run_flat(rnn, tc, groups, gate, override_flat, chunk=4096):
    """Readouts per length group, with overrides given as flat per-step arrays."""
    outs, start = [], 0
    for g in groups:
        n, T = g["inputs"].shape[:2]
        ov = {i: torch.from_numpy(v[start:start + n * T]).float().reshape(n, T) for i, v in override_flat.items()}
        start += n * T
        outs.append(torch.cat([run_override(rnn, tc, g["inputs"][s:s + chunk], g["h0"][s:s + chunk], gate,
                                            {i: o[s:s + chunk] for i, o in ov.items()})
                               for s in range(0, n, chunk)]))
    return outs


def behaviour(rnn, tc, groups, refs, gate, c, acts, sets):
    fmt = lambda m: f"switch {m['switch_agreement']:.3f} go {m['go_agreement']:.3f} value {m['value_nmae']:.3f}"
    plain = behaviour_metrics(run_flat(rnn, tc, groups, gate, {}), refs)
    print(f"  plain transcoder: {fmt(plain)}", flush=True)
    results = {"plain_transcoder": plain, "sets": []}
    for name, ids in sets:
        row = {"set": name, "features": ids, "conditions": {}}
        for cname, key in CONDITIONS:
            m = behaviour_metrics(run_flat(rnn, tc, groups, gate, {i: conditional_means(c, acts[i], key) for i in ids}), refs)
            row["conditions"][cname] = m
            print(f"  {name:>8} | {cname:<34} {fmt(m)}", flush=True)
        results["sets"].append(row)
    return results


def tables(c, acts, min_count=50):
    """Residual activation (minus its (phase, decile) mean) by phase x previous-trial pattern,
    and at D1 by the last two trials' patterns."""
    def pattern(a, s, r):
        out = np.full(len(a), -1)
        for k, (key, _) in enumerate(PATTERN.items()):
            out[(a == key[0]) & (s == key[1]) & (r == key[2])] = PATTERN_ORDER.index(PATTERN[key])
        return out
    last = pattern(c["prev_action"], c["prev_state"], c["prev_reward"])
    before = pattern(c["prev2_action"], c["prev2_state"], c["prev2_reward"])
    report = {"patterns": PATTERN_ORDER, "phases": ["D1", "GO", "D2"]}
    for fid, y in acts.items():
        res = y - conditional_means(c, y, TIMING)
        by_phase = []
        for ph in range(3):
            row = []
            for p in range(8):
                sel = (c["phase"] == ph) & (last == p)
                row.append(round(float(res[sel].mean()), 4) if sel.sum() >= min_count else None)
            by_phase.append(row)
        pair = []
        for p2 in range(8):
            row = []
            for p1 in range(8):
                sel = (c["phase"] == 0) & (before == p2) & (last == p1)
                row.append(round(float(res[sel].mean()), 4) if sel.sum() >= min_count else None)
            pair.append(row)
        report[fid] = {"mean": float(y.mean()), "residual_std": float(res.std()),
                       "phase_x_last": by_phase, "d1_before_x_last": pair}
        d1 = sorted(((PATTERN_ORDER[p], v) for p, v in enumerate(by_phase[0]) if v is not None), key=lambda x: -x[1])
        print(f"  feature {fid} at D1 by last trial: " + "  ".join(f"{p} {v:+.2f}" for p, v in d1), flush=True)
    return report


SECTIONS = ["profile", "regression", "decoder", "behaviour", "tables"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rnn", required=True)
    ap.add_argument("--transcoder", required=True)
    ap.add_argument("--gate", default="candidate", choices=["reset", "update", "candidate"])
    ap.add_argument("--features", nargs="+", type=int, required=True)
    ap.add_argument("--controls", nargs="*", type=int, default=[], help="explained features probed for comparison")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--max_windows", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sections", nargs="*", default=SECTIONS, choices=SECTIONS)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    rnn = build_task_rnn("rl", input_size=8, hidden_size=48)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    rnn.eval()
    tc = load_transcoder(args.transcoder)
    groups = load_eval(args.traces, args.max_windows)
    refs = [run_all(rnn, g) for g in groups]
    X, _ = teacher_forced(groups, args.gate)
    probed = list(args.features) + list(args.controls)
    feats = torch.cat([tc.act(tc.input_to_features(X[i:i + 65536]))[:, probed] for i in range(0, len(X), 65536)])
    acts = {f: feats[:, k].numpy().astype(np.float64) for k, f in enumerate(probed)}
    c = cues(groups, refs)
    sections = set(args.sections)
    report = json.load(open(args.out)) if os.path.exists(args.out) else {}
    report["args"] = vars(args)

    if "profile" in sections:
        print("profile")
        report["profile"] = {}
        for f, y in acts.items():
            p = {ph: {"fire_rate": float((y[c["phase"] == k] > 0).mean()), "mean": float(y[c["phase"] == k].mean())}
                 for k, ph in enumerate(["D1", "GO", "D2"])}
            report["profile"][f] = p
            print(f"  feature {f}: " + "  ".join(f"{ph} rate {v['fire_rate']:.2f} mean {v['mean']:.3f}" for ph, v in p.items()))
        report["correlation"] = np.corrcoef(np.stack([acts[f] for f in probed])).round(3).tolist()

    if "regression" in sections:
        print("regression (held-out R^2 of activation magnitude)")
        report["regression"] = regression(c, acts)

    if "decoder" in sections and args.gate == "candidate":
        print("decoder (direct readout of the decoder vector)")
        D = tc.features_to_outputs.weight                                     # (48, feats)
        W = rnn.layers[-1].weight                                             # (4, 48): 3 policy logits, value
        Dn = torch.nn.functional.normalize(D[:, probed], dim=0)
        report["decoder"] = {f: {"norm": float(D[:, f].norm()), "policy_a1_minus_a2": float((W[1] - W[2]) @ D[:, f]),
                                 "value": float(W[3] @ D[:, f])} for f in probed}
        report["decoder_cosine"] = (Dn.T @ Dn).numpy().round(3).tolist()
        for f, d in report["decoder"].items():
            print(f"  feature {f}: |d| {d['norm']:.2f}  policy a1-a2 {d['policy_a1_minus_a2']:+.3f}  value {d['value']:+.3f}")

    if "behaviour" in sections:
        print("conditional ablation (agreement with the original GRU)")
        sets = [(str(f), [f]) for f in probed] + [("all", list(args.features))]
        report["behaviour"] = behaviour(rnn, tc, groups, refs, args.gate, c, acts, sets)

    if "tables" in sections:
        print("residual tables (activation minus its (phase, episode-trial decile) mean)")
        report["tables"] = tables(c, acts)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
