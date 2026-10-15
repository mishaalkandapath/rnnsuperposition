"""Are the decision variables localised in a few components (transcoder features) or spread across many (GRU units)?

For each gate and each decision-relevant variable, at D1 steps and at GO steps separately (held-out episodes):
both the variable and every component are first centred within (phase, episode-decile) cells, so timing is removed.
Then, with linear decoders fitted on half the windows and scored on the other half:

  r2_all     R^2 of the variable from all components of the source
  r2_best1   R^2 of the best single component
  k90        components needed (greedy forward selection) to reach 90% of r2_all

Variables (computed from observed inputs; the last is what the components produce):
  belief          ideal-observer log-odds that S1 is the good state (step table `belief`)
  evidence_last   +1 if the previous trial's outcome favours S1 (rewarded at S1 / unrewarded at S2), -1 if S2
  evidence_before the same for the trial before
  choice          the model's policy logit(a1) - logit(a2) at the trial's GO

Sources: all firing transcoder features, all 48 GRU units of the gate, the behaviour-matched sets
(rl_fidelity/matched_sets.json), and a control: the 48 units in a random rotated basis (is the unit basis special?).

    python -m circuit.rl_task.readout_test --out rl_features/rl_fidelity/readout_test.json
"""
import argparse
import json
import os

import numpy as np

from circuit.rl_task.autointerp import Interp

FEATURE_GATE = {"reset": "reset", "update": "update", "candidate": "hidden"}


def cell_center(X, code):
    """Subtract the mean within each cell (columns of X, or a vector)."""
    X = np.asarray(X, dtype=np.float64)
    out = X.copy()
    for v in np.unique(code):
        m = code == v
        out[m] -= X[m].mean(0)
    return out


def r2_fit(Xtr, ytr, Xte, yte, ridge=1e-3):
    if Xtr.shape[1] == 0:
        return 0.0
    A = Xtr.T @ Xtr
    A += ridge * np.trace(A) / max(len(A), 1) * np.eye(len(A))
    w = np.linalg.solve(A, Xtr.T @ ytr)
    res = yte - Xte @ w
    return float(1 - (res ** 2).sum() / max((yte ** 2).sum(), 1e-12))


def greedy_k(Xtr, ytr, Xte, yte, target, k_max=60, ridge=1e-3):
    """Components needed (greedy forward selection on train R^2) until test R^2 >= target.
    Works on covariance matrices, so each candidate costs a small solve."""
    Ctr, ctr, ytt = Xtr.T @ Xtr, Xtr.T @ ytr, float(ytr @ ytr)
    Cte, cte, yte2 = Xte.T @ Xte, Xte.T @ yte, float(yte @ yte)
    lam = ridge * np.trace(Ctr) / max(len(Ctr), 1)

    def weights(cols):
        A = Ctr[np.ix_(cols, cols)] + lam * np.eye(len(cols))
        return np.linalg.solve(A, ctr[cols])

    def r2_train(cols):
        w = weights(cols)
        return 1 - (ytt - 2 * w @ ctr[cols] + w @ Ctr[np.ix_(cols, cols)] @ w) / ytt

    def r2_test(cols):
        w = weights(cols)
        return 1 - (yte2 - 2 * w @ cte[cols] + w @ Cte[np.ix_(cols, cols)] @ w) / yte2

    chosen, rest = [], list(range(Xtr.shape[1]))
    for k in range(1, min(k_max, Xtr.shape[1]) + 1):
        best_j = max(rest, key=lambda j: r2_train(chosen + [j]))
        chosen.append(best_j)
        rest.remove(best_j)
        if r2_test(chosen) >= target:
            return k, chosen
    return None, chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", default="rl_features/rl_chosen/chosen_rl_v1_r128_store")
    ap.add_argument("--unit_values", default="rl_features/rl_unit_values")
    ap.add_argument("--matched", default="rl_features/rl_fidelity/matched_sets.json")
    ap.add_argument("--n_windows", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    ip = Interp(args.store)
    c = {k: np.asarray(v) for k, v in ip.cols.items() if k != "row_start"}
    rs = np.asarray(ip.cols["row_start"])
    test_w = np.nonzero(c["episode"][rs[:-1]] % 2 == 1)[0]
    wins = np.sort(rng.choice(test_w, size=args.n_windows, replace=False))
    rows = np.concatenate([np.arange(rs[w], rs[w + 1]) for w in wins])
    half = np.concatenate([np.full(rs[w + 1] - rs[w], i % 2) for i, w in enumerate(wins)]).astype(bool)   # by window
    phase, code = c["phase"][rows], (c["phase"][rows] * 10 + np.clip(c["ep_trial"][rows] // 10, 0, 9))

    def evidence(state, reward):
        return np.where(state <= 0, 0.0, np.where((state == 1) == (reward == 1), 1.0, -1.0))
    b = np.clip(c["belief"][rows].astype(np.float64), 1e-4, 1 - 1e-4)
    go_logit = c["choice"]                                     # policy logit at each step; take the trial's GO
    trial_go = rows - c["phase"][rows] + 1                     # row of this trial's GO step
    variables = {"belief": np.log(b / (1 - b)),
                 "evidence_last": evidence(c["prev_state"][rows], c["prev_reward"][rows]),
                 "evidence_before": evidence(ip.tv("state", -2)[rows], ip.tv("reward", -2)[rows]),
                 "choice": go_logit[trial_go].astype(np.float64)}
    matched = json.load(open(args.matched))
    counts = json.load(open(os.path.join(args.store, "store.json")))["counts"]
    out = {"args": vars(args)}
    for gate in ("reset", "update", "candidate"):
        feats = sorted(int(f) for f in counts[FEATURE_GATE[gate]])
        F = np.stack([ip.activations(FEATURE_GATE[gate], f)[rows] for f in feats], 1).astype(np.float64)
        U = np.asarray(np.load(os.path.join(args.unit_values, f"{gate}.npy"), mmap_mode="r")[rows], dtype=np.float64)
        Q, _ = np.linalg.qr(rng.standard_normal((48, 48)))
        sources = {"features (all)": F, "units (all)": U, "units, random basis": (U - U.mean(0)) @ Q}
        mf = [feats.index(i) for i in matched[gate]["features"]["ids"] if i in feats]
        sources["features (matched set)"] = F[:, mf]
        sources["units (matched set)"] = U[:, matched[gate]["units"]["ids"]]
        out[gate] = {}
        for ph_name, ph in (("D1", 0), ("GO", 1)):
            sel = phase == ph
            res = {}
            for vname, y in variables.items():
                yv = y[sel]
                if np.std(yv) < 1e-9:
                    continue
                yc = cell_center(yv, code[sel])
                tr, te = ~half[sel], half[sel]
                res[vname] = {}
                for sname, X in sources.items():
                    Xc = cell_center(X[sel], code[sel])
                    keep = Xc[tr].std(0) > 1e-9
                    Xc = Xc[:, keep]
                    r_all = r2_fit(Xc[tr], yc[tr], Xc[te], yc[te])
                    r1 = max((np.corrcoef(Xc[te][:, j], yc[te])[0, 1] ** 2 for j in range(Xc.shape[1])), default=0.0)
                    k, _ = greedy_k(Xc[tr], yc[tr], Xc[te], yc[te], 0.9 * r_all) if r_all > 0.05 else (None, [])
                    res[vname][sname] = {"n_components": int(Xc.shape[1]), "r2_all": r_all, "r2_best1": float(r1),
                                         "k90": k}
                    print(f"{gate:9s} {ph_name} {vname:15s} {sname:22s} n={Xc.shape[1]:3d}  R2 all {r_all:+.2f}  best single "
                          f"{r1:.2f}  components for 90%: {k}", flush=True)
            out[gate][ph_name] = res
        json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
