"""Discrete versions of recorded magnitude short forms (units or features alike).
Per component, test R^2 beyond timing of:
  short     the recorded short form (refit here)
  discrete  graded factors (> 8 distinct values) dropped; discrete factors kept
  fine      one value per cell: every combination of all factors, graded ones cut in 4 quantile bins
            (ramps: zero + 3 bins over the nonzero part)
  L3 / L4   the fine cells merged into 3 / 4 levels (optimal 1-D weighted split of the cell means on dev)
Cell values are mean residuals beyond timing on dev.

    python -m circuit.rl_task.discrete_levels rl_features/rl_magnitude_units rl_features/rl_fidelity/discrete_levels_units.json
    python -m circuit.rl_task.discrete_levels rl_features/rl_magnitude rl_features/rl_fidelity/discrete_levels_features.json
"""
import glob, json, os, sys
import numpy as np
from circuit.rl_task.autointerp_magnitude import Mag

STORE = "rl_features/rl_chosen/chosen_rl_v1_r128_store"


def factors(term):
    out, depth, cur = [], 0, ""
    for ch in term:
        depth += ch == "("; depth -= ch == ")"
        if ch == "*" and depth == 0:
            out.append(cur.strip()); cur = ""
        else:
            cur += ch
    return out + [cur.strip()]


def ksplit(means, w, k):
    o = np.argsort(means); m, w = means[o], w[o]; n = len(m)
    if n <= k:
        lab = np.arange(n)
    else:
        cw, cm, cm2 = np.concatenate([[0], np.cumsum(w)]), np.concatenate([[0], np.cumsum(w * m)]), np.concatenate([[0], np.cumsum(w * m * m)])
        def sse(a, b):
            ww = cw[b] - cw[a]
            return 0.0 if ww <= 0 else (cm2[b] - cm2[a]) - (cm[b] - cm[a]) ** 2 / ww
        D = np.full((k + 1, n + 1), np.inf); D[0, 0] = 0; B = np.zeros((k + 1, n + 1), int)
        for j in range(1, k + 1):
            for i in range(j, n + 1):
                best = min(((D[j - 1, s] + sse(s, i), s) for s in range(j - 1, i)), key=lambda x: x[0])
                D[j, i], B[j, i] = best
        lab = np.zeros(n, int); i = n
        for j in range(k, 0, -1):
            s = B[j, i]; lab[s:i] = j - 1; i = s
    out = np.empty(n, int); out[o] = lab
    return out


def run(path):
    rec = json.load(open(path))
    gate, f = rec["gate"], rec["feature"]
    st = rec.get("short_terms")
    if not st:
        return None
    terms = [t.strip() for t in (st if isinstance(st, str) else ";;".join(st)).split(";;") if t.strip()]
    m = Mag(STORE, gate, f); ns = m.namespace(); dev, te = m.rows["dev"], m.rows["test"]
    ev = lambda e: np.broadcast_to(np.asarray(eval(e, {"__builtins__": {}}, ns), dtype=np.float64), (m.ip.n_rows,))
    r = m.act - m.timing
    def r2(pred):
        return float(1 - ((r[te] - pred[te]) ** 2).sum() / (r[te] ** 2).sum())
    def fit(cols):
        X = np.column_stack(cols + [np.ones(m.ip.n_rows)])
        w = np.linalg.lstsq(X[dev], r[dev], rcond=None)[0]
        return r2(X @ w)
    out = {"gate": gate, "id": f, "verdict": rec.get("final_verdict"), "recorded": rec["test_scores"]["r2_beyond_timing"],
           "short": fit([ev(t) for t in terms])}
    keys, disc_cols, graded = [], [], []
    for t in terms:
        fs = [(e, ev(e)) for e in factors(t)]
        d = [v for e, v in fs if len(np.unique(v[dev])) <= 8]
        g = [(e, v) for e, v in fs if len(np.unique(v[dev])) > 8]
        dprod = np.prod(d, 0) if d else np.ones(m.ip.n_rows)
        if d: disc_cols.append(dprod)
        for e, v in fs:
            if len(np.unique(v[dev])) <= 8:
                keys.append(np.unique(v, return_inverse=True)[1].ravel())
            else:
                graded.append(e)
                on = dev[(dprod[dev] != 0) & (v[dev] != 0)]
                if len(on) < 50:
                    on = dev
                ramp = (v[dev] == 0).mean() > 0.2
                edges = np.unique(np.quantile(v[on], np.linspace(0, 1, 4 if ramp else 5)[1:-1]))
                if ramp: edges = np.concatenate([[1e-9], edges])
                keys.append(np.searchsorted(edges, v, side="right"))
    out["graded"] = sorted(set(graded))
    out["discrete"] = fit(disc_cols) if disc_cols else 0.0
    K = np.stack(keys, 1)
    _, cell = np.unique(K, axis=0, return_inverse=True); cell = cell.ravel()
    nc = cell.max() + 1
    cnt = np.bincount(cell[dev], minlength=nc); s = np.bincount(cell[dev], weights=r[dev], minlength=nc)
    mean = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
    out["n_cells"] = int((cnt > 0).sum()); out["fine"] = r2(mean[cell])
    live = np.nonzero(cnt > 0)[0]
    for k in (3, 4):
        lab = ksplit(mean[live], cnt[live].astype(float), k)
        L = np.zeros(nc, int); L[live] = lab
        lv = np.array([r[dev][L[cell[dev]] == j].mean() if (L[cell[dev]] == j).any() else 0 for j in range(k)])
        out[f"L{k}"] = r2(lv[L[cell]])
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    res = []
    for p in sorted(glob.glob(os.path.join(src, "*.json"))):
        try:
            o = run(p)
        except Exception as e:
            print("ERR", p, e, flush=True); continue
        if o:
            res.append(o)
            print(f"{o['gate']:15s} {o['id']:4d} short {o['short']:.2f} discrete {o['discrete']:.2f} fine {o['fine']:.2f} "
                  f"({o['n_cells']}) L3 {o['L3']:.2f} L4 {o['L4']:.2f} {o['graded']}", flush=True)
            json.dump(res, open(dst, "w"), indent=1)
