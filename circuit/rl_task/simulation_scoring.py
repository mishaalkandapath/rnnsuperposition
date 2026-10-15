"""Words-only explanations scored by blind simulation (Anthropic / OpenAI style), for features and GRU units alike.

Following Towards Monosemanticity's automated interpretability: an explainer sees a component's activations on
example windows and writes a short explanation in WORDS (no code); a separate simulator sees only that explanation
and held-out windows WITHOUT activations, and predicts the activation at every step. The score is the correlation
between predicted and true activations over held-out top-activating and random windows together, so the whole
activation range counts. The same pipeline and scale are used for transcoder features and the GRU's own units:
activations are raw (no thresholds), each rescaled to 0-10 over its own range (1st-99th percentile of the dev split;
features: 0 = not active).

Also reported: the correlation beyond timing, after removing each (phase, episode decile) cell's mean from both
the true and the predicted values within the scored windows: whether the explanation predicts anything beyond
"high at GO" or "higher late in the episode".

Commands (dev split = even episodes, for the explainer; test split = odd episodes, for the simulator):
    python -m circuit.rl_task.simulation_scoring explain_sample STORE GATE ID          # windows WITH 0-10 values
    python -m circuit.rl_task.simulation_scoring simulate_sample STORE GATE ID         # held-out windows, NO values
    python -m circuit.rl_task.simulation_scoring score STORE GATE ID PREDICTIONS_JSON --explanation TEXT --out DIR
    python -m circuit.rl_task.simulation_scoring report DIR
GATE: reset / update / hidden (features) or unit_reset / unit_update / unit_candidate (GRU units' raw values).
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

from circuit.rl_task.autointerp import Interp
from circuit.rl_task.autointerp_magnitude import UNIT_GATES, UNIT_VALUES

N_TOP, N_RANDOM, N_LOW = 12, 12, 6          # explainer sample
N_SIM_TOP, N_SIM_RANDOM = 20, 20            # simulator sample (held out)


class Component:
    def __init__(self, store, gate, comp):
        self.ip, self.gate, self.comp = Interp(store), gate, comp
        if gate in UNIT_GATES:
            act = np.asarray(np.load(os.path.join(UNIT_VALUES, f"{UNIT_GATES[gate]}.npy"), mmap_mode="r")[:, comp],
                             dtype=np.float64)
        else:
            act = self.ip.activations(gate, comp).astype(np.float64)
        dev = ~self.ip.test
        lo = 0.0 if gate not in UNIT_GATES else float(np.quantile(act[dev], 0.01))
        hi = float(np.quantile(act[dev][act[dev] > lo] if gate not in UNIT_GATES else act[dev], 0.99)) \
            if (act[dev] > lo).any() else lo + 1.0
        self.lo, self.hi = lo, hi                       # the 0-10 scale: 0 = lo, 10 = hi
        self.scaled = np.clip(10 * (act - lo) / max(hi - lo, 1e-9), 0, 10)
        c = self.ip.cols
        self.row_start = np.asarray(c["row_start"])
        win_ep = np.asarray(c["episode"])[self.row_start[:-1]]
        self.win_test = (win_ep % 2).astype(bool)
        mx = np.maximum.reduceat(self.scaled, self.row_start[:-1])
        mn = np.minimum.reduceat(self.scaled, self.row_start[:-1])
        self.win_max, self.win_min = mx, mn

    def pick(self, split, n_top, n_random, n_low, seed):
        rng = np.random.default_rng(seed)
        pool = np.nonzero(self.win_test if split == "test" else ~self.win_test)[0]
        top = pool[np.argsort(-self.win_max[pool])[:n_top * 20]]
        top = rng.choice(top, size=min(n_top, len(top)), replace=False)
        rest = np.setdiff1d(pool, top)
        rand = rng.choice(rest, size=min(n_random, len(rest)), replace=False)
        low = []
        if n_low:
            rest = np.setdiff1d(rest, rand)
            low = rest[np.argsort(self.win_min[rest])[:n_low * 20]]
            low = rng.choice(low, size=min(n_low, len(low)), replace=False)
        return [("top", int(w)) for w in top] + [("random", int(w)) for w in rand] + [("low", int(w)) for w in low]

    def render(self, w, values=True):
        """One window, one line per trial: trial in episode, then D1 (reward seen) | GO (choice) | D2 (state reached,
        common / rare), each step with its 0-10 value when values=True, else a step id [s0], [s1], ..."""
        c = self.ip.cols
        lo, hi = int(self.row_start[w]), int(self.row_start[w + 1])
        rew = lambda v: {1: "rewarded", 0: "unrewarded", -1: "penalty"}.get(int(v), "?")
        lines = []
        for r0 in range(lo, hi, 3):
            t = int(c["ep_trial"][r0])
            cells = [f"D1 (previous trial {rew(c['prev_reward'][r0])})",
                     f"GO (chose a{int(c['action'][r0])})" if c["action"][r0] > 0 else "GO (no valid choice)",
                     f"D2 (reached S{int(c['state'][r0])}, "
                     f"{'common' if c['common'][r0] == 1 else 'rare' if c['common'][r0] == 0 else '?'})"]
            out = []
            for k, cell in enumerate(cells):
                r = r0 + k
                out.append(f"{cell} = {self.scaled[r]:.0f}" if values else f"{cell} [s{r - lo}]")
            lines.append(f"  trial {t:3d}: " + " | ".join(out))
        return "\n".join(lines)


def explain_sample(m, seed=0):
    picks = m.pick("dev", N_TOP, N_RANDOM, N_LOW, seed)
    kind = "GRU unit (raw gate value)" if m.gate in UNIT_GATES else "transcoder feature"
    out = [f"# {m.gate}/{m.comp} ({kind}): activation per step, rescaled to 0-10 over its own range. "
           f"Windows: {N_TOP} with its highest values, {N_RANDOM} random, {N_LOW} with its lowest values (dev split).",
           "Each trial: D1 = fixation, the previous trial's reward is shown; GO = the agent chooses a1 or a2; D2 = the "
           "state reached is shown (a1 -> S1 and a2 -> S2 are common, the others rare). trial = index in the 100-trial "
           "episode. The good state (which pays 0.9 vs 0.1) is hidden and reverses a few times per episode."]
    for kind_, w in picks:
        out.append(f"\n## window ({kind_})\n" + m.render(w))
    return "\n".join(out)


def simulate_sample(m, seed=1):
    picks = m.pick("test", N_SIM_TOP, N_SIM_RANDOM, 0, seed)
    out = [f"# Held-out windows for {m.gate}/{m.comp}, WITHOUT activations. Predict the activation (0-10) at every "
           f"step marked [sK]. Window ids are w0, w1, ...; order is shuffled."]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(picks))
    for j, i in enumerate(order):
        out.append(f"\n## w{j}\n" + m.render(picks[i][1], values=False))
    return "\n".join(out), [picks[i] for i in order]


def score(m, preds, seed=1):
    """preds: {"w0": [v per step], ...} for the simulate_sample windows."""
    _, picks = simulate_sample(m, seed)
    c = m.ip.cols
    y, p, cell, kind = [], [], [], []
    for j, (k, w) in enumerate(picks):
        lo, hi = int(m.row_start[w]), int(m.row_start[w + 1])
        pr = preds.get(f"w{j}")
        if pr is None or len(pr) != hi - lo:
            raise ValueError(f"w{j}: need {hi - lo} predictions, got {None if pr is None else len(pr)}")
        y.append(m.scaled[lo:hi]); p.append(np.asarray(pr, dtype=np.float64))
        cell.append(np.asarray(c["phase"][lo:hi]) * 10 + np.clip(np.asarray(c["ep_trial"][lo:hi]) // 10, 0, 9))
        kind.append(np.full(hi - lo, k == "top"))
    y, p, cell, kind = map(np.concatenate, (y, p, cell, kind))

    def corr(a, b):
        return float(pearsonr(a, b)[0]) if a.std() > 0 and b.std() > 0 else 0.0

    def beyond(a):
        out = a.copy()
        for v in np.unique(cell):
            out[cell == v] -= a[cell == v].mean()
        return out
    return {"correlation": corr(y, p), "spearman": float(spearmanr(y, p)[0]) if p.std() > 0 else 0.0,
            "correlation_beyond_timing": corr(beyond(y), beyond(p)),
            "correlation_random_windows": corr(y[~kind], p[~kind]),
            "n_steps": int(len(y)),
            # how much of the true activation in these windows timing (phase x episode decile) alone accounts for
            "true_share_explained_by_timing": float(1 - (beyond(y) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9))}


# ----------------------------------------------------------------------------- shared evaluation set

N_SHARED, SHARED_SEED, CHUNK = 100, 7, 10


def shared_windows(ip, n=N_SHARED, seed=SHARED_SEED):
    """The same random held-out (odd-episode) windows for every component."""
    c = ip.cols
    rs = np.asarray(c["row_start"])
    test = np.nonzero(np.asarray(c["episode"])[rs[:-1]] % 2 == 1)[0]
    return [int(w) for w in np.sort(np.random.default_rng(seed).choice(test, size=n, replace=False))]


def shared_sample(m, chunk):
    wins = shared_windows(m.ip)
    n_chunks = (len(wins) + CHUNK - 1) // CHUNK
    sel = wins[chunk * CHUNK:(chunk + 1) * CHUNK]
    out = [f"# Shared held-out windows, chunk {chunk} of {n_chunks} (chunks 0..{n_chunks - 1}), WITHOUT activations. "
           f"Predict the activation (0-10) of {m.gate}/{m.comp} at every step marked [sK]. Window ids are "
           f"w{chunk * CHUNK} ... w{chunk * CHUNK + len(sel) - 1}."]
    for j, w in enumerate(sel):
        out.append(f"\n## w{chunk * CHUNK + j}\n" + m.render(w, values=False))
    return "\n".join(out)


def record_chunk(m, chunk, preds, out_dir):
    wins = shared_windows(m.ip)
    sel = wins[chunk * CHUNK:(chunk + 1) * CHUNK]
    for j, w in enumerate(sel):
        k = f"w{chunk * CHUNK + j}"
        n = int(m.row_start[w + 1] - m.row_start[w])
        if k not in preds or len(preds[k]) != n:
            raise ValueError(f"{k}: need {n} predictions, got {None if k not in preds else len(preds[k])}")
    d = os.path.join(out_dir, "chunks", f"{m.gate}_{m.comp}")
    os.makedirs(d, exist_ok=True)
    json.dump(preds, open(os.path.join(d, f"{chunk}.json"), "w"))
    return f"recorded chunk {chunk} ({len(sel)} windows)"


def score_shared(m, out_dir, explanation):
    """Combine the chunk predictions; correlation over every step of the shared windows (and beyond timing)."""
    wins = shared_windows(m.ip)
    d = os.path.join(out_dir, "chunks", f"{m.gate}_{m.comp}")
    preds = {}
    for k in range((len(wins) + CHUNK - 1) // CHUNK):
        p = os.path.join(d, f"{k}.json")
        if not os.path.exists(p):
            raise ValueError(f"chunk {k} not recorded")
        preds.update(json.load(open(p)))
    c = m.ip.cols
    y, p, cell = [], [], []
    for j, w in enumerate(wins):
        lo, hi = int(m.row_start[w]), int(m.row_start[w + 1])
        y.append(m.scaled[lo:hi]); p.append(np.asarray(preds[f"w{j}"], dtype=np.float64))
        cell.append(np.asarray(c["phase"][lo:hi]) * 10 + np.clip(np.asarray(c["ep_trial"][lo:hi]) // 10, 0, 9))
    y, p, cell = map(np.concatenate, (y, p, cell))

    def beyond(a):
        out = a.copy()
        for v in np.unique(cell):
            out[cell == v] -= a[cell == v].mean()
        return out
    corr = lambda a, b: float(pearsonr(a, b)[0]) if a.std() > 0 and b.std() > 0 else 0.0
    s = {"correlation": corr(y, p), "correlation_beyond_timing": corr(beyond(y), beyond(p)),
         "variance_explained": float(1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)),
         "active_share": float((y > 0.5).mean()), "n_steps": int(len(y)),
         "true_share_explained_by_timing": float(1 - (beyond(y) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9))}
    rec = {"gate": m.gate, "id": m.comp, "explanation": explanation, "scores": s,
           "windows": wins, "predictions": [float(v) for v in p]}
    json.dump(rec, open(os.path.join(out_dir, f"{m.gate}_{m.comp}.json"), "w"))
    return s


def report(out_dir):
    rows = [json.load(open(os.path.join(out_dir, n))) for n in sorted(os.listdir(out_dir)) if n.endswith(".json")]
    lines = [f"{len(rows)} simulation scores"]
    for gate in sorted({r["gate"] for r in rows}):
        x = [r["scores"] for r in rows if r["gate"] == gate]
        lines.append(f"{gate}: n={len(x)}, median correlation {np.median([s['correlation'] for s in x]):.2f}, "
                     f"beyond timing {np.median([s['correlation_beyond_timing'] for s in x]):.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("explain_sample", "simulate_sample", "score"):
        s = sub.add_parser(name)
        s.add_argument("store"); s.add_argument("gate"); s.add_argument("comp", type=int)
        if name == "score":
            s.add_argument("predictions", help='JSON {"w0": [values per step], ...}')
            s.add_argument("--explanation", required=True)
            s.add_argument("--out", required=True)
    for name in ("shared_sample", "record_chunk", "score_shared"):
        s = sub.add_parser(name)
        s.add_argument("store"); s.add_argument("gate"); s.add_argument("comp", type=int)
        if name in ("shared_sample", "record_chunk"):
            s.add_argument("--chunk", type=int, required=True)
        if name == "record_chunk":
            s.add_argument("predictions", help='JSON {"w<i>": [values per step], ...} for the chunk')
        if name in ("record_chunk", "score_shared"):
            s.add_argument("--out", required=True)
        if name == "score_shared":
            s.add_argument("--explanation", required=True)
    e = sub.add_parser("show_explanation", help="print only the recorded words-only explanation of a component")
    e.add_argument("gate"); e.add_argument("comp", type=int); e.add_argument("--source", required=True)
    r = sub.add_parser("report"); r.add_argument("dir")
    args = ap.parse_args()
    if args.cmd == "report":
        print(report(args.dir))
        return
    if args.cmd == "show_explanation":
        print(json.load(open(os.path.join(args.source, f"{args.gate}_{args.comp}.json")))["explanation"])
        return
    m = Component(args.store, args.gate, args.comp)
    if args.cmd == "shared_sample":
        print(shared_sample(m, args.chunk))
        return
    if args.cmd == "record_chunk":
        print(record_chunk(m, args.chunk, json.loads(args.predictions), args.out))
        return
    if args.cmd == "score_shared":
        print(json.dumps(score_shared(m, args.out, args.explanation), indent=1))
        return
    if args.cmd == "explain_sample":
        print(explain_sample(m))
    elif args.cmd == "simulate_sample":
        print(simulate_sample(m)[0])
    else:
        preds = json.loads(open(args.predictions).read() if os.path.exists(args.predictions) else args.predictions)
        s = score(m, preds)
        os.makedirs(args.out, exist_ok=True)
        json.dump({"gate": args.gate, "id": args.comp, "explanation": args.explanation, "scores": s},
                  open(os.path.join(args.out, f"{args.gate}_{args.comp}.json"), "w"), indent=1)
        print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
