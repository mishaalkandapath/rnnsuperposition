"""How the gates write, keep, erase and read one token's memory trace.

For each sequence (distinct tokens) and slot i, run the real GRU on the
original and on a copy with token i swapped (feedback changed to match). The
difference of hidden states, dh_t, is the memory trace of which token is in
slot i: everything token-independent cancels. Each step splits exactly
(bars are means of the two runs):

  dh_t = (1 - zb_t) * dh_{t-1}            kept     (update gate closed)
       + zb_t * dn_t                      written  (update gate open, new candidate)
       + dz_t * (nb_t - hb_{t-1})         gate     (the token changed the gate itself)

Per step (means over sequences):
  norm        |dh_t|
  kept_frac   <(1 - zb) dh_{t-1}, dh_{t-1}> / |dh_{t-1}|^2: fraction of the trace kept;
              1 - kept_frac is erased by the update gate.
  terms       share of dh_t from kept / written / gate (projections on dh_t, sum to 1)
  readout     the trace as seen by the output layer: logit(orig token) - logit(swapped token)
              read off dh_t; nonzero only once the token is "say-able"
  reset_read  the candidate preactivation change dp_t split into the stored trace read
              through the reset gate, W_hn (rb * dh_{t-1}), the reset gate's own change,
              W_hn (dr * hb_{t-1}), and the input (projections on dp_t)

Gate features: the update gate value is an exact sum of update-transcoder
features + bias + error (likewise reset), so each term above splits by gate
feature (averaged over the two runs):
  erase_by    update pieces' share of the erased fraction  <piece * dh_{t-1}, dh_{t-1}> / |dh_{t-1}|^2
  write_by    update pieces' share of the written term     <piece * dn_t, dh_t> / |dh_t|^2
  read_by     reset pieces' share of the stored-trace read <W_hn (piece * dh_{t-1}), dp_t> / |dp_t|^2

    python -m circuit.copy_task.gate_mechanism --rnn RNN.ckpt --traces T3.pt T4.pt \\
        --transcoders reset=R.ckpt update=U.ckpt --feature_results features/<run>_autointerp --out gates.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import StackDataset

from circuit.copy_task.copy_circuits import load_labels, step_label
from circuit.copy_task.explained_fidelity import load_transcoder
from training.splice_eval import build_task_rnn

torch.serialization.add_safe_globals([StackDataset])


@torch.no_grad()
def record(rnn, x):
    """(h_prev, r, z, n, h) of the real GRU, each (N, T, H)."""
    _, _, r, z, n, h_prev = rnn.layers[0](x, record_gates=True)
    h = (1 - z) * h_prev + z * n
    return h_prev, r, z, n, h


@torch.no_grad()
def pieces(tc, h_prev, x):
    """Feature activations (N, T, F) of a gate transcoder on [h_{t-1}, x_t]."""
    return tc.act(tc.input_to_features(torch.cat([h_prev, x], -1)))


@torch.no_grad()
def slot_trace(rnn, upd, rst, x, L, i, new, unit_means=None):
    """Per-step quantities of the trace of slot i (see module docstring); each (N,) or (N, F).

    unit_means: (mean z, mean r) per GRU unit. If given, every gate term is
    also split over raw gate units: the constant mean gate (one component, the
    analogue of the transcoder bias) plus each unit's deviation from it.
    """
    N, T, _ = x.shape
    x2 = x.clone()
    for t in [i] + ([L + 1 + i] if i + 1 < L else []):
        x2[:, t] = 0
        x2[torch.arange(N), t, new] = 1
    runs = [record(rnn, xx) for xx in (x, x2)]
    (hp1, r1, z1, n1, h1), (hp2, r2, z2, n2, h2) = runs
    dh, dhp, dz, dn, dr = h1 - h2, hp1 - hp2, z1 - z2, n1 - n2, r1 - r2
    zb, nb, hpb, rb = (z1 + z2) / 2, (n1 + n2) / 2, (hp1 + hp2) / 2, (r1 + r2) / 2

    gru, readout = rnn.layers[0], rnn.layers[-1]
    W_hn = gru.hidden_to_new.weight                                      # (H, H)
    dp = gru.hidden_to_new(r1 * hp1) - gru.hidden_to_new(r2 * hp2) + gru.input_to_new(x - x2)
    p_mem = (rb * dhp) @ W_hn.T
    p_gate = (dr * hpb) @ W_hn.T
    p_in = gru.input_to_new(x - x2)

    # Gate pieces averaged over the two runs: features (F), bias, error.
    fu = (pieces(upd, hp1, x) + pieces(upd, hp2, x2)) / 2
    fr = (pieces(rst, hp1, x) + pieces(rst, hp2, x2)) / 2
    eu = (z1 - upd(torch.cat([hp1, x], -1))[0] + z2 - upd(torch.cat([hp2, x2], -1))[0]) / 2
    er = (r1 - rst(torch.cat([hp1, x], -1))[0] + r2 - rst(torch.cat([hp2, x2], -1))[0]) / 2
    Mu, bu = upd.features_to_outputs.weight, upd.features_to_outputs.bias
    Mr, br = rst.features_to_outputs.weight, rst.features_to_outputs.bias

    orig = x[:, i, :-1].argmax(-1)
    w_read = readout.weight[orig] - readout.weight[new]                  # (N, H)

    def proj(a, b):                                                      # <a, b> / |b|^2, batched
        return (a * b).sum(-1) / (b * b).sum(-1).clamp_min(1e-12)

    out = []
    for t in range(i, T):
        d_prev, d = dhp[:, t], dh[:, t]
        kept = (1 - zb[:, t]) * d_prev
        written = zb[:, t] * dn[:, t]
        gate = dz[:, t] * (nb[:, t] - hpb[:, t])
        w_erase = d_prev ** 2 / (d_prev ** 2).sum(-1, keepdim=True).clamp_min(1e-12)   # weights for erase shares
        w_write = dn[:, t] * d / (d * d).sum(-1, keepdim=True).clamp_min(1e-12)
        v_read = d_prev * (dp[:, t] @ W_hn) / (dp[:, t] ** 2).sum(-1, keepdim=True).clamp_min(1e-12)
        step = {
            "norm": d.norm(dim=-1), "kept_frac": proj(kept, d_prev) if t > i else torch.zeros(N),
            "terms": {"kept": proj(kept, d), "written": proj(written, d), "gate": proj(gate, d)},
            "readout": (w_read * d).sum(-1),
            "reset_read": {"stored_trace": proj(p_mem[:, t], dp[:, t]), "reset_change": proj(p_gate[:, t], dp[:, t]),
                           "input": proj(p_in[:, t], dp[:, t])},
            "erase_by": {"features": fu[:, t] * (w_erase @ Mu), "bias": w_erase @ bu, "error": (eu[:, t] * w_erase).sum(-1)},
            "write_by": {"features": fu[:, t] * (w_write @ Mu), "bias": w_write @ bu, "error": (eu[:, t] * w_write).sum(-1)},
            "read_by": {"features": fr[:, t] * (v_read @ Mr), "bias": v_read @ br, "error": (er[:, t] * v_read).sum(-1)},
        }
        if unit_means is not None:
            mz, mr = unit_means
            for name, gate, mean, w in (("erase_units", zb[:, t], mz, w_erase), ("write_units", zb[:, t], mz, w_write),
                                        ("read_units", rb[:, t], mr, v_read)):
                step[name] = {"const": w @ mean, "units": (gate - mean) * w}
        if t == i:                                                       # nothing stored yet to erase or read
            step["erase_by"] = step["read_by"] = None
            step.pop("erase_units", None)
            step.pop("read_units", None)
        out.append(step)
    return out


def k_cover(shares, frac=0.8):
    """Fewest components whose |mean share| covers frac of the total |mean share|."""
    a = np.sort(np.abs(np.asarray(shares)))[::-1]
    total = a.sum()
    return int(np.searchsorted(np.cumsum(a), frac * total) + 1) if total > 0 else 0


def concentration(steps):
    """Per gate term and step: components covering 80% for features vs units,
    with the constant part (transcoder bias / mean gate) as one component, and
    without it (modulators only; the transcoder error counts as a modulator)."""
    rows = []
    for s in steps:
        for term, units in (("erase_by", "erase_units"), ("write_by", "write_units"), ("read_by", "read_units")):
            if s.get(term) is None or units not in s:
                continue
            f, u = s[term], s[units]
            feat_mod = np.append(f["features"].mean(0).numpy(), float(f["error"].mean()))
            unit_mod = u["units"].mean(0).numpy()
            feat_const, unit_const = float(f["bias"].mean()), float(u["const"].mean())
            rows.append({"term": term.split("_")[0],
                         "features_k80": k_cover(np.append(feat_mod, feat_const)),
                         "units_k80": k_cover(np.append(unit_mod, unit_const)),
                         "features_k80_modulators": k_cover(feat_mod), "units_k80_modulators": k_cover(unit_mod),
                         "features_const_share": feat_const, "units_const_share": unit_const,
                         "features_modulator_mass": float(np.abs(feat_mod).sum()),
                         "units_modulator_mass": float(np.abs(unit_mod).sum())})
    return rows


@torch.no_grad()
def ablated_run(rnn, x, gate, tc=None, feats=(), feat_means=None, units=(), unit_mean=None):
    """Real GRU with one gate ("update" / "reset") ablated at every step.

    feats: transcoder features replaced by their mean activation, i.e. the
    gate minus (a_f - mean a_f) * d_f; units: gate units set to their mean.
    Returns output-step logits (N, L, V) and the mean |change| of the gate vector.
    """
    gru, readout = rnn.layers[0], rnn.layers[-1]
    N, T, _ = x.shape
    h = torch.zeros(N, gru.hidden_size)
    feats, units = list(feats), list(units)
    out, moved = [], 0.0
    for t in range(T):
        xt = x[:, t]
        r = torch.sigmoid(gru.input_to_reset(xt) + gru.hidden_to_reset(h))
        z = torch.sigmoid(gru.input_to_update(xt) + gru.hidden_to_update(h))
        g = z if gate == "update" else r
        g2 = g.clone()
        if feats:
            a = tc.act(tc.input_to_features(torch.cat([h, xt], -1)))[:, feats]
            g2 = (g2 - (a - feat_means[feats]) @ tc.features_to_outputs.weight[:, feats].T).clamp(0, 1)
        if units:
            g2[:, units] = unit_mean[units]
        moved += float((g2 - g).norm(dim=-1).mean()) / T
        z, r = (g2, r) if gate == "update" else (z, g2)
        n = torch.tanh(gru.input_to_new(xt) + gru.hidden_to_new(r * h))
        h = (1 - z) * h + z * n
        if t >= T - T // 2:
            out.append(readout(h))
    return torch.stack(out, 1), moved


@torch.no_grad()
def ablations(rnn, x, L, upd, rst, rank, ks, n_random, seed):
    """Top-k gate features vs top-k gate units (ranked by their share of the gate
    term: erase for update, read for reset), plus random-k controls of each.
    Per setting: accuracy and change in the correct logit per output position."""
    orig = x[:, :L, :-1].argmax(-1)
    centred = lambda o: o - o.mean(-1, keepdim=True)
    base = centred(ablated_run(rnn, x, "update")[0]).gather(2, orig[:, :, None])[..., 0]
    h_prev, r, z, _, _ = record(rnn, x)
    hx = torch.cat([h_prev, x], -1)
    g = torch.Generator().manual_seed(seed)
    results = []
    for gate, tc in (("update", upd), ("reset", rst)):
        acts = tc.act(tc.input_to_features(hx))
        feat_means, unit_mean = acts.mean((0, 1)), (z if gate == "update" else r).mean((0, 1))
        alive = torch.nonzero((acts > 0).flatten(0, 1).any(0)).flatten()
        for k in ks:
            choices = [("top features", "feats", rank[gate]["features"][:k]), ("top units", "units", rank[gate]["units"][:k])]
            for s in range(n_random):
                choices += [(f"random features", "feats", alive[torch.randperm(len(alive), generator=g)[:k]].tolist()),
                            (f"random units", "units", torch.randperm(z.shape[-1], generator=g)[:k].tolist())]
            for name, kind, ids in choices:
                logits, moved = ablated_run(rnn, x, gate, tc, feats=ids if kind == "feats" else (),
                                            feat_means=feat_means, units=ids if kind == "units" else (),
                                            unit_mean=unit_mean)
                lc = centred(logits).gather(2, orig[:, :, None])[..., 0]
                results.append({"gate": gate, "k": k, "selection": name, "ids": ids, "gate_change": moved,
                                "accuracy": (logits.argmax(-1) == orig).float().mean(0).tolist(),
                                "dlogit": (lc - base).mean(0).tolist()})
    # average the random draws
    merged = {}
    for r_ in results:
        key = (r_["gate"], r_["k"], r_["selection"])
        m = merged.setdefault(key, {**r_, "n": 0, "ids": [], "accuracy": np.zeros(L), "dlogit": np.zeros(L), "gate_change": 0.0})
        m["n"] += 1
        m["ids"].append(r_["ids"])
        m["accuracy"] += np.array(r_["accuracy"])
        m["dlogit"] += np.array(r_["dlogit"])
        m["gate_change"] += r_["gate_change"]
    return [{**m, "accuracy": (m["accuracy"] / m["n"]).tolist(), "dlogit": (m["dlogit"] / m["n"]).tolist(),
             "gate_change": m["gate_change"] / m["n"], "ids": m["ids"][0] if m["n"] == 1 else m["ids"]}
            for m in merged.values()]


def summarise(steps, L, i, labels, top):
    rows = []
    for k, s in enumerate(steps):
        t = i + k
        row = {"step": step_label(t, L), "norm": float(s["norm"].mean()), "kept_frac": float(s["kept_frac"].mean()),
               "terms": {key: float(v.mean()) for key, v in s["terms"].items()},
               "readout": float(s["readout"].mean()),
               "reset_read": {key: float(v.mean()) for key, v in s["reset_read"].items()}}
        for name, gate in (("erase_by", "update"), ("write_by", "update"), ("read_by", "reset")):
            if s[name] is None:
                continue
            f = s[name]["features"].mean(0)
            order = torch.argsort(f.abs(), descending=True)[:top].tolist()
            row[name] = {"bias": float(s[name]["bias"].mean()), "error": float(s[name]["error"].mean()),
                         "features_total": float(f.sum()),
                         "top": [{"feature": j, "share": float(f[j]),
                                  **({"label": labels[(gate, j)][0]} if (gate, j) in labels else {})} for j in order]}
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rnn", required=True)
    parser.add_argument("--traces", nargs="+", required=True, help="one trace file per length")
    parser.add_argument("--transcoders", nargs=2, required=True, metavar="GATE=PATH", help="reset=... update=...")
    parser.add_argument("--feature_results", default=None)
    parser.add_argument("--n", type=int, default=2000, help="sequences per length")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8], help="components ablated")
    parser.add_argument("--n_random", type=int, default=3, help="random draws per random control")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = dict(item.split("=", 1) for item in args.transcoders)
    rnn = build_task_rnn("copy", 31, 128)
    rnn.load_state_dict(torch.load(args.rnn, map_location="cpu"))
    upd, rst = load_transcoder(paths["update"]), load_transcoder(paths["reset"])
    labels = load_labels(args.feature_results)
    rng = np.random.default_rng(args.seed)

    results = []
    for path in args.traces:
        data = torch.load(path, map_location="cpu", mmap=True).datasets
        L = data["outputs"].shape[1]
        orig = data["inputs"][:, :L].argmax(-1)
        distinct = np.flatnonzero((orig[:, :, None] == orig[:, None, :]).sum((1, 2)).numpy() == L)
        rows = np.sort(rng.choice(distinct, size=min(args.n, len(distinct)), replace=False))
        x = data["inputs"][torch.from_numpy(rows)].float()
        V = x.shape[-1] - 1
        _, r_all, z_all, _, _ = record(rnn, x)
        unit_means = (z_all.mean((0, 1)), r_all.mean((0, 1)))
        score = {gate: {"features": torch.zeros(tc.n_feats), "units": torch.zeros(rnn.hidden_size)}
                 for gate, tc in (("update", upd), ("reset", rst))}
        slots = []
        for i in range(L):
            # a replacement token not already in the sequence
            present = torch.zeros(len(x), V, dtype=torch.bool).scatter_(1, orig[torch.from_numpy(rows)], True)
            new = torch.multinomial((~present).float(), 1, generator=torch.Generator().manual_seed(args.seed + i))[:, 0]
            steps = slot_trace(rnn, upd, rst, x, L, i, new, unit_means)
            slots.append({"slot": i, "steps": summarise(steps, L, i, labels, args.top),
                          "concentration": concentration(steps)})
            for s in steps:                                              # ranking for the ablations
                for gate, term, units in (("update", "erase_by", "erase_units"), ("reset", "read_by", "read_units")):
                    if s.get(term) is not None:
                        score[gate]["features"] += s[term]["features"].mean(0).abs()
                        score[gate]["units"] += s[units]["units"].mean(0).abs()
            print(f"L={L} slot {i}: " + " | ".join(
                f"{s['step']} |dh| {s['norm']:.2f} kept {s['kept_frac']:.2f} read {s['readout']:+.1f}"
                for s in slots[-1]["steps"]))
        conc = [c for s in slots for c in s["concentration"]]
        for term in ("erase", "write", "read"):
            rows_t = [c for c in conc if c["term"] == term]
            print(f"L={L} {term}: components covering 80% (median over steps) - features "
                  f"{np.median([c['features_k80'] for c in rows_t]):.0f} vs units {np.median([c['units_k80'] for c in rows_t]):.0f}; "
                  f"modulators only: features {np.median([c['features_k80_modulators'] for c in rows_t]):.0f} vs units "
                  f"{np.median([c['units_k80_modulators'] for c in rows_t]):.0f}")
        rank = {gate: {kind: torch.argsort(v, descending=True).tolist() for kind, v in s_.items()}
                for gate, s_ in score.items()}
        abl = ablations(rnn, x, L, upd, rst, rank, args.ks, args.n_random, args.seed)
        for a in abl:
            print(f"L={L} ablate {a['gate']:>6} {a['selection']:>16} k={a['k']}: acc per output "
                  + " ".join(f"{v:.2f}" for v in a["accuracy"]) + f" | gate change {a['gate_change']:.3f}")
        results.append({"L": L, "n_sequences": len(rows), "slots": slots, "ablations": abl,
                        "ranking": {gate: {kind: ids[:20] for kind, ids in r_.items()} for gate, r_ in rank.items()}})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "lengths": results}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
