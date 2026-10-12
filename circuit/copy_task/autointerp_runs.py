"""Helpers around the agent workflows in circuit/copy_task/autointerp_workflows/.

items  : build the `items` argument (feature list) for a workflow run from a
         store, skipping features that already have results; optionally split
         into batches that stay under the workflow agent cap.
sparse : record features too sparse for an agent (fewer than --min_sequences
         sequences, or never firing) without one.
merge  : after a run, fold the skeptic verdicts from its journal into the
         per-feature result JSONs (interpreter results are already on disk,
         written by the agents via `circuit.copy_task.autointerp record`).

    python -m circuit.copy_task.autointerp_runs items  STORE RESULTS [--min_sequences 20] [--batch 600] [--gates ...]
    python -m circuit.copy_task.autointerp_runs sparse STORE RESULTS --widths reset=512 update=1024 hidden=512
    python -m circuit.copy_task.autointerp_runs merge  RESULTS JOURNAL [JOURNAL ...]

The journal is <transcript dir>/journal.jsonl, printed when a workflow launches.
"""
import argparse
import json
import os
from collections import Counter

GATE_INITIALS = {"reset": "r", "update": "u", "hidden": "h"}


def _counts(store):
    return json.load(open(os.path.join(store, "store.json")))["counts"]


def items(store, results, min_sequences=20, batch=None, gates=None):
    done = {n[:-5] for n in os.listdir(results) if n.endswith(".json")} if os.path.isdir(results) else set()
    out = []
    for gate, counts in _counts(store).items():
        if gates and gate not in gates:
            continue
        for feat, (n_seq, n_act) in sorted(counts.items(), key=lambda kv: int(kv[0])):
            if n_seq >= min_sequences and f"{gate}_{feat}" not in done:
                out.append([GATE_INITIALS.get(gate, gate), int(feat), n_seq, n_act])
    return [out[i:i + batch] for i in range(0, len(out), batch)] if batch else [out]


def sparse(store, results, widths, min_sequences=20):
    os.makedirs(results, exist_ok=True)
    counts, n = _counts(store), 0
    for gate, width in widths.items():
        for feat in range(width):
            n_seq, n_act = counts.get(gate, {}).get(str(feat), [0, 0])
            if n_seq >= min_sequences:
                continue
            path = os.path.join(results, f"{gate}_{feat}.json")
            rec = json.load(open(path)) if os.path.exists(path) else {}
            verdict = "never_fires" if n_seq == 0 else "too_sparse"
            rec.update(gate=gate, feature=feat, n_sequences=n_seq, n_activations=n_act, final_verdict=verdict)
            rec.setdefault("label", f"{verdict.replace('_', ' ')} ({n_seq} sequences)")
            json.dump(rec, open(path, "w"), indent=2)
            n += 1
    return n


def merge(results, journal):
    labels, interps, skeptics = {}, {}, {}
    for line in open(journal):
        d = json.loads(line)
        if d["type"] == "started":
            labels[d["key"]] = d["label"]
        elif d["type"] == "result" and d.get("result"):
            lab, res = labels.get(d["key"], ""), d["result"]
            if lab.startswith("skeptic "):
                g, f = lab.split()[1].rsplit("/", 1)
                skeptics[(g, int(f))] = res
            elif isinstance(res, dict) and "gate" in res:
                interps[(res["gate"], res["feature"])] = res
    missing, verdicts, sk = [], Counter(), Counter()
    for key, i in interps.items():
        path = os.path.join(results, f"{key[0]}_{key[1]}.json")
        if os.path.exists(path):
            rec = json.load(open(path))
        else:      # agent returned without running `record`: keep what it returned
            missing.append(key)
            rec = {k: i.get(k) for k in ("label", "predicate", "verdict", "confidence", "explanation")}
            rec.update(gate=key[0], feature=key[1], recorded_by="merge (agent did not run record)")
        rec["final_verdict"] = i["verdict"]
        rec.pop("final_label", None)
        s = skeptics.get(key)
        if s:
            rec["skeptic"] = s
            rec["final_verdict"] = s["revised_verdict"]
            if s.get("revised_label"):
                rec["final_label"] = s["revised_label"]
            sk["upheld" if s["upheld"] else "refuted"] += 1
        json.dump(rec, open(path, "w"), indent=2)
        verdicts[rec["final_verdict"]] += 1
    print(f"{journal}: {len(interps)} interpreted, skeptic {dict(sk)}, final {dict(verdicts)}; "
          f"agents that did not record: {len(missing)} {missing[:10]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("items"); i.add_argument("store"); i.add_argument("results")
    i.add_argument("--min_sequences", type=int, default=20); i.add_argument("--batch", type=int, default=None)
    i.add_argument("--gates", nargs="*")
    s = sub.add_parser("sparse"); s.add_argument("store"); s.add_argument("results")
    s.add_argument("--widths", nargs="+", required=True, help="gate=width, e.g. reset=512")
    s.add_argument("--min_sequences", type=int, default=20)
    m = sub.add_parser("merge"); m.add_argument("results"); m.add_argument("journals", nargs="+")
    args = ap.parse_args()
    if args.cmd == "items":
        for b in items(args.store, args.results, args.min_sequences, args.batch, args.gates):
            print(json.dumps(b, separators=(",", ":")))
    elif args.cmd == "sparse":
        widths = {w.split("=")[0]: int(w.split("=")[1]) for w in args.widths}
        print(f"recorded {sparse(args.store, args.results, widths, args.min_sequences)} sparse / never-firing features")
    else:
        for j in args.journals:
            merge(args.results, j)


if __name__ == "__main__":
    main()
