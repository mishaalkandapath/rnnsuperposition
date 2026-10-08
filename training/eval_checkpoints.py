"""Evaluate saved transcoder checkpoints over training and plot the metrics.

For every run directory, every periodic checkpoint (e{N}.ckpt) and the final
checkpoint are evaluated on:
  - validation FVU (and L0 / dead-feature fraction), on the same held-out
    rows the run was validated on (same split rule and seed as training);
  - optionally, the splice eval (training/splice_eval.py): the copy-task GRU
    with this gate replaced by the checkpoint, compared with the original
    model on the same inputs: KL, top-1 agreement, task accuracy.

Run directories from training/sweep_transcoders.py carry their target and
seed in sweep_metadata.json. For other runs pass --target and --split_seed
(the training --split_seed, which defaulted to --seed).

Example:
    python -m training.eval_checkpoints \
        --run_dirs runs/update/w64_* --dataset_paths data/update.pt \
        --rnn_path data/copy.ckpt --splice_sequence_paths data/seqs_*.pt \
        --out_dir eval/update
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import ConcatDataset, StackDataset

from datasets.utils import create_transcoder_dataloaders
from models.rnn import RNN
from models.transcoders import Transcoder
from training.splice_eval import SPLICE_TARGETS, SpliceEvaluator, load_splice_traces

torch.serialization.add_safe_globals([StackDataset])

METRICS = ["fvu", "l0", "dead_fraction",
           "splice_kl", "splice_top1_agreement", "splice_task_acc", "original_task_acc"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run_dirs", nargs="+", required=True, type=Path,
                   help="Training output dirs containing e{N}.ckpt / final_model.ckpt")
    p.add_argument("--dataset_paths", nargs="+", required=True,
                   help="The gate dataset(s) the runs were trained on (same order as training)")
    p.add_argument("--target", choices=SPLICE_TARGETS, default=None,
                   help="Gate the transcoders replace; read from sweep_metadata.json if absent")
    p.add_argument("--split_seed", type=int, default=None,
                   help="Training split seed; read from sweep_metadata.json / hyperparams.json if absent")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_val_rows", type=int, default=None,
                   help="Evaluate FVU on at most this many validation rows (same rows for every checkpoint)")
    p.add_argument("--rnn_path", default=None, help="Copy-task GRU checkpoint; enables the splice eval")
    p.add_argument("--splice_sequence_paths", nargs="+", default=None,
                   help="Trace sequence files (datasets/transcoder_copy_datasets.py) for the splice eval")
    p.add_argument("--splice_samples", type=float, default=0.05,
                   help="Sequences for the splice eval: <= 1 is a fraction, > 1 a count")
    p.add_argument("--splice_seed", type=int, default=0,
                   help="Seed for sampling splice sequences. Runs whose datasets carry no sequence "
                        "ids all share one sample, so their splice metrics are comparable")
    p.add_argument("--skip_final", action="store_true", help="Ignore final_model.ckpt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def run_settings(run_dir: Path, args) -> Dict:
    """Target and split seed of a run, from its metadata unless given on the CLI."""
    meta, hyper = {}, {}
    if (run_dir / "sweep_metadata.json").exists():
        meta = json.loads((run_dir / "sweep_metadata.json").read_text())
    if (run_dir / "hyperparams.json").exists():
        hyper = json.loads((run_dir / "hyperparams.json").read_text())
    target = args.target or meta.get("target")
    if target is None:
        raise ValueError(f"{run_dir}: no sweep_metadata.json target; pass --target")
    split_seed = args.split_seed
    if split_seed is None:
        # Sweeps split with their seed; direct runs record dpi_seed, which
        # defaulted to the split seed.
        split_seed = meta.get("seed", hyper.get("split_seed", hyper.get("dpi_seed")))
        if split_seed is None:
            raise ValueError(f"{run_dir}: cannot infer the split seed; pass --split_seed")
        print(f"-- {run_dir.name}: split seed {split_seed} (from metadata)")
    n_epochs = hyper.get("n_epochs", meta.get("config", {}).get("n_epochs"))
    previous = hyper.get("previous_epochs", 0) or 0
    return {"target": target, "split_seed": int(split_seed),
            "final_epoch": None if n_epochs is None else n_epochs + previous}


def checkpoints(run_dir: Path, final_epoch: Optional[int], skip_final: bool) -> List[Dict]:
    found = []
    for path in run_dir.glob("e*.ckpt"):
        match = re.fullmatch(r"e(\d+)\.ckpt", path.name)
        if match:
            found.append({"epoch": int(match.group(1)), "path": path, "label": path.stem})
    found.sort(key=lambda c: c["epoch"])
    final = run_dir / "final_model.ckpt"
    if final.exists() and not skip_final:
        ckpt_epoch = torch.load(final, map_location="cpu", weights_only=True).get("completed_epochs")
        epoch = ckpt_epoch or final_epoch or (found[-1]["epoch"] + 1 if found else 0)
        if not any(c["epoch"] == epoch for c in found):
            found.append({"epoch": epoch, "path": final, "label": "final"})
    if not found:
        raise ValueError(f"{run_dir}: no e{{N}}.ckpt or final_model.ckpt")
    return found


def load_transcoder(path: Path, device) -> Transcoder:
    """Rebuild a transcoder from a checkpoint, inferring its sizes from the weights.

    Checkpoints store raw-unit weights (normalization is folded in before
    saving), and legacy scalar thresholds are broadcast on load.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)["transcoder"]
    n_feats, input_dim = state["input_to_features.weight"].shape
    out_size = state["features_to_outputs.weight"].shape[0]
    transcoder = Transcoder(input_size=input_dim, out_size=out_size, n_feats=n_feats,
                            bias="input_to_features.bias" in state)
    transcoder.load_state_dict(state)
    return transcoder.to(device).eval()


@torch.no_grad()
def reconstruction_metrics(transcoder: Transcoder, val_batches: List, device) -> Dict[str, float]:
    """FVU = SSE / total variance (float64 running sums), mean L0, dead-feature fraction."""
    sse = target_sum = target_sq_sum = 0.0
    n_rows = l0_sum = 0
    ever_active = torch.zeros(transcoder.n_feats, dtype=torch.bool, device=device)
    for inputs, targets in val_batches:
        inputs, targets = inputs.to(device), targets.to(device).double()
        predictions, features, _ = transcoder(inputs)
        sse += ((predictions.double() - targets) ** 2).sum().item()
        target_sum = target_sum + targets.sum(0)
        target_sq_sum += (targets ** 2).sum().item()
        active = features > 0
        l0_sum += active.sum().item()
        ever_active |= active.any(0)
        n_rows += targets.shape[0]
    total_variance = target_sq_sum - (target_sum ** 2).sum().item() / max(n_rows, 1)
    return {"fvu": sse / max(total_variance, 1e-12), "l0": l0_sum / max(n_rows, 1),
            "dead_fraction": 1.0 - ever_active.float().mean().item()}


def validation_batches(dataset_paths, split_seed, args) -> (List, Optional[torch.Tensor]):
    """Held-out rows of a run, materialised once so every checkpoint sees the same rows."""
    datasets = [torch.load(path, map_location="cpu") for path in dataset_paths]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    _, val_loader, val_sequence_ids = create_transcoder_dataloaders(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        split_seed=split_seed, return_val_sequence_ids=True)
    batches, n_rows = [], 0
    for batch in val_loader:
        inputs, targets = batch["input"], batch["output"]
        if args.max_val_rows is not None:
            keep = args.max_val_rows - n_rows
            if keep <= 0:
                break
            inputs, targets = inputs[:keep], targets[:keep]
        batches.append((inputs, targets))
        n_rows += inputs.shape[0]
    print(f"-- Validation: {n_rows} rows (split seed {split_seed})")
    return batches, val_sequence_ids


def plot(rows: List[Dict], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted({r["run"] for r in rows})
    present = [m for m in METRICS if any(r.get(m) is not None for r in rows)]
    labels = {"fvu": "Validation FVU", "l0": "L0 (active features / row)",
              "dead_fraction": "Dead-feature fraction", "splice_kl": "Splice KL(original || spliced)",
              "splice_top1_agreement": "Top-1 agreement with original",
              "splice_task_acc": "Task accuracy", "original_task_acc": "Original task accuracy"}

    def draw(ax, metric):
        for run in runs:
            pts = sorted((r["epoch"], r[metric]) for r in rows if r["run"] == run and r.get(metric) is not None)
            if pts:
                ax.plot(*zip(*pts), marker="o", markersize=3, label=run)
        if metric == "splice_task_acc":
            ref = [r["original_task_acc"] for r in rows if r.get("original_task_acc") is not None]
            if ref:
                ax.axhline(ref[0], color="black", linestyle="--", linewidth=1, label="original model")
        if metric in ("fvu", "splice_kl"):
            ax.set_yscale("log")
        ax.set_title(labels[metric])
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)

    shown = [m for m in present if m != "original_task_acc"]
    for metric in shown:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        draw(ax, metric)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png", dpi=150)
        plt.close(fig)

    cols = min(3, len(shown))
    nrows = math.ceil(len(shown) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(5.5 * cols, 4 * nrows), squeeze=False)
    for ax, metric in zip(axes.flat, shown):
        draw(ax, metric)
    for ax in list(axes.flat)[len(shown):]:
        ax.axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=min(4, len(legend_labels)), fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "all_metrics.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.rnn_path and not args.splice_sequence_paths:
        raise ValueError("--rnn_path needs --splice_sequence_paths")

    splice_traces = load_splice_traces(args.splice_sequence_paths) if args.rnn_path else None
    val_cache, splice_cache, rows = {}, {}, []
    for run_dir in args.run_dirs:
        settings = run_settings(run_dir, args)
        ckpts = checkpoints(run_dir, settings["final_epoch"], args.skip_final)
        print(f"== {run_dir}: target={settings['target']}, {len(ckpts)} checkpoints")

        seed = settings["split_seed"]
        if seed not in val_cache:
            val_cache[seed] = validation_batches(args.dataset_paths, seed, args)
        val_batches, val_sequence_ids = val_cache[seed]

        evaluator = None
        if args.rnn_path:
            # With held-out sequence ids the sample must come from this run's
            # held-out set; without them every run shares one sample.
            key = (settings["target"], seed if val_sequence_ids is not None else None)
            if key not in splice_cache:
                probe = load_transcoder(ckpts[0]["path"], "cpu")
                hidden_size = probe.out_size
                input_size = probe.input_size - hidden_size
                # Copy-task GRU: input = vocab + delimiter, output = vocab.
                rnn = RNN(input_size=input_size, hidden_size=hidden_size, out_size=input_size - 1,
                          out_act=lambda x: x, use_gru=True)
                rnn.load_state_dict(torch.load(args.rnn_path, map_location="cpu"))
                splice_cache[key] = SpliceEvaluator(
                    rnn, settings["target"], splice_traces, args.splice_samples,
                    val_sequence_ids=val_sequence_ids, seed=args.splice_seed, device=device)
            evaluator = splice_cache[key]

        for ckpt in ckpts:
            transcoder = load_transcoder(ckpt["path"], device)
            row = {"run": str(run_dir.name), "run_dir": str(run_dir), "target": settings["target"],
                   "epoch": ckpt["epoch"], "checkpoint": ckpt["label"]}
            row.update(reconstruction_metrics(transcoder, val_batches, device))
            if evaluator is not None:
                row.update(evaluator.evaluate(transcoder))
            rows.append(row)
            shown = ", ".join(f"{m}={row[m]:.4g}" for m in METRICS if m in row)
            print(f"   {ckpt['label']:>6} (epoch {ckpt['epoch']}): {shown}")

    fields = ["run", "run_dir", "target", "epoch", "checkpoint"] + [m for m in METRICS if any(m in r for r in rows)]
    with (args.out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    plot(rows, args.out_dir)
    print(f"-- Wrote {args.out_dir}/metrics.csv, metrics.json and plots")


if __name__ == "__main__":
    main()
