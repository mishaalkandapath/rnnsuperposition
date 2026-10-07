"""Train comparable reset, update, and candidate transcoders from one dataset set.

This intentionally runs one transcoder per process/configuration: dictionaries are
independent, but every target is recorded from the same frozen backbone rollout.
Use a job-array wrapper to distribute the printed commands if desired.
"""
import argparse
import itertools
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, StackDataset

from training.splice_eval import load_splice_traces
from training.train_transcoder import create_and_train_transcoders


PROFILES = {
    # Existing copy update/candidate settings, plus two wider-coverage settings.
    "update_like": {"l_sparsity": 0.025, "l_penalty": 0.0, "c_sparsity": 1.0},
    "candidate_like": {"l_sparsity": 0.004, "l_penalty": 3e-6, "c_sparsity": 3.0},
    "sparse": {"l_sparsity": 0.01, "l_penalty": 3e-6, "c_sparsity": 2.0},
    "dense": {"l_sparsity": 0.001, "l_penalty": 0.0, "c_sparsity": 1.0},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reset-datasets", nargs="+", required=True)
    p.add_argument("--update-datasets", nargs="+", required=True)
    p.add_argument("--candidate-datasets", nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--hidden-size", type=int, required=True)
    p.add_argument("--input-size", type=int, required=True)
    p.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    p.add_argument("--profiles", choices=PROFILES, nargs="+", default=list(PROFILES))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--init-modes", choices=["random", "paired_dpi"], nargs="+", default=["random"])
    p.add_argument("--dpi-scale", type=float, default=0.4)
    p.add_argument("--dpi-calibration-samples", type=int, default=8192)
    p.add_argument("--dpi-seed", type=int, default=None,
                   help="DPI sample seed; defaults to each run's --seed")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=270)
    p.add_argument("--batch-size", type=int, default=32784)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--schedule", type=int, default=1)
    p.add_argument("--schedule-offset", type=int, default=220)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rnn-path", default=None,
                   help="Copy-task GRU checkpoint; enables the splice eval for each run's target")
    p.add_argument("--splice-sequence-paths", nargs="+", default=None,
                   help="Trace sequence files (*_seqN.pt) for the splice eval")
    p.add_argument("--splice-samples", type=float, default=0.05,
                   help="Held-out sequences to splice: <=1 is a fraction, >1 a count")
    p.add_argument("--normalize", action="store_true",
                   help="Train on centred, scalar-scaled inputs/targets; checkpoints are folded back to raw units")
    p.add_argument("--legacy-sparsity", action="store_true",
                   help="Use the older λ²·max-batch-L0 sparsity loss the PROFILES were tuned with")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_dataset(paths):
    datasets = [torch.load(path, map_location="cpu") for path in paths]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def main():
    args = parse_args()
    if args.rnn_path and not args.splice_sequence_paths:
        raise SystemExit("--rnn-path requires --splice-sequence-paths")
    target_paths = {
        "reset": args.reset_datasets,
        "update": args.update_datasets,
        "candidate": args.candidate_datasets,
    }
    configs = list(itertools.product(target_paths, args.widths, args.profiles, args.seeds, args.init_modes))
    print(f"Planned runs: {len(configs)}")
    if args.dry_run:
        for target, width, profile, seed, init_mode in configs:
            print(f"{target}: width={width}, profile={profile}, seed={seed}, init={init_mode}")
        return

    datasets = {target: load_dataset(paths) for target, paths in target_paths.items()}
    # Load splice traces once for the whole sweep instead of once per run.
    splice_traces = load_splice_traces(args.splice_sequence_paths) if args.rnn_path else None
    results_path = args.output_dir / "sweep_results.jsonl"
    for target, width, profile_name, seed, init_mode in configs:
        torch.manual_seed(seed)
        profile = PROFILES[profile_name]
        run_dir = args.output_dir / target / f"w{width}_{profile_name}_{init_mode}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=False)
        cfg = {
            "lr": args.lr, "l_schedule": args.schedule,
            "l_sched_offset": args.schedule_offset, "w_det": False,
            "scale_pen": False, "legacy_sparsity": args.legacy_sparsity,
            "normalize": args.normalize,
            "ctd_from": None, "n_epochs": args.epochs,
            "n_feats": width, "batch_size": args.batch_size,
            "init_mode": init_mode, "dpi_scale": args.dpi_scale,
            "dpi_calibration_samples": args.dpi_calibration_samples,
            "dpi_seed": args.dpi_seed if args.dpi_seed is not None else seed, **profile,
        }
        with (run_dir / "sweep_metadata.json").open("w") as f:
            json.dump({"target": target, "seed": seed, "init_mode": init_mode, "input_size": args.input_size,
                       "hidden_size": args.hidden_size, "config": cfg}, f, indent=2)
        trainer, _ = create_and_train_transcoders(
            datasets[target], cfg, hidden_size=args.hidden_size,
            input_size=args.input_size, n_feats=width, device=args.device,
            n_epochs=args.epochs, batch_size=args.batch_size, save_path=str(run_dir),
            num_workers=args.num_workers, split_seed=seed, init_mode=init_mode,
            dpi_scale=args.dpi_scale, dpi_calibration_samples=args.dpi_calibration_samples,
            dpi_seed=args.dpi_seed if args.dpi_seed is not None else seed,
            rnn_path=args.rnn_path, splice_target=target,
            splice_samples=args.splice_samples, splice_traces=splice_traces,
        )
        result = {"target": target, "width": width, "profile": profile_name,
                  "seed": seed, "init_mode": init_mode, **trainer.final_metrics}
        with results_path.open("a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
