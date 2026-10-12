from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, StackDataset
import numpy as np
from tqdm import tqdm

from models.rnn import RNN
from models.transcoders import Transcoder
from circuit.copy_task.copy_find_features import FeatureActivationAnalyzer, convert_dict, checkpoint_fingerprint
from circuit.edge_att import compute_rl_input_baseline
torch.serialization.add_safe_globals([StackDataset])

PATTERNS = ["commonp", "common_p", "uncommonp", "uncommon_p"]
STARTS = ["high_first", "low_first"]
PAD = 5          # value of whole padded trials in trace windows
CORE_MARK = "*"  # prefix on the core trial's strings in sequence keys
# Trial bins relative to the core trial: 1, 2, 3 trials before, core, next.
TRIAL_BINS = ["1b", "2b", "3b", "a", "n"]


class RLFeatureActivationAnalyzer(FeatureActivationAnalyzer):
    """Analyze feature activations across RL trace windows for RNN transcoders.

    Sequence keys are tuples: (f"{start}_{pattern}", then 3 strings per trial:
    "prev. reward …", "took action …", "reached state …"). The core trial's
    strings start with CORE_MARK. Padded trials are not part of the key, so
    activation positions index the real trials (position = 3 * trial + phase).
    """

    def __init__(self,
                 rnn_model: nn.Module,
                 update_transcoder: nn.Module,
                 hidden_transcoder: nn.Module,
                 device: str = 'cuda',
                 reset_transcoder: Optional[nn.Module] = None,
                 rl=False):
        """
        Args:
            rnn_model: Trained RNN model
            update_transcoder: Trained update gate transcoder
            hidden_transcoder: Trained hidden context transcoder
            device: Device to run analysis on
        """
        super().__init__(rnn_model, update_transcoder, hidden_transcoder, device, reset_transcoder)
        # (pattern_idx, start_idx) for windows without metadata (old files)
        self.cur_type = None

    @staticmethod
    def strip_padding(window: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int]:
        """Drop leading padded trials from a single (trials, phases, ·) window."""
        n_pad = 0
        while n_pad < window["inputs"].size(0) and torch.all(window["inputs"][n_pad] == PAD):
            n_pad += 1
        stripped = {k: (v[n_pad:] if torch.is_tensor(v) and v.dim() >= 2 else v) for k, v in window.items()}
        return stripped, n_pad

    def window_key(self, window: Dict[str, torch.Tensor]) -> Tuple[str, ...]:
        """Key of one window (padded or already stripped), from its metadata.

        The label and core trial come from the window's pattern / start_high /
        core_trial columns when present. Older files have none: the label falls
        back to self.cur_type and the core to the second-last trial (wrong for
        windows without a following trial, which those files cannot mark).
        """
        window, _ = self.strip_padding(window)
        inputs, outputs = window["inputs"], window["outputs"]
        n_trials = inputs.size(0)
        if "core_trial" in window:
            core = int(window["core_trial"]) - (5 - n_trials)
            label = f"{STARTS[0 if bool(window['start_high']) else 1]}_{PATTERNS[int(window['pattern'])]}"
        else:
            core = max(n_trials - 2, 0)
            label = f"{STARTS[self.cur_type[1]]}_{PATTERNS[self.cur_type[0]]}"
        return (label,) + self._trial_strings(inputs, outputs, core)

    @staticmethod
    def _trial_strings(inputs: torch.Tensor, outputs: torch.Tensor, core: int) -> Tuple[str, ...]:
        strings = []
        for i in range(inputs.size(0)):
            mark = CORE_MARK if i == core else ""
            strings += [f"{mark}prev. reward {float(inputs[i, 0, -2]):g}",
                        f"{mark}took action {inputs[i, 2, 3:6].argmax().item()} w value {outputs[i, 1, -1].item()}",
                        f"{mark}reached state {inputs[i, 2, :3].argmax().item()}"]
        return tuple(strings)

    def convert_sequence_to_text(self,
                                 inputs: torch.Tensor,
                                 outputs: torch.Tensor) -> Tuple[str, ...]:
        """Key from inputs/outputs alone (no metadata): label from cur_type,
        core assumed to be the second-last trial. Prefer window_key."""
        return self.window_key({"inputs": inputs, "outputs": outputs})

    def sequence_key(self, batched: Dict[str, torch.Tensor], batch_idx: int):
        return self.window_key({k: v[batch_idx] for k, v in batched.items()})

    def analyze_batch_activations(self,
                                 batch: Dict[str, torch.Tensor]) -> None:
        """
        Analyze feature activations for a batch of windows with the same
        number of (unpadded) trials.
        """
        with torch.no_grad():
            for batch_idx in range(batch["inputs"].size(0)):
                self.all_sequences.add(self.sequence_key(batch, batch_idx))
            seq_len = batch["inputs"].size(1)*3
            for t in range(seq_len):
                valid_h_prev = batch["h_prevs"][:, t//3, t%3].to(self.device)
                valid_x_t = batch["inputs"][:, t//3, t%3].to(self.device)
                valid_r_t = batch["r_ts"][:, t//3, t%3].to(self.device)

                self.analyze_activations(
                                         batch,
                                         valid_h_prev,
                                         valid_x_t,
                                         valid_r_t,
                                         t)

    def analyze_all_sequences(self,
                            batch_size: int = 8192,
                            cache=[],
                            num_workers: int = 0) -> None:
        """Analyze every window in the given trace-window files.

        Windows are grouped by (pattern, start, number of real trials); the
        pattern and start come from each window's metadata, or for older files
        from the file's position in `cache` (4 patterns per start config).
        """
        print("Accumulating sequences...")
        grouped = defaultdict(lambda: defaultdict(list))   # (pattern, start, n_trials) -> key -> values
        for idx, path in enumerate(cache):
            data = torch.load(path, map_location=torch.device("cpu"))
            for window in data:
                assert window["inputs"].size(0) == 5
                window, n_pad = self.strip_padding(window)
                assert n_pad <= 3
                if "pattern" in window:
                    group = (int(window["pattern"]), 0 if bool(window["start_high"]) else 1)
                else:
                    group = (idx % 4, idx // 4)
                for key, value in window.items():
                    grouped[group + (5 - n_pad,)][key].append(value)

        for (pattern, start, n_trials) in sorted(grouped):
            self.cur_type = (pattern, start)
            columns = {k: torch.stack(v) for k, v in grouped[(pattern, start, n_trials)].items()}
            dataloader = DataLoader(StackDataset(**columns), batch_size, num_workers=num_workers)
            for batch in tqdm(dataloader, desc=f"{STARTS[start]}_{PATTERNS[pattern]}, {n_trials} trials"):
                self.analyze_batch_activations(batch)
        print("Feature activation analysis complete!")

    @staticmethod
    def core_trial_of(entry: Tuple[str, ...]) -> int:
        """Core trial index of a sequence key (second-last trial for keys
        without a core mark, from older caches)."""
        trials = (len(entry) - 1) // 3
        for trial in range(trials):
            if entry[1 + 3 * trial].startswith(CORE_MARK):
                return trial
        return max(trials - 2, 0)

    def get_feature_summary(self, transcoder_type: str, feature_idx: int) -> Dict:
        """
        Get summary statistics for a specific feature

        Args:
            transcoder_type: 'reset', 'update' or 'hidden'
            feature_idx: Index of the feature to analyze

        Returns:
            Dictionary with activation counts / magnitudes by start config,
            phase and trial (relative to the core trial: TRIAL_BINS), overall
            and per pattern.
        """
        if feature_idx not in self.feature_activations[transcoder_type]:
            return {"n_sequences": 0, "n_activations": 0}

        activations = self.feature_activations[transcoder_type][feature_idx]
        total_activations = sum(len(activations[entry]['positions']) for entry in activations)
        total_magnitudes = sum(sum(activations[entry]['magnitudes']) for entry in activations)
        total_sequences = len(activations)

        starts, phases, trials, mags, patterns = [], [], [], [], []
        for entry in activations:
            core = self.core_trial_of(entry)
            start = STARTS.index("_".join(entry[0].split("_")[:2]))
            pattern = PATTERNS.index("_".join(entry[0].split("_")[2:]))
            for position, magnitude in zip(activations[entry]['positions'], activations[entry]['magnitudes']):
                trial, phase = position // 3, position % 3
                offset = trial - core                     # -3 .. +1
                trial_bin = 3 + offset if offset >= 0 else -offset - 1
                assert 0 <= trial_bin <= 4, (entry, position)
                starts.append(start); phases.append(phase); trials.append(trial_bin)
                mags.append(magnitude); patterns.append(pattern)
        starts, phases, trials = np.array(starts, dtype=int), np.array(phases, dtype=int), np.array(trials, dtype=int)
        mags, patterns = np.array(mags, dtype=float), np.array(patterns, dtype=int)

        def dist(values, n, mask=None, weights=None):
            mask = np.ones(len(values), dtype=bool) if mask is None else mask
            return np.bincount(values[mask], weights=None if weights is None else weights[mask],
                               minlength=n).tolist()

        stat_dict = {
            "n_sequences": total_sequences,
            "n_activations": total_activations,
            "n_magnitudes": total_magnitudes,
            "avg_activations_per_sequence": total_activations / max(total_sequences, 1),
            "start_distribution": dist(starts, 2),
            "start_distribution_mag": dist(starts, 2, weights=mags),
            "phase_distribution": dist(phases, 3),
            "phase_distribution_mag": dist(phases, 3, weights=mags),
            "trial_distribution": dist(trials, 5),
            "trial_distribution_mag": dist(trials, 5, weights=mags),
            "magnitude_stats": {"mean": float(mags.mean()), "std": float(mags.std()),
                                "min": float(mags.min()), "max": float(mags.max())} if len(mags) else None,
        }
        for p, name in enumerate(PATTERNS):
            mask = patterns == p
            stat_dict[f"{name}_phases_distribution"] = dist(phases, 3, mask)
            stat_dict[f"{name}_phases_distribution_mag"] = dist(phases, 3, mask, mags)
            stat_dict[f"{name}_trials_distribution"] = dist(trials, 5, mask)
            stat_dict[f"{name}_trials_distribution_mag"] = dist(trials, 5, mask, mags)
        return stat_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_feats_hidden", type=int, required=True)
    parser.add_argument("--n_feats_update", type=int, required=True)
    parser.add_argument("--n_feats_reset", type=int)
    parser.add_argument("--update_transcoder_path", required=True)
    parser.add_argument("--reset_transcoder_path")
    parser.add_argument("--hidden_transcoder_path", required=True)
    parser.add_argument("--rnn_path", required=True)
    parser.add_argument("--cached_sequences", nargs="+", required=True,
                        help="Trace-window files (*_rl.pt). Files with window metadata can be "
                             "given in any order; older files must be the 4 patterns "
                             "(commonp, common_p, uncommonp, uncommon_p) for high_first, then low_first")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for run-specific activation caches")
    parser.add_argument("--run_name", default=None,
                        help="Optional human-readable cache prefix")
    parser.add_argument("--batch_size", type=int, default=4096)

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    rnn_model = RNN(input_size=8, hidden_size=48, out_size=4,
                    use_gru=True, num_layers=1, learn_init=True)
    update_transcoder = Transcoder(input_size=56, out_size=48,
                                   n_feats=args.n_feats_update)
    hidden_transcoder = Transcoder(input_size=56, out_size=48,
                                   n_feats=args.n_feats_hidden)
    reset_transcoder = (Transcoder(input_size=56, out_size=48, n_feats=args.n_feats_reset)
                        if args.reset_transcoder_path else None)

    rnn_model.load_state_dict(torch.load(args.rnn_path, map_location=device))
    update_transcoder.load_state_dict(torch.load(args.update_transcoder_path, map_location=device)["transcoder"])
    hidden_transcoder.load_state_dict(torch.load(args.hidden_transcoder_path, map_location=device)["transcoder"])
    if reset_transcoder:
        reset_transcoder.load_state_dict(torch.load(args.reset_transcoder_path, map_location=device)["transcoder"])

    analyzer = RLFeatureActivationAnalyzer(
        rnn_model=rnn_model,
        update_transcoder=update_transcoder,
        hidden_transcoder=hidden_transcoder,
        reset_transcoder=reset_transcoder,
        device=device
    )

    analyzer.analyze_all_sequences(batch_size=args.batch_size, cache=args.cached_sequences)

    print("Top Update Gate Features:")
    for feature_idx, count in analyzer.get_most_active_features('update', top_k=20):
        summary = analyzer.get_feature_summary('update', feature_idx)
        print(f"Feature {feature_idx}: {count} activations across {summary['n_sequences']} sequences")

    # save feature data (same layout and provenance metadata as copy_find_features)
    new_dict_features = convert_dict(analyzer.feature_activations, "positions", "magnitudes")
    new_dict_sequences = convert_dict(analyzer.sequence_activations, "features", "magnitudes")
    fingerprints = {
        "rnn": checkpoint_fingerprint(args.rnn_path),
        "update": checkpoint_fingerprint(args.update_transcoder_path),
        "hidden": checkpoint_fingerprint(args.hidden_transcoder_path),
        "reset": checkpoint_fingerprint(args.reset_transcoder_path) if reset_transcoder else None,
    }
    fingerprint_suffix = "_".join([
        fingerprints["rnn"][:10], fingerprints["update"][:10], fingerprints["hidden"][:10],
        fingerprints["reset"][:10] if fingerprints["reset"] else "noreset",
    ])
    run_name = args.run_name or f"h{args.n_feats_hidden}_u{args.n_feats_update}_{fingerprint_suffix}"
    os.makedirs(args.output_dir, exist_ok=True)
    reset_suffix = f"_r{args.n_feats_reset}" if reset_transcoder else ""
    cache_prefix = os.path.join(args.output_dir, f"{run_name}{reset_suffix}")
    with open(f"{cache_prefix}_features.p", "wb") as f:
        pickle.dump(new_dict_features, f)
    with open(f"{cache_prefix}_sequences.p", "wb") as f:
        pickle.dump(new_dict_sequences, f)
    with open(f"{cache_prefix}_metadata.p", "wb") as f:
        pickle.dump({
            "format_version": 2,
            "task": "rl",
            "checkpoint_paths": {
                "rnn": os.path.abspath(args.rnn_path),
                "update": os.path.abspath(args.update_transcoder_path),
                "hidden": os.path.abspath(args.hidden_transcoder_path),
                "reset": os.path.abspath(args.reset_transcoder_path) if reset_transcoder else None,
            },
            "checkpoint_fingerprints": fingerprints,
            "feature_counts": {"update": args.n_feats_update, "hidden": args.n_feats_hidden,
                               "reset": args.n_feats_reset if reset_transcoder else None},
            "all_sequences": list(analyzer.all_sequences),
            "cached_sentence_paths": [os.path.abspath(path) for path in args.cached_sequences],
            # Per-phase input reference for attribution graphs (circuit_viz),
            # fixed for this run rather than recomputed per viz session.
            "input_baseline": compute_rl_input_baseline(
                [torch.load(path, map_location="cpu") for path in args.cached_sequences]).tolist(),
        }, f)
    print(f"-- Wrote {cache_prefix}_features.p / _sequences.p / _metadata.p")
