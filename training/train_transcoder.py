import copy
import math
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, StackDataset, ConcatDataset
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.rnn import RNN
from models.transcoders import (Transcoder, compute_normalization, fold_normalization_,
                                initialize_paired_dpi, normalize_inputs, normalize_targets,
                                set_transcoder_weights, unfold_normalization_)
from datasets.utils import create_transcoder_dataloaders, ConsolidatedStackDataset
from training.splice_eval import SPLICE_TARGETS, SpliceEvaluator, load_splice_traces
from training.train_utils import SignalManager, normalize_batch
torch.serialization.add_safe_globals([StackDataset])

# Training rows sampled to estimate the --normalize means and scales.
NORMALIZATION_SAMPLES = 65536


def sample_paired_dpi_calibration(dataset, n_samples: int, dpi_seed: int = None):
    """Sample paired training rows without materializing the whole dataset."""
    if len(dataset) == 0:
        raise ValueError("Cannot sample DPI calibration data from an empty dataset")
    generator = None if dpi_seed is None else torch.Generator().manual_seed(dpi_seed)
    indices = torch.randint(len(dataset), (n_samples,), generator=generator)
    examples = [dataset[int(index)] for index in indices]
    try:
        inputs = torch.stack([example["input"] for example in examples])
        targets = torch.stack([example["output"] for example in examples])
    except KeyError as exc:
        raise ValueError("Paired DPI requires dataset examples with 'input' and 'output' keys") from exc
    return inputs, targets

class TranscoderLoss(nn.Module):
    """
    Custom loss function for transcoder training:
    L(x,y) = ||y - ŷ(x)||²₂ + λ_S * Σ tanh(c * |f_i(x)| * ||W_d,i||₂) + L_P(x)
    where L_P(x) = λ_P * Σ ReLU(exp(t) - f_i(x)) * ||W_d,i||₂

    legacy_sparsity reproduces the older variant
    λ_S² * max_b(Σ_i tanh(...)) * Σ tanh(...), which scales the penalty by the
    soft L0 of the densest sample in the batch.
    """

    def __init__(self, lambda_sparsity=1e-3, lambda_penalty=1e-4,
                 c_sparsity=1.0, sparse_sched=0, sparse_sched_off=1, w_detach=False, scale_pen_distance=False,
                 legacy_sparsity=False):
        super().__init__()
        self.legacy_sparsity = legacy_sparsity
        self.lambda_sparsity = lambda_sparsity
        self.lambda_penalty = lambda_penalty
        self.c_sparsity = c_sparsity
        self.mse_loss = nn.MSELoss()

        # options for traiing variants
        self.total_steps = 0
        self.steps = 0
        self.w_detach = w_detach
        self.off = sparse_sched_off
        self.eval = False
        self.scale_pen_distance = scale_pen_distance


        self.sparse_scheduler = self.set_lambda_sparse_schedule(sparse_sched)

    def set_lambda_sparse_schedule(self, typ):
        match typ:
            case 1: # linear rise + cut
                return lambda: min(self.steps/self.off, 1)
            case 2:
                # smoother linear rise + cut
                return lambda: 1/(1+math.exp(-10*((self.steps/self.total_steps)-0.5)))
            case 3:
                # cosine annealing sparsity loss
                return lambda: math.sin(0.5*math.pi*(self.steps % self.off)/self.off) if self.steps < 0.75*self.total_steps else 1
            case 4: return lambda: 1
            case 5: 
                return lambda: (self.steps/self.total_steps) * math.sin(0.5*math.pi*(self.steps % self.off)/self.off) if self.steps < 0.75*self.total_steps else 1
            case 6:
                return lambda: 1
            case 7:
                return lambda: 1 + min(self.steps/self.off, 1)
            case _:
                return lambda: (self.steps/self.total_steps)

        
    def forward(self, predictions, targets, features, 
                decoder_weights, threshold):
        """
        Args:
            predictions: Model predictions ŷ(x)
            targets: True targets y
            features: Feature activations f(x) from encoder
            decoder_weights: Weight matrix W_d from features_to_outputs layer
            threshold: JumpReLU threshold parameter (exp(t))

            W_d is of shape out_vec x n_feats
        """
        batch_size = targets.size(0)
        # Validation uses the same scheduled coefficient so train and val
        # totals are comparable at every point in training.
        sparsity_coeff = self.lambda_sparsity * self.sparse_scheduler()
        # Reconstruction loss: ||y - ŷ(x)||²₂, summed over output dims and
        # averaged over the batch (as in the paper), so λ_S does not silently
        # scale with the hidden size.
        reconstruction_loss = ((predictions - targets) ** 2).sum(dim=-1).mean()
        with torch.no_grad():
            # eps: with --normalize, centred targets can have near-zero norm.
            normalized_reconstruction_loss = self.mse_loss(
                predictions / torch.norm(predictions, dim=-1, keepdim=True).clamp_min(1e-8),
                targets / torch.norm(targets, dim=-1, keepdim=True).clamp_min(1e-8))
        
        # Sparsity loss: λ_S * Σ tanh(c * |f_i(x)|||W_d,i||₂)
        decoder_norms = torch.norm(decoder_weights, dim=0)  # ||W_d,i||₂ for each feature
        feature_magnitudes = torch.abs(features)  # |f_i(x)|
        
        decoder_norms = torch.clamp(decoder_norms, min=1e-8)
        decoder_norms = decoder_norms if not self.w_detach else decoder_norms.detach()
        
        normalized_features = feature_magnitudes * decoder_norms.unsqueeze(0)
        sparsity_terms = torch.tanh(self.c_sparsity * normalized_features)
        if self.legacy_sparsity:
            # Not in the paper; kept only to reproduce checkpoints trained with it.
            max_multiplier = torch.max(torch.sum(sparsity_terms, dim=-1))
            sparsity_loss = max_multiplier * (sparsity_coeff**2) * torch.sum(sparsity_terms)/batch_size
        else:
            sparsity_loss = sparsity_coeff * torch.sum(sparsity_terms)/batch_size
        
        # Penalty loss: L_P(x) = λ_P * Σ ReLU(exp(t) - f_i(x)) * ||W_d,i||₂
        act_distance = torch.exp(threshold) - features if not self.scale_pen_distance else (torch.exp(threshold) - features)/torch.exp(threshold)
        penalty_terms = torch.relu(act_distance) * decoder_norms.unsqueeze(0)
        penalty_loss = self.lambda_penalty * torch.sum(penalty_terms)/batch_size
        
        total_loss = reconstruction_loss + sparsity_loss + penalty_loss
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            "norm_recon_loss": normalized_reconstruction_loss,
            'sparsity_loss': sparsity_loss,
            'penalty_loss': penalty_loss
        }

class TranscoderTrainer:
    """Trainer class for transcoder models"""
    
    def __init__(self, 
                 transcoder: nn.Module,
                 optimizer: optim.Optimizer,
                 device: str = 'cuda',
                 loss_fn: TranscoderLoss=TranscoderLoss(10, 3e-6, 4),
                 splice_evaluator=None,
                 normalization=None):

        self.transcoder = transcoder.to(device)
        self.device = device
        self.splice_evaluator = splice_evaluator
        # With normalization the transcoder trains in normalised units; every
        # saved checkpoint and the splice eval use folded raw-unit weights.
        self.normalization = ({k: v.to(device) for k, v in normalization.items()}
                              if normalization is not None else None)
        
        self.loss_fn = loss_fn
        
        self.optimizer = optimizer
        self.completed_epochs = 0

        self.train_history = {
            'total': [], 'reconstruction': [], 'sparsity': [], 'penalty': [], "norm_recon": []
        }
        self.val_history = {
            'total': [], 'reconstruction': [], 'sparsity': [], 'penalty': [], "norm_recon": []
        }

    def _prepare_batch(self, batch):
        inputs = batch['input'].to(self.device)
        targets = batch['output'].to(self.device)
        if self.normalization is not None:
            inputs = normalize_inputs(inputs, self.normalization)
            targets = normalize_targets(targets, self.normalization)
        return inputs, targets

    def raw_transcoder(self) -> nn.Module:
        """The transcoder in raw units (a folded copy when normalising)."""
        if self.normalization is None:
            return self.transcoder
        return fold_normalization_(copy.deepcopy(self.transcoder), self.normalization)

    def checkpoint(self) -> Dict:
        """Resumable checkpoint. "transcoder" always holds raw-unit weights so
        every consumer can load it as a plain Transcoder; "normalization" lets
        --ctd_from unfold back to the units the optimizer state lives in."""
        return {
            "transcoder": self.raw_transcoder().state_dict(),
            "optim": self.optimizer.state_dict(),
            "normalization": ({k: v.cpu() for k, v in self.normalization.items()}
                              if self.normalization is not None else None),
            "completed_epochs": self.completed_epochs,
        }

    def train_epoch(self, train_loader: DataLoader, run=None, epoch: int = None) -> Dict[str, float]:
        """Train for one epoch"""
        self.transcoder.train()
        feature_activation_densities = torch.zeros((self.transcoder.n_feats)).to(self.device)
        self.loss_fn.eval = False
        epoch_losses = {
            'total': 0, 'reconstruction': 0, 'sparsity': 0, 'penalty': 0, "norm_recon": 0
        }
        
        n_batches = len(train_loader)
        
        description = f"Epoch {epoch + 1}" if epoch is not None else "Training"
        pbar = tqdm(train_loader, total=n_batches, leave=False, desc=description)
        for batch in pbar:
            inputs, targets = self._prepare_batch(batch)

            self.optimizer.zero_grad()
            
            predictions, features_activated, features = self.transcoder(inputs)
            
            loss_dict = self.loss_fn(
                predictions, 
                targets,
                features_activated,
                self.transcoder.features_to_outputs.weight,
                self.transcoder.act.threshold
            )
            
            loss_dict['total_loss'].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.transcoder.parameters(), 1.0)
            if run:
                run.log({"grad_norm_preclip": grad_norm.item()})
            self.optimizer.step()
            
            for loss_type in epoch_losses.keys():
                epoch_losses[loss_type] += loss_dict[f'{loss_type}_loss'].item()
                if run:
                    run.log({f"train_{loss_type}": loss_dict[f'{loss_type}_loss'].item()})

            with torch.no_grad():
                if run:
                    run.log({"train_wdecoder_norms": torch.norm(self.transcoder.features_to_outputs.weight, dim=0).mean()})
                    run.log({"train_bdecoder_norms": torch.norm(self.transcoder.features_to_outputs.bias)})
                    run.log({"train_wencoder_norms": torch.norm(self.transcoder.input_to_features.weight, dim=0).mean()})
                    run.log({"train_bencoder_norms": torch.norm(self.transcoder.input_to_features.bias)})
                    log_thresholds = self.transcoder.act.threshold
                    run.log({"jrelu_thresh": log_thresholds.mean().item(),
                             "jrelu_thresh_min": log_thresholds.min().item(),
                             "jrelu_thresh_max": log_thresholds.max().item()})
                    run.log({"features_active": torch.count_nonzero(features_activated)/inputs.size(0)})
                    run.log({"feature_magnitudes": torch.abs(features_activated[features_activated > 0]).mean()})
                    run.log({"sparsity_coeff": self.loss_fn.sparse_scheduler()})
                feature_activation_densities += (features_activated > 0).sum(dim=0)
            
            self.loss_fn.steps +=1
            pbar.set_postfix(loss=f"{loss_dict['total_loss'].item():.4g}")
        for loss_type in epoch_losses:
            epoch_losses[loss_type] /= n_batches
        if run:
            run.log({"num_never_active": feature_activation_densities.size(0) - torch.count_nonzero(feature_activation_densities)})
        return epoch_losses
    
    def validate(self, val_loader: DataLoader, run=None) -> Dict[str, float]:
        """Validate the model"""
        self.transcoder.eval()
        self.loss_fn.eval = True
        epoch_losses = {
            'total': 0, 'reconstruction': 0, 'sparsity': 0, 'penalty': 0, "norm_recon": 0
        }
        
        n_batches = len(val_loader)
        # Running sums for fraction of variance unexplained over the whole
        # validation set: FVU = SSE / Σ||y - ȳ||². Float64 since N is large.
        sse = torch.zeros((), dtype=torch.float64, device=self.device)
        target_sum = None
        target_sq_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        n_rows = 0

        with torch.no_grad():
            for batch in val_loader:
                # FVU is invariant to the affine normalisation, so it is
                # comparable between normalised and raw runs.
                inputs, targets = self._prepare_batch(batch)

                features = self.transcoder.input_to_features(inputs)
                features_activated = self.transcoder.act(features)
                predictions = self.transcoder.features_to_outputs(features_activated)

                targets64 = targets.double()
                sse += ((predictions.double() - targets64) ** 2).sum()
                batch_sum = targets64.sum(dim=0)
                target_sum = batch_sum if target_sum is None else target_sum + batch_sum
                target_sq_sum += (targets64 ** 2).sum()
                n_rows += targets.size(0)
                
                loss_dict = self.loss_fn(
                    predictions,
                    targets,
                    features_activated,
                    self.transcoder.features_to_outputs.weight,
                    self.transcoder.act.threshold
                )
                
                for loss_type in epoch_losses.keys():
                    epoch_losses[loss_type] += loss_dict[f'{loss_type}_loss'].item()
        
        for loss_type in epoch_losses:
            epoch_losses[loss_type] /= n_batches
            if run:
                run.log({f"valid_{loss_type}": epoch_losses[loss_type]})

        total_variance = target_sq_sum - (target_sum ** 2).sum() / max(n_rows, 1)
        epoch_losses["fvu"] = (sse / total_variance.clamp_min(1e-12)).item()
        if run:
            run.log({"valid_fvu": epoch_losses["fvu"]})

        if self.splice_evaluator is not None:
            splice_metrics = self.splice_evaluator.evaluate(self.raw_transcoder())
            epoch_losses.update(splice_metrics)
            if run:
                run.log({f"valid_{k}": v for k, v in splice_metrics.items()})

        return epoch_losses
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              n_epochs: int,
              previous_epochs: int = 0,
              save_path: str = None,
              save_every: int = 25, run=None) -> Dict[str, float]:
        if previous_epochs < 0:
            raise ValueError("previous_epochs must be non-negative")

        # Put a resumed run on the same batch-level schedule timeline as an
        # uninterrupted run. --n_epochs is the number of *additional* epochs.
        steps_per_epoch = len(train_loader)
        self.loss_fn.total_steps = (previous_epochs + n_epochs) * steps_per_epoch
        self.loss_fn.off *= steps_per_epoch
        self.loss_fn.steps = previous_epochs * steps_per_epoch
        print(f"Running for {n_epochs} more epochs (resuming after {previous_epochs} epochs)")
        pbar = tqdm(range(n_epochs))
        final_val_losses = None
        self.completed_epochs = previous_epochs
        for epoch in pbar:
            absolute_epoch = previous_epochs + epoch
            train_losses = self.train_epoch(train_loader, run=run, epoch=absolute_epoch)
            self.completed_epochs = absolute_epoch + 1
            
            if epoch % save_every == 0 or epoch == n_epochs - 1:
                val_losses = self.validate(val_loader, run=run)
                final_val_losses = val_losses

            for loss_type in train_losses.keys():
                self.train_history[loss_type].append(train_losses[loss_type])
                self.val_history[loss_type].append(
                    val_losses[loss_type]
                    if epoch % save_every == 0 or epoch == n_epochs - 1
                    else float("nan")
                )

            pbar.set_description(f"Train: {train_losses['total']:.4f}, Val: {val_losses['total']:.4f}")
            
            if absolute_epoch % 10 == 0 and save_path:
                torch.save(self.checkpoint(), f"{save_path}/e{absolute_epoch}.ckpt")
        if save_path:
            torch.save(self.checkpoint(), f"{save_path}/final_model.ckpt")
        self.final_metrics = {
            "previous_epochs": previous_epochs,
            "final_train": train_losses,
            "final_validation": final_val_losses,
            "validation_epochs": [
                previous_epochs + i for i, losses in enumerate(self.val_history["total"])
                if not math.isnan(losses)
            ],
        }
        if save_path:
            with open(f"{save_path}/training_metrics.json", "w") as f:
                json.dump(self.final_metrics, f, indent=2)
        return self.final_metrics
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for j, loss_type in enumerate(['total', 'reconstruction', 'sparsity', 'penalty']):
            ax = axes[i, j]
            epochs = range(len(self.train_history[loss_type]))
            
            ax.plot(epochs, self.train_history[loss_type], label='Train')
            ax.plot(epochs, self.val_history[loss_type], label='Val')
            ax.set_title(f'{loss_type.title()} Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()

def create_and_train_transcoders(dataset: Dict[str, torch.Tensor],
                                 train_cfg: Dict[str, float],
                                 hidden_size: int,
                                 input_size: int,
                                 n_feats: int = 512,
                                 device: str = 'cuda',
                                 n_epochs: int = 500,
                                 batch_size=64,
                                 run=None, save_path=None,
                                 *,
                                 num_workers: int = None,
                                 split_seed: int = None,
                                 init_mode: str = "random",
                                 dpi_scale: float = 0.4,
                                 dpi_calibration_samples: int = 8192,
                                 dpi_seed: int = None,
                                 rnn_path: str = None,
                                 splice_target: str = None,
                                 splice_sequence_paths: List[str] = None,
                                 splice_samples: float = 0.05,
                                 splice_traces: List[Dict] = None):
    """
    Create and train transcoder models
    
    Args:
        dataset: Output from TranscoderDataGenerator
        hidden_size: Hidden size of the GRU
        input_size: Input size to the GRU 
        n_feats: Number of features in transcoder
        device: Device to train on
        n_epochs: Number of training epochs
    """
    if init_mode not in {"random", "paired_dpi"}:
        raise ValueError(f"Unknown init_mode: {init_mode}")
    if rnn_path and (splice_target is None or not (splice_sequence_paths or splice_traces)):
        raise ValueError("Splice eval (--rnn_path) also needs a splice target and trace sequence paths")

    # Split before sampling DPI examples so calibration never sees validation
    # rows.
    train_loader, val_loader, val_sequence_ids = create_transcoder_dataloaders(
        dataset, batch_size=batch_size, num_workers=num_workers,
        split_seed=split_seed, return_val_sequence_ids=True)
    print("--Created Dataloader--")

    splice_evaluator = None
    if rnn_path:
        # Copy-task GRU: input = vocab + delimiter, output = vocab.
        rnn_model = RNN(input_size=input_size, hidden_size=hidden_size,
                        out_size=input_size - 1, out_act=lambda x: x, use_gru=True)
        rnn_model.load_state_dict(torch.load(rnn_path, map_location=device))
        # Callers running many configs (the sweep) pass preloaded traces.
        if splice_traces is None:
            splice_traces = load_splice_traces(splice_sequence_paths)
        splice_evaluator = SpliceEvaluator(
            rnn_model, splice_target, splice_traces, splice_samples,
            val_sequence_ids=val_sequence_ids,
            seed=split_seed if split_seed is not None else 0, device=device)
        del splice_traces  # the evaluator keeps only its sampled sequences

    # Create transcoder
    input_dim = hidden_size + input_size  # [h_{t-1}, x_t]
    
    transcoder = Transcoder(
        input_size=input_dim,
        out_size=hidden_size,  # Update gate size
        n_feats=n_feats
    )
    optimizer = optim.Adam(transcoder.parameters(), lr=train_cfg["lr"])
    continuing = bool(train_cfg["ctd_from"])
    normalize = train_cfg.get("normalize", False)
    normalization = None
    if continuing and init_mode != "random":
        raise ValueError("Continuation training cannot also apply a fresh initialization")
    if continuing:
        ckpt = torch.load(train_cfg["ctd_from"], weights_only=True, map_location=device)
        transcoder.load_state_dict(ckpt["transcoder"])
        normalization = ckpt.get("normalization")
        if normalize != (normalization is not None):
            raise ValueError("--normalize must match how the --ctd_from checkpoint was trained")
        if normalization is not None:
            # Checkpoints hold raw-unit weights; the optimizer state belongs to
            # the normalised-unit weights, so unfold before resuming.
            # (The transcoder is still on CPU here; the trainer moves both.)
            normalization = {k: v.cpu() for k, v in normalization.items()}
            unfold_normalization_(transcoder, normalization)
        optimizer.load_state_dict(ckpt["optim"])
        for param, state in optimizer.state.items():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    # Moments saved for a legacy scalar threshold must match
                    # the per-feature threshold now ("step" stays scalar).
                    if k != "step" and v.ndim == 0 and param.ndim == 1:
                        v = v.expand_as(param).clone()
                    state[k] = v.to(device)
    print("--Initialized Transcoder--")
    initialization_metadata = {"mode": init_mode}
    if normalize and not continuing:
        norm_inputs, norm_targets = sample_paired_dpi_calibration(
            train_loader.dataset, NORMALIZATION_SAMPLES, dpi_seed=split_seed)
        normalization = compute_normalization(norm_inputs, norm_targets)
        print(f"--Normalizing: input scale {normalization['input_scale'].item():.4g}, "
              f"target scale {normalization['target_scale'].item():.4g}--")
    if normalization is not None:
        initialization_metadata["normalization"] = {
            "input_scale": float(normalization["input_scale"]),
            "target_scale": float(normalization["target_scale"]),
            "samples": NORMALIZATION_SAMPLES,
        }
    if not continuing:
        if init_mode == "paired_dpi":
            calibration_count = max(n_feats, dpi_calibration_samples)
            calibration_inputs, calibration_targets = sample_paired_dpi_calibration(
                train_loader.dataset, calibration_count, dpi_seed=dpi_seed)
            if normalization is not None:
                # DPI initialises the normalised-unit weights being trained.
                cpu_norm = {k: v.cpu() for k, v in normalization.items()}
                calibration_inputs = normalize_inputs(calibration_inputs.float(), cpu_norm)
                calibration_targets = normalize_targets(calibration_targets.float(), cpu_norm)
            dpi_generator = None if dpi_seed is None else torch.Generator().manual_seed(dpi_seed)
            initialization_metadata.update(initialize_paired_dpi(
                transcoder, calibration_inputs, calibration_targets,
                datapoint_scale=dpi_scale, generator=dpi_generator))
            initialization_metadata["dpi_seed"] = dpi_seed
            print("--Initialized Transcoder with paired DPI--")
        else:
            weight_init_fn = set_transcoder_weights(p=0.01)
            transcoder.input_to_features.apply(weight_init_fn)
            transcoder.features_to_outputs.apply(weight_init_fn)
    if save_path:
        with open(f"{save_path}/initialization.json", "w") as f:
            json.dump(initialization_metadata, f, indent=2)
    loss_fn = TranscoderLoss(lambda_sparsity=train_cfg["l_sparsity"], 
                             lambda_penalty=train_cfg["l_penalty"],
                             c_sparsity=train_cfg["c_sparsity"], 
                             sparse_sched=train_cfg["l_schedule"],
                             sparse_sched_off=train_cfg["l_sched_offset"],
                             w_detach=train_cfg["w_det"],
                             scale_pen_distance=train_cfg["scale_pen"],
                             legacy_sparsity=train_cfg.get("legacy_sparsity", False))

    trainer = TranscoderTrainer(
        transcoder=transcoder,
        optimizer=optimizer,
        device=device,
        loss_fn=loss_fn,
        splice_evaluator=splice_evaluator,
        normalization=normalization
    )

    sig_handler = SignalManager()
    # Same format as regular checkpoints so --ctd_from can resume from it;
    # completed_epochs is the value to pass as --previous_epochs.
    sig_handler.set_training_context(trainer.transcoder, save_path,
                                     checkpoint_fn=trainer.checkpoint)
    sig_handler.register_handler()

    print("--Beginning Training--")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        previous_epochs=train_cfg.get("previous_epochs", 0),
        save_path=save_path,
        run=run
    )
    
    return trainer, transcoder

if __name__ == "__main__":
    import argparse
    import os
    import wandb
    import json

    parser = argparse.ArgumentParser(description="Train Copy Transcoder")

    parser.add_argument("--input_size", type=int, required=True, help="Vocab Size")
    parser.add_argument("--n_feats", type=int, required=True, help="Number of sequences to generate")
    parser.add_argument("--dataset_paths", nargs="+", type=str, required=True, help="Name of dataset")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--l_sparsity", type=float, required=True)
    parser.add_argument("--l_penalty", type=float, required=True)
    parser.add_argument("--c_sparsity", type=float, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--lambda_sparse_schedule", type=int, required=True)
    parser.add_argument("--l_sparse_offset", type=int, default=1)
    parser.add_argument("--w_detach", action="store_true")
    parser.add_argument("--scale_pen_distance", action="store_true")
    parser.add_argument("--normalize", action="store_true",
                        help="Train on centred, scalar-scaled inputs/targets; checkpoints are folded back to raw units")
    parser.add_argument("--legacy_sparsity", action="store_true",
                        help="Use the older λ²·max-batch-L0 sparsity loss instead of the paper's")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--ctd_from", default=None)
    parser.add_argument("--previous_epochs", type=int, default=0,
                        help="Completed epochs represented by --ctd_from")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--split_seed", type=int, default=None,
                        help="Train/validation split seed; defaults to --seed")
    parser.add_argument("--init_mode", choices=["random", "paired_dpi"], default="random",
                        help="Weight initialization; paired_dpi uses paired training examples")
    parser.add_argument("--dpi_scale", type=float, default=0.4,
                        help="Data-point contribution to paired-DPI initialization")
    parser.add_argument("--dpi_calibration_samples", type=int, default=8192,
                        help="Training rows sampled to calibrate paired-DPI initialization")
    parser.add_argument("--dpi_seed", type=int, default=None,
                        help="DPI sample seed; defaults to --split_seed")
    parser.add_argument("--rnn_path", default=None,
                        help="Copy-task GRU checkpoint; enables the splice eval at validation steps")
    parser.add_argument("--splice_target", choices=SPLICE_TARGETS, default=None,
                        help="Gate this transcoder replaces in the splice eval")
    parser.add_argument("--splice_sequence_paths", nargs="+", default=None,
                        help="Trace sequence files (*_seqN.pt) to run the splice eval on")
    parser.add_argument("--splice_samples", type=float, default=0.05,
                        help="Held-out sequences to splice: <=1 is a fraction, >1 a count")

    args = parser.parse_args()
    if args.previous_epochs and not args.ctd_from:
        parser.error("--previous_epochs requires --ctd_from")
    if args.rnn_path and (args.splice_target is None or not args.splice_sequence_paths):
        parser.error("--rnn_path requires --splice_target and --splice_sequence_paths")

    run = wandb.init(
        entity="mishaalkandapath",
        project="rnnsuperpos",
        config={
            "lr": args.lr,
            "l_sparse": args.l_sparsity,
            "c_sparsity": args.c_sparsity,
            "l_penalty": args.l_penalty,
            "n_hidden": args.hidden_size,
            "bandwidth": 2,
            "n_feats": args.n_feats,
            "n_epochs": args.n_epochs,
            "previous_epochs": args.previous_epochs,
            "seed": args.seed,
            "split_seed": args.split_seed if args.split_seed is not None else args.seed,
            "init_mode": args.init_mode,
            "dpi_scale": args.dpi_scale,
            "dpi_calibration_samples": args.dpi_calibration_samples,
            "dpi_seed": args.dpi_seed if args.dpi_seed is not None else (args.split_seed if args.split_seed is not None else args.seed),
            "w_det": int(args.w_detach),
            "scale_pen_distance": int(args.scale_pen_distance),
            "legacy_sparsity": int(args.legacy_sparsity),
            "normalize": int(args.normalize),
            "splice_target": args.splice_target if args.rnn_path else None,
            "splice_samples": args.splice_samples if args.rnn_path else None
        },
    )
    # run = None
    torch.manual_seed(args.seed)
    train_cfg = {"lr": args.lr, "l_sparsity": args.l_sparsity, 
                 "l_schedule": args.lambda_sparse_schedule, 
                 "l_sched_offset": args.l_sparse_offset, "w_det": args.w_detach,
                 "l_penalty":args.l_penalty, "c_sparsity":args.c_sparsity, "scale_pen": args.scale_pen_distance, "legacy_sparsity": args.legacy_sparsity, "normalize": args.normalize, "ctd_from":args.ctd_from,
                 "n_epochs": args.n_epochs, "n_feats": args.n_feats, "batch_size":args.batch_size,
                 "num_workers": args.num_workers, "previous_epochs": args.previous_epochs,
                 "init_mode": args.init_mode, "dpi_scale": args.dpi_scale,
                 "dpi_calibration_samples": args.dpi_calibration_samples,
                 "dpi_seed": args.dpi_seed if args.dpi_seed is not None else (args.split_seed if args.split_seed is not None else args.seed)}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("--Loading Dataset(s)--")
    datasets = []
    for data_path in args.dataset_paths:
        datasets += [torch.load(data_path, map_location=torch.device("cpu"))]
    dataset = ConcatDataset(datasets)
    print("--Finished Loading Dataset--")
    os.makedirs(args.save_path, exist_ok=True)
    with open(f"{args.save_path}/hyperparams.json", "w") as f:
        json.dump(train_cfg, f, indent=4)

    create_and_train_transcoders(dataset, train_cfg, 
                                 hidden_size=args.hidden_size, 
                                 input_size=args.input_size, 
                                 n_feats=args.n_feats, device=device, n_epochs=args.n_epochs,
                                 batch_size=args.batch_size, num_workers=args.num_workers,
                                 split_seed=args.split_seed if args.split_seed is not None else args.seed,
                                 init_mode=args.init_mode, dpi_scale=args.dpi_scale,
                                 dpi_calibration_samples=args.dpi_calibration_samples,
                                 dpi_seed=args.dpi_seed if args.dpi_seed is not None else (args.split_seed if args.split_seed is not None else args.seed),
                                 rnn_path=args.rnn_path, splice_target=args.splice_target,
                                 splice_sequence_paths=args.splice_sequence_paths,
                                 splice_samples=args.splice_samples,
                                 save_path=args.save_path, run=run)
