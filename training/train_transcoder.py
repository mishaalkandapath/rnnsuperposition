import math
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, StackDataset, ConcatDataset
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.transcoders import Transcoder, set_transcoder_weights
from datasets.utils import create_transcoder_dataloaders, ConsolidatedStackDataset
from training.train_utils import SignalManager, normalize_batch
torch.serialization.add_safe_globals([StackDataset])

class TranscoderLoss(nn.Module):
    """
    Custom loss function for transcoder training:
    L(x,y) = ||y - ŷ(x)||²₂ + λ_S * Σ tanh(c * |f_i(x)| / ||W_d,i||₂) + L_P(x)
    where L_P(x) = λ_P * Σ ReLU(exp(t) - f_i(x)) * ||W_d,i||₂
    """
    
    def __init__(self, lambda_sparsity=1e-3, lambda_penalty=1e-4, 
                 c_sparsity=1.0, sparse_sched=0, sparse_sched_off=1, w_detach=False, scale_pen_distance=False):
        super().__init__()
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
        sparsity_coeff = self.lambda_sparsity
        if not self.eval:
            sparsity_coeff *= self.sparse_scheduler()
        # Reconstruction loss: ||y - ŷ(x)||²₂
        reconstruction_loss = self.mse_loss(predictions, targets)
        with torch.no_grad():
            normalized_reconstruction_loss = self.mse_loss(predictions/torch.norm(predictions, dim=-1, keepdim=True), targets/torch.norm(targets, dim=-1, keepdim=True))
        
        # Sparsity loss: λ_S * Σ tanh(c * |f_i(x)|||W_d,i||₂)
        decoder_norms = torch.norm(decoder_weights, dim=0)  # ||W_d,i||₂ for each feature
        feature_magnitudes = torch.abs(features)  # |f_i(x)|
        
        decoder_norms = torch.clamp(decoder_norms, min=1e-8)
        decoder_norms = decoder_norms if not self.w_detach else decoder_norms.detach()
        
        normalized_features = feature_magnitudes * decoder_norms.unsqueeze(0)
        sparsity_terms = torch.tanh(self.c_sparsity * normalized_features)
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
                 loss_fn: TranscoderLoss=TranscoderLoss(10, 3e-6, 4)):
        
        self.transcoder = transcoder.to(device)
        self.device = device
        
        self.loss_fn = loss_fn
        
        self.optimizer = optimizer
        
        self.train_history = {
            'total': [], 'reconstruction': [], 'sparsity': [], 'penalty': [], "norm_recon": []
        }
        self.val_history = {
            'total': [], 'reconstruction': [], 'sparsity': [], 'penalty': [], "norm_recon": []
        }
        
    def train_epoch(self, train_loader: DataLoader, run=None) -> Dict[str, float]:
        """Train for one epoch"""
        self.transcoder.train()
        feature_activation_densities = torch.zeros((self.transcoder.n_feats)).to(self.device)
        self.loss_fn.eval = False
        epoch_losses = {
            'total': 0, 'reconstruction': 0, 'sparsity': 0, 'penalty': 0, "norm_recon": 0
        }
        
        n_batches = len(train_loader)
        
        # pbar = tqdm(train_loader, total=n_batches)
        for batch in train_loader:
            inputs = batch['input'].to(self.device)
            targets = batch['output'].to(self.device)

            inputs = (inputs)
            targets = (targets)
        
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
            torch.nn.utils.clip_grad_norm_(self.transcoder.parameters(), 1.0)
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
                    run.log({"jrelu_thresh": self.transcoder.act.threshold.item()})
                    run.log({"features_active": torch.count_nonzero(features_activated)/inputs.size(0)})
                    run.log({"feature_magnitudes": torch.abs(features_activated[features_activated > 0]).mean()})
                    run.log({"sparsity_coeff": self.loss_fn.sparse_scheduler()})
                feature_activation_densities += (features_activated > 0).sum(dim=0)
            
            self.loss_fn.steps +=1
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
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = (batch['input']).to(self.device)
                targets = (batch['output']).to(self.device)
                
                features = self.transcoder.input_to_features(inputs)
                features_activated = self.transcoder.act(features)
                predictions = self.transcoder.features_to_outputs(features_activated)
                
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
                
        return epoch_losses
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              n_epochs: int,
              save_path: str = None,
              save_every: int = 25, run=None) -> None:
        
        self.loss_fn.total_steps = n_epochs * (len(train_loader.dataset)//train_loader.batch_size + 1)
        self.loss_fn.off *= (len(train_loader.dataset)//train_loader.batch_size + 1)
        print("Running for ", n_epochs)
        pbar = tqdm(range(n_epochs))
        for epoch in pbar:
            train_losses = self.train_epoch(train_loader, run=run)
            
            if epoch % save_every == 0:
                val_losses = self.validate(val_loader, run=run)
                best_val_loss = val_losses['total']

            for loss_type in train_losses.keys():
                self.train_history[loss_type].append(train_losses[loss_type])
                self.val_history[loss_type].append(val_losses[loss_type])

            pbar.set_description(f"Train: {train_losses['total']:.4f}, Val: {val_losses['total']:.4f}")
            
            if epoch % 10 == 0 and save_path:
                torch.save({"transcoder":self.transcoder.state_dict(),
                            "optim": self.optimizer.state_dict()}, f"{save_path}/e{epoch}.ckpt")
        torch.save({"transcoder":self.transcoder.state_dict(),
                            "optim": self.optimizer.state_dict()}, f"{save_path}/final_model.ckpt")
    
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
                                 run=None, save_path=None):
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
    # Create transcoders
    input_dim = hidden_size + input_size  # [h_{t-1}, x_t]
    
    transcoder = Transcoder(
        input_size=input_dim,
        out_size=hidden_size,  # Update gate size
        n_feats=n_feats
    )
    optimizer = optim.Adam(transcoder.parameters(), lr=train_cfg["lr"])
    if train_cfg["ctd_from"]:
        ckpt = torch.load(train_cfg["ctd_from"], weights_only=True, map_location=device)
        transcoder.load_state_dict(ckpt["transcoder"])
        optimizer.load_state_dict(ckpt["optim"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    print("--Initialized Transcoder--")
    weight_init_fn = set_transcoder_weights(p=0.01)
    transcoder.input_to_features.apply(weight_init_fn)
    transcoder.features_to_outputs.apply(weight_init_fn)
    
    train_loader, val_loader = create_transcoder_dataloaders(dataset, batch_size=batch_size)
    print("--Created Dataloader--")
    loss_fn = TranscoderLoss(lambda_sparsity=train_cfg["l_sparsity"], 
                             lambda_penalty=train_cfg["l_penalty"],
                             c_sparsity=train_cfg["c_sparsity"], 
                             sparse_sched=train_cfg["l_schedule"],
                             sparse_sched_off=train_cfg["l_sched_offset"],
                             w_detach=train_cfg["w_det"],
                             scale_pen_distance=train_cfg["scale_pen"])

    trainer = TranscoderTrainer(
        transcoder=transcoder,
        optimizer=optimizer,
        device=device,
        loss_fn=loss_fn
    )

    sig_handler = SignalManager()
    sig_handler.set_training_context(trainer.transcoder, save_path)
    sig_handler.register_handler()

    print("--Beginning Training--")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
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
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--ctd_from", default=None)

    args = parser.parse_args()

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
            "w_det": int(args.w_detach),
            "scale_pen_distance": int(args.scale_pen_distance)
        },
    )
    # run = None
    torch.manual_seed(2)
    train_cfg = {"lr": args.lr, "l_sparsity": args.l_sparsity, 
                 "l_schedule": args.lambda_sparse_schedule, 
                 "l_sched_offset": args.l_sparse_offset, "w_det": args.w_detach,
                 "l_penalty":args.l_penalty, "c_sparsity":args.c_sparsity, "scale_pen": args.scale_pen_distance, "ctd_from":args.ctd_from,
                 "n_epochs": args.n_epochs, "n_feats": args.n_feats, "batch_size":args.batch_size}

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
                                 n_feats=args.n_feats, device=device, n_epochs=args.n_epochs, batch_size=args.batch_size, save_path=args.save_path, run=run)
