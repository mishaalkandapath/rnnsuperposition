import math
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, StackDataset
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm

from transcoders import Transcoder, set_transcoder_weights
from transcoder_datasets import create_transcoder_dataloaders
# torch.serialization.add_safe_globals([StackDataset])

interrupted = False
train_obj_global = None
run_name_global = None

def signal_handler(signum, frame):
    global interrupted, train_obj_global, run_name_global
    print("\n\nReceived interrupt signal (Ctrl+C). Saving model and exiting gracefully...")
    interrupted = True
    
    if train_obj_global is not None and run_name_global is not None:
        try:
            os.makedirs(f"models/copy_transcoder/{run_name_global}", exist_ok=True)
            save_path = f"models/copy_transcoder/{run_name_global}/interrupted_{run_name_global}.ckpt"
            torch.save(train_obj_global.transcoder.state_dict(), save_path)
            print(f"Model saved to: {save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    print("Exiting...")
    sys.exit(0)

def normalize_batch(x, eps=1e-8):
    """Normalize per batch to zero mean, unit variance."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(eps)
    return (x - mean) / std

class TranscoderLoss(nn.Module):
    """
    Custom loss function for transcoder training:
    L(x,y) = ||y - ŷ(x)||²₂ + λ_S * Σ tanh(c * |f_i(x)| / ||W_d,i||₂) + L_P(x)
    where L_P(x) = λ_P * Σ ReLU(exp(t) - f_i(x)) * ||W_d,i||₂
    """
    
    def __init__(self, lambda_sparsity=1e-3, lambda_penalty=1e-4, 
                 c_sparsity=1.0, sparse_sched=0, sparse_sched_off=1, w_detach=False):
        super().__init__()
        self.lambda_sparsity = lambda_sparsity
        self.lambda_penalty = lambda_penalty
        self.c_sparsity = c_sparsity
        self.mse_loss = nn.MSELoss()

        # options for traiing variants
        self.total_steps = 0
        self.steps = 0
        self.w_detach = w_detach
        self.eval = False

        self.sparse_scheduler = self.set_lambda_sparse_schedule(sparse_sched, sparse_sched_off)

    def set_lambda_sparse_schedule(self, typ, off=1):
        x = (self.steps)/self.total_steps
        match typ:
            case 1: # linear rise + cut
                return lambda: min(self.steps/off, 1)
            case 2:
                # smoother linear rise + cut
                return lambda: 1/(1+math.exp(-10*(x-0.5)))
            case 3:
                # cosine annealing sparsity loss
                x = (self.steps % off)/off
                return lambda: math.sin(0.5*math.pi*x)
            case _:
                return lambda: x

        
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
        sparsity_coeff = self.lambda_sparsity
        if not self.eval:
            sparsity_coeff *= self.sparse_scheduler()
        # Reconstruction loss: ||y - ŷ(x)||²₂
        reconstruction_loss = self.mse_loss(predictions, targets)
        
        # Sparsity loss: λ_S * Σ tanh(c * |f_i(x)| / ||W_d,i||₂)
        decoder_norms = torch.norm(decoder_weights, dim=0)  # ||W_d,i||₂ for each feature
        feature_magnitudes = torch.abs(features)  # |f_i(x)|
        
        # Avoid division by zero
        decoder_norms = torch.clamp(decoder_norms, min=1e-8)
        decoder_norms = decoder_norms if not self.w_detach else decoder_norms.detach()
        
        normalized_features = feature_magnitudes / decoder_norms.unsqueeze(0)  # Broadcast over batch
        sparsity_terms = torch.tanh(self.c_sparsity * normalized_features)
        sparsity_loss = sparsity_coeff * torch.sum(sparsity_terms)
        
        # Penalty loss: L_P(x) = λ_P * Σ ReLU(exp(t) - f_i(x)) * ||W_d,i||₂
        penalty_terms = torch.relu(torch.exp(threshold) - features) * decoder_norms.unsqueeze(0)
        penalty_loss = self.lambda_penalty * torch.sum(penalty_terms)
        
        total_loss = reconstruction_loss + sparsity_loss + penalty_loss
        # print(
        #     "Penalty threshold:", threshold.item(),
        #     "Penalty features min/max:", features.min().item(), features.max().item(),
        #     "Penalty inactive count:", (features == 0).sum().item()
        # )
        # print(penalty_loss)
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'sparsity_loss': sparsity_loss,
            'penalty_loss': penalty_loss
        }

class TranscoderTrainer:
    """Trainer class for transcoder models"""
    
    def __init__(self, 
                 transcoder: nn.Module,
                 device: str = 'cuda',
                 lr: float = 1e-3,
                 loss_fn: TranscoderLoss=TranscoderLoss(10, 3e-6, 4)):
        
        self.transcoder = transcoder.to(device)
        self.device = device
        
        # Initialize loss functions
        self.loss_fn = loss_fn
        
        # Initialize optimizers
        self.optimizer = optim.Adam(self.transcoder.parameters(), lr=lr)
        
        # Training history
        self.train_history = {
            'total': [], 'reconstruction': [], 'sparsity': [], 'penalty': []
        }
        self.val_history = {
            'total': [], 'reconstruction': [], 'sparsity': [], 'penalty': []
        }
        
    def train_epoch(self, train_loader: DataLoader, run=None) -> Dict[str, float]:
        """Train for one epoch"""
        self.transcoder.train()
        feature_activation_densities = torch.zeros((self.transcoder.n_feats)).to(torch.device("cuda"))
        self.loss_fn.eval = False
        epoch_losses = {
            'total': 0, 'reconstruction': 0, 'sparsity': 0, 'penalty': 0
        }
        
        n_batches = len(train_loader)
        
        # pbar = tqdm(train_loader, total=n_batches)
        for batch in train_loader:
            # Move data to device
            inputs = batch['input'].to(self.device)
            targets = batch['output'].to(self.device)

            inputs = normalize_batch(inputs)
            targets = normalize_batch(targets)
            
            # Train update gate transcoder
            self.optimizer.zero_grad()
            
            # Forward pass through encoder to get features
            predictions, features_activated, features = self.transcoder(inputs)
            
            # Compute loss
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
            
            # Accumulate losses
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
                    run.log({"feature_magnitudes": torch.abs(features_activated[features_activated != 0]).mean()})
                    run.log({"sparsity_coeff": self.loss_fn.steps/self.loss_fn.total_steps})
                feature_activation_densities += (features != 0).sum(dim=0)
            
            self.loss_fn.steps +=1
        # Average losses
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
            'total': 0, 'reconstruction': 0, 'sparsity': 0, 'penalty': 0
        }
        
        n_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                inputs = batch['input'].to(self.device)
                targets = batch['output'].to(self.device)
                
                # Update gate transcoder
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
                
                # Accumulate losses
                for loss_type in epoch_losses.keys():
                    epoch_losses[loss_type] += loss_dict[f'{loss_type}_loss'].item()
        
        # Average losses
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
        """Train the transcoders"""
        
        self.loss_fn.total_steps = n_epochs * (len(train_loader.dataset)//train_loader.batch_size + 1)
        best_val_loss = float('inf')
        print("Running for ", n_epochs)
        pbar = tqdm(range(n_epochs))
        for epoch in pbar:
            # Train
            train_losses = self.train_epoch(train_loader, run=run)
            
            if epoch % save_every == 0:
                # Validate
                val_losses = self.validate(val_loader, run=run)
                best_val_loss = val_losses['total']

            # Store history
            for loss_type in train_losses.keys():
                self.train_history[loss_type].append(train_losses[loss_type])
                self.val_history[loss_type].append(val_losses[loss_type])

            pbar.set_description(f"Train: {train_losses['total']:.4f}, Val: {val_losses['total']:.4f}")
            
            # Save best model
            if epoch % save_every == 0 and save_path and val_losses['total'] < best_val_loss:
                torch.save({
                    'transcoder': self.transcoder.state_dict(),
                    'epoch': epoch,
                    'train_history': self.train_history,
                    'val_history': self.val_history
                }, f"{save_path}/best_val_loss.ckpt")
        torch.save({
                    'transcoder': self.transcoder.state_dict(),
                    'epoch': epoch,
                    'train_history': self.train_history,
                    'val_history': self.val_history
                }, f"{save_path}/final_model.ckpt")
    
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

# Example usage
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
    global run_name_global, train_obj_global
    # Create transcoders
    input_dim = hidden_size + input_size  # [h_{t-1}, x_t]
    
    transcoder = Transcoder(
        input_size=input_dim,
        out_size=hidden_size,  # Update gate size
        n_feats=n_feats
    )
    print("--Initialized Transcoder--")
    # Initialize weights
    weight_init_fn = set_transcoder_weights(p=0.01)
    transcoder.input_to_features.apply(weight_init_fn)
    transcoder.features_to_outputs.apply(weight_init_fn)
    transcoder.input_to_features.apply(weight_init_fn)
    transcoder.features_to_outputs.apply(weight_init_fn)
    
    # Create data loaders
    train_loader, val_loader = create_transcoder_dataloaders(dataset, batch_size=batch_size)
    print("--Created Dataloader--")
    # Create trainer
    loss_fn = TranscoderLoss(train_cfg["l_sparsity"], train_cfg["l_penalty"],
                             train_cfg["c_sparsity"], train_cfg["l_schedule"],
                             train_cfg["l_schedule_offset"], train_cfg["w_det"])

    trainer = TranscoderTrainer(
        transcoder=transcoder,
        device=device,
        lr=train_cfg["lr"],
        lambda_sparsity=train_cfg["l_sparsity"],
        lambda_penalty=train_cfg["l_penalty"],
        c_sparsity=train_cfg["c_sparsity"]
    )

    train_obj_global = trainer
    run_name_global = save_path.split("/")[-1]
    print("--Beginning Training--")
    # Train
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
    import signal 

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
                  
    parser = argparse.ArgumentParser(description="Train Copy Transcoder")

    parser.add_argument("--input_size", type=int, required=True, help="Vocab Size")
    parser.add_argument("--n_feats", type=int, required=True, help="Number of sequences to generate")
    parser.add_argument("--dataset_path", type=str, required=True, help="Name of dataset")
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
    parser.add_argument("--save_path", required=True)

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
            "n_epochs": args.n_epochs
        },
    )
    # run = None
    torch.manual_seed(2)
    train_cfg = {"lr": args.lr, "l_sparsity": args.l_sparsity, 
                 "l_schedule": args.lambda_sparse_schedule, 
                 "l_sched_offset": args.l_sparse_offset, "w_det": args.w_detach,
                 "l_penalty":args.l_penalty, "c_sparsity":args.c_sparsity}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("--Loading Dataset--")
    dataset = torch.load(args.dataset_path)
    print("--Finished Loading Dataset--")
    os.makedirs(args.save_path, exist_ok=True)

    create_and_train_transcoders(dataset, train_cfg, 
                                 hidden_size=args.hidden_size, 
                                 input_size=args.input_size+1, 
                                 n_feats=args.n_feats, device=device, n_epochs=args.n_epochs, batch_size=args.batch_size, save_path=args.save_path, run=run)
