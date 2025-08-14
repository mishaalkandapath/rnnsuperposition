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
torch.serialization.add_safe_globals([StackDataset])

class TranscoderLoss(nn.Module):
    """
    Custom loss function for transcoder training:
    L(x,y) = ||y - ŷ(x)||²₂ + λ_S * Σ tanh(c * |f_i(x)| / ||W_d,i||₂) + L_P(x)
    where L_P(x) = λ_P * Σ ReLU(exp(t) - f_i(x)) * ||W_d,i||₂
    """
    
    def __init__(self, lambda_sparsity=1e-3, lambda_penalty=1e-4, c_sparsity=1.0):
        super().__init__()
        self.lambda_sparsity = lambda_sparsity
        self.lambda_penalty = lambda_penalty
        self.c_sparsity = c_sparsity
        self.mse_loss = nn.MSELoss()
        
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
        # Reconstruction loss: ||y - ŷ(x)||²₂
        reconstruction_loss = self.mse_loss(predictions, targets)
        
        # Sparsity loss: λ_S * Σ tanh(c * |f_i(x)| / ||W_d,i||₂)
        decoder_norms = torch.norm(decoder_weights, dim=0)  # ||W_d,i||₂ for each feature
        feature_magnitudes = torch.abs(features)  # |f_i(x)|
        
        # Avoid division by zero
        decoder_norms = torch.clamp(decoder_norms, min=1e-8)
        
        normalized_features = feature_magnitudes * decoder_norms.unsqueeze(0)  # Broadcast over batch
        sparsity_terms = torch.tanh(self.c_sparsity * normalized_features)
        sparsity_loss = self.lambda_sparsity * torch.sum(sparsity_terms)
        
        # Penalty loss: L_P(x) = λ_P * Σ ReLU(exp(t) - f_i(x)) * ||W_d,i||₂
        threshold_expanded = threshold.expand_as(features)
        penalty_terms = torch.relu(threshold_expanded - features) * decoder_norms.unsqueeze(0)
        penalty_loss = self.lambda_penalty * torch.mean(penalty_terms)
        
        total_loss = reconstruction_loss + sparsity_loss + penalty_loss
        
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
                 lambda_sparsity: float = 10,
                 lambda_penalty: float = 3e-6,
                 c_sparsity: float = 4.0):
        
        self.transcoder = transcoder.to(device)
        self.device = device
        
        # Initialize loss functions
        self.loss_fn = TranscoderLoss(lambda_sparsity, lambda_penalty, c_sparsity)
        
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
        
        epoch_losses = {
            'total': 0, 'reconstruction': 0, 'sparsity': 0, 'penalty': 0
        }
        
        n_batches = len(train_loader)
        
        # pbar = tqdm(train_loader, total=n_batches)
        for batch in train_loader:
            # Move data to device
            inputs = batch['input'].to(self.device)
            targets = batch['output'].to(self.device)
            
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
            self.optimizer.step()
            
            # Accumulate losses
            for loss_type in epoch_losses.keys():
                epoch_losses[loss_type] += loss_dict[f'{loss_type}_loss'].item()
                if run:
                    run.log({f"train_{loss_type}": loss_dict[f'{loss_type}_loss'].item()})
        # Average losses
        for loss_type in epoch_losses:
            epoch_losses[loss_type] /= n_batches
            if run:
                    run.log({f"epoch_train_{loss_type}": loss_dict[f'{loss_type}_loss'].item()})
                
        return epoch_losses
    
    def validate(self, val_loader: DataLoader, run=None) -> Dict[str, float]:
        """Validate the model"""
        self.transcoder.eval()
        
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
              save_every: int = 10, run=None) -> None:
        """Train the transcoders"""
        
        best_val_loss = float('inf')
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
                                 n_epochs: int = 100,
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
    trainer = TranscoderTrainer(
        transcoder=transcoder,
        device=device,
        lr=train_cfg["lr"],
        lambda_sparsity=train_cfg["l_sparsity"],
        lambda_penalty=train_cfg["l_penalty"],
        c_sparsity=train_cfg["c_sparsity"]
    )
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
            "n_feats": args.n_feats
        },
    )
    # run = None
    torch.manual_seed(2)

    train_cfg = {"lr": args.lr, "l_sparsity": args.l_sparsity, 
                 "l_penalty":args.l_penalty, "c_sparsity":args.c_sparsity}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("--Loading Dataset--")
    dataset = torch.load(args.dataset_path)
    print("--Finished Loading Dataset--")
    os.makedirs(args.save_path, exist_ok=True)

    create_and_train_transcoders(dataset, train_cfg, 
                                 hidden_size=args.hidden_size, 
                                 input_size=args.input_size+1, 
                                 n_feats=args.n_feats, device=device, n_epochs=100, batch_size=args.batch_size, save_path=args.save_path, run=run)
