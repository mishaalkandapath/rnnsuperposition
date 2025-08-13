import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm

from transcoders import Transcoder, set_transcoder_weights
from transcoder_datasets import create_transcoder_dataloaders

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
        """
        # Reconstruction loss: ||y - ŷ(x)||²₂
        reconstruction_loss = self.mse_loss(predictions, targets)
        
        # Sparsity loss: λ_S * Σ tanh(c * |f_i(x)| / ||W_d,i||₂)
        decoder_norms = torch.norm(decoder_weights, dim=1)  # ||W_d,i||₂ for each feature
        feature_magnitudes = torch.abs(features)  # |f_i(x)|
        
        # Avoid division by zero
        decoder_norms = torch.clamp(decoder_norms, min=1e-8)
        
        normalized_features = feature_magnitudes / decoder_norms.unsqueeze(0)  # Broadcast over batch
        sparsity_terms = torch.tanh(self.c_sparsity * normalized_features)
        sparsity_loss = self.lambda_sparsity * torch.mean(sparsity_terms)
        
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
                 update_transcoder: nn.Module,
                 hidden_transcoder: nn.Module,
                 device: str = 'cuda',
                 lr: float = 1e-3,
                 lambda_sparsity: float = 1e-3,
                 lambda_penalty: float = 1e-4,
                 c_sparsity: float = 1.0):
        
        self.update_transcoder = update_transcoder.to(device)
        self.hidden_transcoder = hidden_transcoder.to(device)
        self.device = device
        
        # Initialize loss functions
        self.update_loss_fn = TranscoderLoss(lambda_sparsity, lambda_penalty, c_sparsity)
        self.hidden_loss_fn = TranscoderLoss(lambda_sparsity, lambda_penalty, c_sparsity)
        
        # Initialize optimizers
        self.update_optimizer = optim.Adam(self.update_transcoder.parameters(), lr=lr)
        self.hidden_optimizer = optim.Adam(self.hidden_transcoder.parameters(), lr=lr)
        
        # Training history
        self.train_history = {
            'update': {'total': [], 'reconstruction': [], 'sparsity': [], 'penalty': []},
            'hidden': {'total': [], 'reconstruction': [], 'sparsity': [], 'penalty': []}
        }
        self.val_history = {
            'update': {'total': [], 'reconstruction': [], 'sparsity': [], 'penalty': []},
            'hidden': {'total': [], 'reconstruction': [], 'sparsity': [], 'penalty': []}
        }
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.update_transcoder.train()
        self.hidden_transcoder.train()
        
        epoch_losses = {
            'update': {'total': 0, 'reconstruction': 0, 'sparsity': 0, 'penalty': 0},
            'hidden': {'total': 0, 'reconstruction': 0, 'sparsity': 0, 'penalty': 0}
        }
        
        n_batches = len(train_loader)
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move data to device
            update_inputs = batch['update_gate_inputs'].to(self.device)
            update_targets = batch['update_gate_targets'].to(self.device)
            hidden_inputs = batch['hidden_context_inputs'].to(self.device)
            hidden_targets = batch['hidden_context_targets'].to(self.device)
            
            # Train update gate transcoder
            self.update_optimizer.zero_grad()
            
            # Forward pass through encoder to get features
            update_features = self.update_transcoder.input_to_features(update_inputs)
            update_features_activated = self.update_transcoder.act(update_features)
            update_predictions = self.update_transcoder.features_to_outputs(update_features_activated)
            
            # Compute loss
            update_loss_dict = self.update_loss_fn(
                update_predictions, 
                update_targets,
                update_features_activated,
                self.update_transcoder.features_to_outputs.weight,
                self.update_transcoder.act.threshold
            )
            
            update_loss_dict['total_loss'].backward()
            self.update_optimizer.step()
            
            # Train hidden context transcoder
            self.hidden_optimizer.zero_grad()
            
            # Forward pass through encoder to get features
            hidden_features = self.hidden_transcoder.input_to_features(hidden_inputs)
            hidden_features_activated = self.hidden_transcoder.act(hidden_features)
            hidden_predictions = self.hidden_transcoder.features_to_outputs(hidden_features_activated)
            
            # Compute loss
            hidden_loss_dict = self.hidden_loss_fn(
                hidden_predictions,
                hidden_targets,
                hidden_features_activated,
                self.hidden_transcoder.features_to_outputs.weight,
                self.hidden_transcoder.act.threshold
            )
            
            hidden_loss_dict['total_loss'].backward()
            self.hidden_optimizer.step()
            
            # Accumulate losses
            for loss_type in epoch_losses['update'].keys():
                epoch_losses['update'][loss_type] += update_loss_dict[f'{loss_type}_loss'].item()
                epoch_losses['hidden'][loss_type] += hidden_loss_dict[f'{loss_type}_loss'].item()
        
        # Average losses
        for transcoder in epoch_losses:
            for loss_type in epoch_losses[transcoder]:
                epoch_losses[transcoder][loss_type] /= n_batches
                
        return epoch_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.update_transcoder.eval()
        self.hidden_transcoder.eval()
        
        epoch_losses = {
            'update': {'total': 0, 'reconstruction': 0, 'sparsity': 0, 'penalty': 0},
            'hidden': {'total': 0, 'reconstruction': 0, 'sparsity': 0, 'penalty': 0}
        }
        
        n_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                update_inputs = batch['update_gate_inputs'].to(self.device)
                update_targets = batch['update_gate_targets'].to(self.device)
                hidden_inputs = batch['hidden_context_inputs'].to(self.device)
                hidden_targets = batch['hidden_context_targets'].to(self.device)
                
                # Update gate transcoder
                update_features = self.update_transcoder.input_to_features(update_inputs)
                update_features_activated = self.update_transcoder.act(update_features)
                update_predictions = self.update_transcoder.features_to_outputs(update_features_activated)
                
                update_loss_dict = self.update_loss_fn(
                    update_predictions,
                    update_targets,
                    update_features_activated,
                    self.update_transcoder.features_to_outputs.weight,
                    self.update_transcoder.act.threshold
                )
                
                # Hidden context transcoder
                hidden_features = self.hidden_transcoder.input_to_features(hidden_inputs)
                hidden_features_activated = self.hidden_transcoder.act(hidden_features)
                hidden_predictions = self.hidden_transcoder.features_to_outputs(hidden_features_activated)
                
                hidden_loss_dict = self.hidden_loss_fn(
                    hidden_predictions,
                    hidden_targets,
                    hidden_features_activated,
                    self.hidden_transcoder.features_to_outputs.weight,
                    self.hidden_transcoder.act.threshold
                )
                
                # Accumulate losses
                for loss_type in epoch_losses['update'].keys():
                    epoch_losses['update'][loss_type] += update_loss_dict[f'{loss_type}_loss'].item()
                    epoch_losses['hidden'][loss_type] += hidden_loss_dict[f'{loss_type}_loss'].item()
        
        # Average losses
        for transcoder in epoch_losses:
            for loss_type in epoch_losses[transcoder]:
                epoch_losses[transcoder][loss_type] /= n_batches
                
        return epoch_losses
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              n_epochs: int,
              save_path: str = None,
              print_every: int = 10) -> None:
        """Train the transcoders"""
        
        best_val_loss = float('inf')
        
        for epoch in range(n_epochs):
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_losses = self.validate(val_loader)
            
            # Store history
            for transcoder in ['update', 'hidden']:
                for loss_type in train_losses[transcoder].keys():
                    self.train_history[transcoder][loss_type].append(train_losses[transcoder][loss_type])
                    self.val_history[transcoder][loss_type].append(val_losses[transcoder][loss_type])
            
            # Print progress
            if epoch % print_every == 0:
                print(f"Epoch {epoch}/{n_epochs}")
                print(f"  Update - Train: {train_losses['update']['total']:.4f}, Val: {val_losses['update']['total']:.4f}")
                print(f"  Hidden - Train: {train_losses['hidden']['total']:.4f}, Val: {val_losses['hidden']['total']:.4f}")
            
            # Save best model
            if save_path and val_losses['update']['total'] + val_losses['hidden']['total'] < best_val_loss:
                best_val_loss = val_losses['update']['total'] + val_losses['hidden']['total']
                torch.save({
                    'update_transcoder': self.update_transcoder.state_dict(),
                    'hidden_transcoder': self.hidden_transcoder.state_dict(),
                    'epoch': epoch,
                    'train_history': self.train_history,
                    'val_history': self.val_history
                }, save_path)
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i, transcoder in enumerate(['update', 'hidden']):
            for j, loss_type in enumerate(['total', 'reconstruction', 'sparsity', 'penalty']):
                ax = axes[i, j]
                epochs = range(len(self.train_history[transcoder][loss_type]))
                
                ax.plot(epochs, self.train_history[transcoder][loss_type], label='Train')
                ax.plot(epochs, self.val_history[transcoder][loss_type], label='Val')
                ax.set_title(f'{transcoder.title()} {loss_type.title()} Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
def create_and_train_transcoders(dataset: Dict[str, torch.Tensor],
                               hidden_size: int,
                               input_size: int,
                               n_feats: int = 512,
                               device: str = 'cuda',
                               n_epochs: int = 100):
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
    update_input_dim = hidden_size + input_size  # [h_{t-1}, x_t]
    hidden_input_dim = hidden_size + input_size  # [r_t ⊙ h_{t-1}, x_t]
    
    update_transcoder = Transcoder(
        input_size=update_input_dim,
        out_size=hidden_size,  # Update gate size
        n_feats=n_feats
    )
    
    hidden_transcoder = Transcoder(
        input_size=hidden_input_dim,
        out_size=hidden_size,  # Hidden context size
        n_feats=n_feats
    )
    
    # Initialize weights
    weight_init_fn = set_transcoder_weights(p=0.01)
    update_transcoder.input_to_features.apply(weight_init_fn)
    update_transcoder.features_to_outputs.apply(weight_init_fn)
    hidden_transcoder.input_to_features.apply(weight_init_fn)
    hidden_transcoder.features_to_outputs.apply(weight_init_fn)
    
    # Create data loaders
    train_loader, val_loader = create_transcoder_dataloaders(dataset, batch_size=256)
    
    # Create trainer
    trainer = TranscoderTrainer(
        update_transcoder=update_transcoder,
        hidden_transcoder=hidden_transcoder,
        device=device,
        lr=1e-3,
        lambda_sparsity=1e-3,
        lambda_penalty=1e-4
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        save_path='best_transcoders.pt'
    )
    
    # Plot results
    trainer.plot_training_curves()
    
    return trainer, update_transcoder, hidden_transcoder