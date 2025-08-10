from dataclasses import dataclass
from typing import Literal
# import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm 

from rnn import RNN
from data import generate_sparse_copyset    

# Learning rate schedulers
def linear_lr(step, steps):
    return 1 - (step / steps)

def constant_lr(*_):
    return 1.0

def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))

@dataclass
class CopySuperPosConfig:
    n_inst: int
    n_features: int = 5
    d_hidden: int = 2
    copy_length: int = 3
    batch_size: int = 32
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0
    feat_mag_distn: Literal["unif", "normal"] = "unif"
    gru: bool = False

def add_delimiter_dimension(data, concat_del=True):
    """
    Add delimiter dimension to data
    Args:
        data: (batch_size, seq_len, n_features)
        delimiter_pos: position to place delimiter, if None adds at end
    Returns:
        enhanced_data: (batch_size, seq_len + 1, n_features + 1) if delimiter added
                      or (batch_size, seq_len, n_features + 1) if delimiter_pos specified
    """
    batch_size, seq_len, n_features = data.shape
    
    # Add delimiter at the end
    # Expand original data with 0s in delimiter dimension
    expanded_data = torch.cat([data, torch.zeros(batch_size, seq_len, 1)], dim=-1)
    if not concat_del: return expanded_data
    
    # Create delimiter token: all zeros except delimiter dimension = 1
    delimiter = torch.zeros(batch_size, 1, n_features + 1)
    delimiter[:, :, -1] = 1  # Set delimiter dimension to 1
    
    # Concatenate original data + delimiter
    enhanced_data = torch.cat([expanded_data, delimiter], dim=1)
    return enhanced_data

def prepare_training_data(sparse_data):
    """
    Prepare training data with delimiter token and teacher forcing
    Args:
        sparse_data: (batch_size, copy_length, n_features)
    Returns:
        input_seq: (batch_size, seq_len, n_features + 1) - input + delimiter + target
        target_seq: (batch_size, seq_len, n_features) - targets (ignore positions then copy)
        copy_start_idx: index where copying should start in targets
    """
    batch_size, copy_length, n_features = sparse_data.shape
    
    # Add delimiter dimension to input data
    input_with_delimiter = add_delimiter_dimension(sparse_data)
    # input_with_delimiter: (batch_size, copy_length + 1, n_features + 1)
    
    # For teacher forcing, append the target sequence (original data with delimiter dim)
    target_expanded = add_delimiter_dimension(sparse_data, concat_del=False)
    target_expanded = target_expanded[:, :-1] # the last production does not need to be passed in
    # target_expanded: (batch_size, copy_length-1, n_features + 1)
    
    # Full input sequence: input + delimiter + target (for teacher forcing)
    input_seq = torch.cat([input_with_delimiter, target_expanded], dim=1)
    
    # Target sequence: ignore first (copy_length) positions, then expect original data
    target_seq = sparse_data.clone()
    
    return input_seq, target_seq, copy_length

def training_step(model, input_seq, target_generation, 
                  copy_start_idx, criterion, importances):
    """
    Single training step with teacher forcing
    Args:
        model: RNN model
        input_seq: (batch_size, seq_len, n_features + 1)
        target_seq: (batch_size, seq_len, n_features)
        copy_start_idx: where to start computing loss
        criterion: loss function
    Returns:
        loss: scalar loss value
        predictions: model outputs for analysis
    """
    # Forward pass
    outputs, hidden_states = model(input_seq)
    
    # Only compute loss on the generation part (after delimiter)
    pred_generation = outputs[:, copy_start_idx:, :]
    
    loss = criterion(pred_generation, target_generation, importances)
    
    return loss, outputs

def inference_generate(model, input_data):
    """
    Generate sequence autoregressively during inference
    Args:
        model: RNN model
        input_data: (batch_size, copy_length, n_features) - original sparse data
        copy_length: number of positions to generate
    Returns:
        generated: (batch_size, copy_length, n_features) - generated sequence
        all_outputs: all RNN outputs for analysis
    """
    batch_size, copy_length, n_features = input_data.shape
    device = input_data.device
    
    # Add delimiter dimension to input
    input_with_delim = add_delimiter_dimension(input_data)
    # input_with_delim: (batch_size, copy_length + 1, n_features + 1)
    
    # Process input + delimiter
    with torch.no_grad():
        rnn_outputs, final_hiddens = model(input_with_delim)
    # Start generation from the last hidden state
    current_hidden = [h.clone() for h in final_hiddens]
    generated_sequence = [rnn_outputs[:, -1]]
    
    # Generate autoregressively
    for step in range(copy_length-1):
        # Use previous generated output
        prev_output = generated_sequence[-1]  # (batch_size, n_features)
        # Add delimiter dimension (0 for generation)
        next_input = torch.cat([prev_output.unsqueeze(1), 
                                torch.zeros(batch_size, 1, 1, device=device)], dim=-1)
        
        # Forward pass for one step
        with torch.no_grad():
            step_output, current_hidden = model(next_input, current_hidden)
        
        # Extract the generated features (remove delimiter dimension if present)
        generated_step = step_output[:, 0, :]
        generated_sequence.append(generated_step)
        final_hiddens.append(current_hidden)
    
    # Stack generated sequence
    generated = torch.stack(generated_sequence, dim=1)  # (batch_size, copy_length, n_features)
    
    return generated, final_hiddens

def mse_loss(pred, target, importances=1):
    """MSE loss function"""
    return (importances * (pred - target) ** 2).mean()

def train_model(config, feature_probabilities, importances=1, num_epochs=1000, 
                 lr=0.001, lr_schedule=constant_lr, w_decay=0):
    """
    Train RNN model on copy task
    Args:
        config: CopySuperPosConfig
        feature_probabilities: list of feature_probability values for different instances
        num_epochs: number of training epochs
        lr: learning rate
        lr_schedule: learning rate schedule function
    Returns:
        models: list of trained models
        train_losses: training losses
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models for each feature_probability level (multiple instances)
    models = []
    optimizers = []
    
    for i in range(config.n_inst):
        model = RNN(
            input_size=config.n_features + 1,  # +1 for delimiter dimension
            hidden_size=config.d_hidden,
            out_size=config.n_features,  # Output original feature dimensions
            num_layers=1,
            use_gru=config.gru
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
        models.append(model)
        optimizers.append(optimizer)
    
    criterion = mse_loss
    train_losses = [[] for _ in range(config.n_inst)]
    
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        epoch_losses = []
        
        # Generate fresh data each epoch
        for inst_idx, feature_probability in enumerate(feature_probabilities):
            # Generate sparse data
            sparse_data = generate_sparse_copyset(
                config.n_features, feature_probability, config.copy_length, config.batch_size
            ).to(device)
            
            # Prepare training data
            input_seq, target_seq, copy_start_idx = prepare_training_data(sparse_data)
            
            # Training step
            models[inst_idx].train()
            optimizers[inst_idx].zero_grad()
            
            loss, predictions = training_step(
                models[inst_idx], input_seq, target_seq, copy_start_idx, criterion, importances
            )
            
            loss.backward()
            optimizers[inst_idx].step()
            
            epoch_losses.append(loss.item())
            train_losses[inst_idx].append(loss.item())
        
        # Update learning rates
        current_lr_mult = lr_schedule(epoch, num_epochs)
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * current_lr_mult
        
        avg_loss = np.mean(epoch_losses)
        pbar.set_description(f"Average Loss: {avg_loss:.6f}")
    
    return models, train_losses

def evaluate_model(model, config, feature_probability, num_test_batches=10, importances=1):
    """
    Evaluate model performance on copy task
    Args:
        model: trained RNN model
        config: CopySuperPosConfig
        feature_probability: feature_probability level for test data
        num_test_batches: number of test batches
    Returns:
        avg_mse: average MSE on test data
        perfect_copies: fraction of perfectly reconstructed sequences
    """
    device = next(model.parameters()).device
    model.eval()
    
    total_mse = 0
    perfect_copies = 0
    total_sequences = 0
    
    with torch.no_grad():
        for batch_idx in range(num_test_batches):
            # Generate test data
            test_data = generate_sparse_copyset(
                config.n_features, feature_probability, config.copy_length, config.batch_size
            ).to(device)
            
            # Inference generation
            generated, _ = inference_generate(model, test_data)
            
            # Calculate MSE
            mse = mse_loss(generated, test_data, importances=importances)
            total_mse += mse.item()
            
            # Check for perfect reconstructions (within small tolerance)
            perfect_batch = torch.allclose(generated, test_data, atol=1e-3)
            if perfect_batch:
                perfect_copies += config.batch_size
            total_sequences += config.batch_size
    
    avg_mse = total_mse / num_test_batches
    perfect_copy_rate = perfect_copies / total_sequences
    
    return avg_mse, perfect_copy_rate

# Example usage
if __name__ == "__main__":
    # Configuration
    cfg = CopySuperPosConfig(n_inst=1, n_features=5, d_hidden=2, copy_length=3, batch_size=2)
    
    # Create feature_probability levels (from paper setup)
    feature_probabilities = torch.linspace(0.7, 0.9, cfg.n_inst).tolist()
    
    print("Training RNN models on copy task...")
    print(f"Config: {cfg}")
    print(f"feature_probability levels: {feature_probabilities}")
    
    # Train models
    models, losses = train_model(
        cfg, 
        feature_probabilities, 
        num_epochs=500, 
        lr=0.01, 
        lr_schedule=cosine_decay_lr
    )
    
    # Evaluate models
    print("\nEvaluating models...")
    for i, (model, feature_probability) in enumerate(zip(models, feature_probabilities)):
        mse, perfect_rate = evaluate_model(model, cfg, feature_probability, num_test_batches=5)
        print(f"Model {i} (feature_probability={feature_probability:.2f}): MSE={mse:.6f}, Perfect copies={perfect_rate:.2f}")
    
    # # Plot training curves
    # plt.figure(figsize=(10, 6))
    # plt.plot(losses)
    # plt.title("Training Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE Loss")
    # plt.yscale('log')
    # plt.grid(True)
    # plt.show()
    
    # Test inference on a single example
    print("\nTesting inference...")
    test_model = models[0]
    test_sparsity = feature_probabilities[0]
    
    # Generate single test sequence
    test_input = generate_sparse_copyset(cfg.n_features, test_sparsity, cfg.copy_length, 1)
    generated_output, _ = inference_generate(test_model, test_input, cfg.copy_length)
    
    print(f"Input sequence:\n{test_input[0]}")
    print(f"Generated sequence:\n{generated_output[0]}")
    print(f"Reconstruction error: {mse_loss(generated_output, test_input).item():.6f}")