from dataclasses import dataclass
from typing import Literal
import matplotlib.pyplot as plt
import math

import waitGPU
waitGPU.wait(memory_ratio=0.001,
             gpu_ids=[0,1], interval=10, nproc=1, ngpu=1)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm 

from rnn import RNN
from datasets import generate_sparse_copyset, generate_token_copy_dataset

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

@dataclass
class CopyConfig:
    run_name: str
    n_tokens: int = 30
    d_hidden: int = 2
    max_len: int = 9
    min_len: int = 3
    batch_size: int = 32
    n_layers: int = 1
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
    expanded_data = torch.cat([data, torch.zeros(batch_size, seq_len, 1).to(data.device)], dim=-1).to(data.device)
    if not concat_del: return expanded_data
    
    # Create delimiter token: all zeros except delimiter dimension = 1
    delimiter = torch.zeros(batch_size, 1, n_features + 1).to(data.device)
    delimiter[:, :, -1] = 1  # Set delimiter dimension to 1
    
    # Concatenate original data + delimiter
    enhanced_data = torch.cat([expanded_data, delimiter], dim=1)
    return enhanced_data.to(data.device)

def prepare_training_data(data):
    """
    Prepare training data with delimiter token and teacher forcing
    Args:
        data: (batch_size, copy_length, n_features)
    Returns:
        input_seq: (batch_size, seq_len, n_features + 1) - input + delimiter + target
        target_seq: (batch_size, seq_len, n_features) - targets (ignore positions then copy)
        copy_start_idx: index where copying should start in targets
    """
    batch_size, copy_length, n_features = data.shape
    
    # Add delimiter dimension to input data
    input_with_delimiter = add_delimiter_dimension(data)
    # input_with_delimiter: (batch_size, copy_length + 1, n_features + 1)
    
    # For teacher forcing, append the target sequence (original data with delimiter dim)
    target_expanded = add_delimiter_dimension(data, concat_del=False)
    target_expanded = target_expanded[:, :-1] # the last production does not need to be passed in
    # target_expanded: (batch_size, copy_length-1, n_features + 1)
    
    # Full input sequence: input + delimiter + target (for teacher forcing)
    input_seq = torch.cat([input_with_delimiter, target_expanded], dim=1)
    
    # Target sequence: ignore first (copy_length) positions, then expect original data
    target_seq = data.clone()
    
    return input_seq, target_seq, copy_length

def training_step(model, 
                  input_seq, target_generation, copy_start_idx,
                  criterion, importances=1):
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

def cross_entropy_loss(pred, target, importances=1):
    return nn.functional.cross_entropy(pred.transpose(1, 2), target)

def train_model_copy(config, num_epochs=40000, lr=1e-3, 
                     w_decay=0.1, lr_schedule=constant_lr, run=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RNN(
                input_size=config.n_tokens + 1,  # +1 for delimiter dimension
                hidden_size=config.d_hidden,
                out_size=config.n_tokens,  # Output original feature dimensions
                out_act=lambda x: x, 
                num_layers=1,
                use_gru=config.gru
            ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=w_decay)
    
    criterion = cross_entropy_loss
    train_losses = []

    train_dataset, test_dataset = generate_token_copy_dataset(config.n_tokens,
                                                              1e6,
                                                              5e3,
                                                              config.max_len,
                                                              min_len=config.min_len
                                                            )
    # save test set somewhere:
    torch.save(test_dataset, f"data/copy_test/{config.run_name}.pt")
    loader = DataLoader(train_dataset, batch_size=config.batch_size,
                         shuffle=True, pin_memory=True)
    pbar = tqdm(range(num_epochs))
    epoch = 0
    while epoch <= num_epochs:
        for data, loss_mask in loader:
            data, loss_mask = data.to(device), loss_mask.to(device)
            # Prepare training data
            input_seq, target_seq, copy_start_idx = prepare_training_data(data)
            target_seq = data.argmax(dim=-1)
            target_seq[~loss_mask] = -100 # ignore index in CE loss
            
            # Training step
            model.train()
            optimizer.zero_grad()
            
            loss, predictions = training_step(
                model, input_seq, target_seq, copy_start_idx, criterion
            )
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if run:
                run.log({"loss":loss.item()})
            pbar.update(1)
            pbar.set_description(f"Loss: {loss.item():.6f}")
            epoch+=1
            if epoch > num_epochs:
                break
    
        # Update learning rates
        current_lr_mult = lr_schedule(epoch, num_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * current_lr_mult    

    torch.save(model.state_dict(), f"models/copy_train/{config.run_name}/{config.run_name}.ckpt")
    plt.plot(train_losses)
    plt.savefig(f"models/copy_train/{config.run_name}/{config.run_name}_loss.png")
    
    return model, train_losses


def train_model_superpos(config, feature_probabilities, importances=1, num_epochs=1000, 
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

def evaluate_model_copy(config, model, num_test_batches):
    device = next(model.parameters()).device
    model.eval()
    
    total_mse = 0
    perfect_copies = 0
    total_sequences = 0

    with torch.no_grad():
        for batch_idx in range(num_test_batches):
            # Generate test data
            test_data, _ = generate_token_copyset(
                                                 config.n_tokens, 
                                                 1,
                                                 config.max_len, min_len=config.min_len
                                                )
            
            # Inference generation
            generated, _ = inference_generate(model, test_data)
            test_target = test_data.argmax(-1)
            # Calculate MSE
            loss = cross_entropy_loss(generated, test_target)
            total_mse += loss.item()
            
            # Check for perfect reconstructions (within small tolerance)
            perfect_batch = torch.all(
                generated.argmax(-1) == test_target
            )
            if perfect_batch:
                perfect_copies += config.batch_size
            total_sequences += config.batch_size
    
    avg_mse = total_mse / num_test_batches
    perfect_copy_rate = perfect_copies / total_sequences
    
    return avg_mse, perfect_copy_rate

def evaluate_model_superpos(model, config, feature_probability, num_test_batches=10, importances=1):
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
    import argparse
    import os
    import wandb

    parser = argparse.ArgumentParser(description="Train copy model")

    parser.add_argument("--n_tokens", type=int, required=True, help="Number of tokens in vocab")
    parser.add_argument("--batch_size", type=int, required=True, help="Training batch size")
    parser.add_argument("--max_len", type=int, required=True, help="Maximum sequence length")
    parser.add_argument("--min_len", type=int, required=True, help="Minimum sequence length")
    parser.add_argument("--d_hidden", type=int, required=True, help="Hidden layer size")
    parser.add_argument("--n_layers", type=int, required=True, help="Number of RNN layers")
    parser.add_argument("--run_name", type=str, required=True, help="Name of run")
    parser.add_argument("--gru", action="store_true", help="Use GRU?")

    args = parser.parse_args()

    # Return a Config instance populated from args
    cfg = CopyConfig(**vars(args))
    os.makedirs(f"models/copy_train/{args.run_name}", exist_ok=True)
    run = wandb.init(
        entity="mishaalkandapath",
        project="rnnsuperpos",
        config={
            "learning_rate": 1e-3,
            "batch_size": args.batch_size,
            "n_hidden": args.d_hidden, 
            "max_len": args.max_len
        },
    )
    # run=None
    torch.manual_seed(2)
    train_model_copy(cfg, run=run)
    # # Configuration
    # cfg = CopySuperPosConfig(n_inst=1, n_features=5, d_hidden=2, copy_length=3, batch_size=2)
    
    # # Create feature_probability levels (from paper setup)
    # feature_probabilities = torch.linspace(0.7, 0.9, cfg.n_inst).tolist()
    
    # print("Training RNN models on copy task...")
    # print(f"Config: {cfg}")
    # print(f"feature_probability levels: {feature_probabilities}")
    
    # # Train models
    # models, losses = train_model_superpos(
    #     cfg, 
    #     feature_probabilities, 
    #     num_epochs=500, 
    #     lr=0.01, 
    #     lr_schedule=cosine_decay_lr
    # )
    
    # # Evaluate models
    # print("\nEvaluating models...")
    # for i, (model, feature_probability) in enumerate(zip(models, feature_probabilities)):
    #     mse, perfect_rate = evaluate_model_superpos(model, cfg, feature_probability, num_test_batches=5)
    #     print(f"Model {i} (feature_probability={feature_probability:.2f}): MSE={mse:.6f}, Perfect copies={perfect_rate:.2f}")
    
    # # # Plot training curves
    # # plt.figure(figsize=(10, 6))
    # # plt.plot(losses)
    # # plt.title("Training Loss")
    # # plt.xlabel("Epoch")
    # # plt.ylabel("MSE Loss")
    # # plt.yscale('log')
    # # plt.grid(True)
    # # plt.show()
    
    # # Test inference on a single example
    # print("\nTesting inference...")
    # test_model = models[0]
    # test_sparsity = feature_probabilities[0]
    
    # # Generate single test sequence
    # test_input = generate_sparse_copyset(cfg.n_features, test_sparsity, cfg.copy_length, 1)
    # generated_output, _ = inference_generate(test_model, test_input, cfg.copy_length)
    
    # print(f"Input sequence:\n{test_input[0]}")
    # print(f"Generated sequence:\n{generated_output[0]}")
    # print(f"Reconstruction error: {mse_loss(generated_output, test_input).item():.6f}")