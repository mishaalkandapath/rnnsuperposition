from dataclasses import dataclass
from typing import Literal
import matplotlib.pyplot as plt

# import waitGPU
# waitGPU.wait(memory_ratio=0.001,
#              gpu_ids=[0,1], interval=10, nproc=1, ngpu=1)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm 

from models.rnn import RNN
from datasets.task_datasets import generate_sparse_copyset, generate_token_copy_dataset, add_delimiter_dimension
from training.train_utils import SignalManager, CopyConfig, constant_lr, mse_loss, cross_entropy_loss

def training_step(model, 
                  data_seq,
                  targets,
                  loss_mask,
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
    outputs, _ = model(data_seq)
    targets[~loss_mask] = -100
    loss = criterion(outputs, targets, importances)
    return loss, outputs

def inference_generate(model, input_data, discrete=False, record_gates=False):
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
        if record_gates:
            rnn_outputs, final_hiddens, r_t_seq, z_t_seq, h_new_t_seq, h_t_seq = model(input_with_delim, record_gates=True)
        else:
            rnn_outputs, final_hiddens = model(input_with_delim)
    # Start generation from the last hidden state
    current_hidden = [h.clone() for h in final_hiddens]
    generated_sequence = [rnn_outputs[:, -1]]
    z_t_rest, r_t_rest, h_new_t_rest, h_t_rest = [], [], [], []
    
    # Generate autoregressively
    for _ in range(copy_length-1):
        # Use previous generated output
        prev_output = generated_sequence[-1]  # (batch_size, n_features)
        if discrete:
            prev_output = nn.functional.one_hot(prev_output.argmax(-1),
                                                prev_output.shape[-1])
        # Add delimiter dimension (0 for generation)
        next_input = torch.cat([prev_output.unsqueeze(1), 
                                torch.zeros(batch_size, 1, 1, device=device)], dim=-1)
        
        # Forward pass for one step
        with torch.no_grad():
            if record_gates:
                step_output, current_hidden, r_t_current, z_t_current, h_new_t_current, h_t_prev = model(next_input, current_hidden, record_gates=True)
                z_t_rest.append(z_t_current)
                r_t_rest.append(r_t_current)
                h_new_t_rest.append(h_new_t_current)
                h_t_rest.append(h_t_prev)
            else:
                step_output, current_hidden = model(next_input, current_hidden)
        
        # Extract the generated features (remove delimiter dimension if present)
        generated_step = step_output[:, 0, :]
        generated_sequence.append(generated_step)
        final_hiddens.append(current_hidden)
    
    # Stack generated sequence
    generated = torch.stack(generated_sequence, dim=1)  # (batch_size, copy_length, n_features)
    if record_gates:
        return generated, final_hiddens, torch.cat([r_t_seq, torch.cat(r_t_rest, dim=2)], dim=2), torch.cat([z_t_seq, torch.cat(z_t_rest, dim=2)], dim=2), torch.cat([h_new_t_seq, torch.cat(h_new_t_rest, dim=2)], dim=2), torch.cat([h_t_seq, torch.cat(h_t_rest, dim=2)], dim=2) 
    return generated, final_hiddens

# 90K, 1e-4, 1e-3
def train_model_copy(config, num_epochs=15000, lr=1e-3, 
                     w_decay=1e-2, lr_schedule=constant_lr, run=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RNN(
                input_size=config.n_tokens + 1,  # +1 for delimiter dimension
                hidden_size=config.d_hidden,
                out_size=config.n_tokens,  # Output original feature dimensions
                out_act=lambda x: x, 
                num_layers=1,
                use_gru=config.gru
            ).to(device)

    sig_handler = SignalManager()
    sig_handler.set_training_context(model, config.run_name)

    if config.ctd_from:
        model.load_state_dict(torch.load(f"models/copy_train/{config.ctd_from}/{config.ctd_from}.ckpt"))
    if config.data_path:
        print(f"loading dataset {config.data_path}")
        train_dataset = torch.load(f"data/copy_train/{config.data_path}.pt")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=w_decay)
    
    criterion = cross_entropy_loss
    train_losses = []

    if not config.data_path:
        train_dataset, test_dataset = generate_token_copy_dataset(config.n_tokens,
                                                                1e6,
                                                                5e3,
                                                                config.max_len,
                                                                min_len=config.min_len
                                                                )
        # save test set somewhere:
        print("saving generated dataset")
        torch.save(train_dataset, f"{config.run_name}/{config.run_name.split('/')[-1]}.pt")
        torch.save(test_dataset, f"{config.run_name}/{config.run_name.split('/')[-1]}.pt")

    loader = DataLoader(train_dataset, batch_size=config.batch_size,
                         shuffle=True, pin_memory=True)
    pbar = tqdm(range(num_epochs))
    epoch = 0
    while epoch <= num_epochs:
        for data, loss_mask, targets in loader:
            data, loss_mask, targets = data.to(device), loss_mask.to(device), targets.to(device)
            
            # Training step
            model.train()
            optimizer.zero_grad()
            
            loss, predictions = training_step(
                model, data, targets, loss_mask, criterion
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

def evaluate_model_copy(config, model, data_path=None):
    device = next(model.parameters()).device
    model.eval()
    
    total_mse = 0
    perfect_copies = 0
    total_sequences = 0

    with torch.no_grad():
            # Generate test data
        test_dataset = torch.load(f"data/copy_test/{config.run_name if not data_path else data_path}.pt")
        for batch_idx in tqdm(range(len(test_dataset))):
            # Inference generation
            test_data, test_loss_mask = test_dataset[batch_idx]
            test_seq_len = test_loss_mask.sum(-1)
            test_data = test_data[:test_seq_len].unsqueeze(0)
            generated, _ = inference_generate(model, test_data,
                                              discrete=True)
            test_target = test_data.argmax(-1)
            # Calculate MSE
            loss = cross_entropy_loss(generated, test_target)
            total_mse += loss.item()
            generated = generated.argmax(-1)
            # Check for perfect reconstructions (within small tolerance)
            perfect_batch = torch.all(
                generated == test_target
            )
            if perfect_batch:
                perfect_copies += config.batch_size
            total_sequences += config.batch_size
    
    avg_mse = total_mse / len(test_dataset)
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
    parser.add_argument("--ctd_from", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
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