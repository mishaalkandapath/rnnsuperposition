from typing import Dict, Tuple

import torch
from torch.utils.data import StackDataset, Subset
from tqdm import tqdm

from datasets import generate_token_copyset
from train import inference_generate, add_delimiter_dimension
from rnn import RNN

class TranscoderDataGenerator:
    """Generate training data for GRU transcoders"""
    
    def __init__(self, rnn_model, device='cpu'):
        """
        Args:
            rnn_model: Trained RNN model (assumed to be single layer GRU)
            device: Device to run computations on
        """
        self.rnn_model = rnn_model
        self.device = device
        self.rnn_model.eval()  # Set to eval mode
        
    def generate_transcoder_dataset(self, 
                                  n_tokens: int,
                                  n_sequences: int, 
                                  batch_size: int,
                                  max_len: int, 
                                  min_len: int = 2) -> StackDataset:
        """
        Generate dataset for training transcoders using unique sequences only
        
        Args:
            n_tokens: Number of tokens in vocabulary
            n_sequences: Number of unique sequences to generate
            max_len: Maximum sequence length
            min_len: Minimum sequence length
            
        Returns:
            Dict with keys:
            - 'update_gate_inputs': Concatenated [h_{t-1}, x_t] for update gate
            - 'update_gate_targets': True update gate values z_t
            - 'hidden_context_inputs': Concatenated [r_t ⊙ h_{t-1}, x_t] for hidden context
            - 'hidden_context_targets': True hidden context values h̃_t (n_t)
        """
        
        # Generate unique sequences
        print(f"Generating {n_sequences} unique sequences...")
        unique_sequences, unique_masks = self._generate_unique_sequences(
            n_tokens, n_sequences, max_len, min_len
        )
        sequence_lengths = unique_masks.sum(dim=1)
        sequences_by_length = [[] for _ in range(min_len, max_len+1)]

        for seq_idx in range(n_sequences):
            actual_length = int(sequence_lengths[seq_idx])
            if min_len <= actual_length <= max_len:
                # Extract the sequence (only the valid part based on mask)
                sequence = unique_sequences[seq_idx][:actual_length]
                sequences_by_length[actual_length - min_len].append(sequence)
            else: 
                raise Exception

        print(f"Done generating sequences {unique_sequences.shape} {unique_masks.shape}")
        
        all_update_inputs = []
        all_update_targets = []
        all_hidden_inputs = []
        all_hidden_targets = []
        
        for unique_sequences in sequences_by_length:
            # Process in batches for memory efficiency
            n_sequences = len(unique_sequences)
            batch_size = min(batch_size, n_sequences)
            n_batches = (n_sequences + batch_size - 1) // batch_size
            unique_sequences = torch.stack(unique_sequences)
            print(unique_sequences.shape)
            with torch.no_grad():
                for batch_idx in tqdm(range(n_batches)):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, n_sequences)
                    if start_idx >= end_idx: break
                    
                    batch_sequences = unique_sequences[start_idx:end_idx].to(self.device)
                    # Run RNN and record gates
                    outs, _, r_records, z_records, h_new_records, h_records =  inference_generate(self.rnn_model, batch_sequences, 
                                    discrete=True, record_gates=True)
                    
                    outs = torch.nn.functional.one_hot(outs.argmax(-1), num_classes=30)
                    
                    # Extract data from single layer (index 0)
                    r_t = r_records[0]  # (batch_size, seq_len, 2*hidden_size)
                    z_t = z_records[0]  # (batch_size, seq_len, 2*hidden_size)  
                    h_new_t = h_new_records[0]  # (batch_size, 2*seq_len, hidden_size)
                    h_prev = h_records[0]  # (batch_size, 2*seq_len, hidden_size)
                    
                    outs = add_delimiter_dimension(outs, concat_del=False)
                    batch_sequences = add_delimiter_dimension(batch_sequences)
                    x_t = torch.cat([batch_sequences, outs[:, :-1]],
                                    dim=1)   # (batch_size, 2*seq_len, input_size)
                    
                    # Process each timestep
                    for t in range(2*unique_sequences.size(1)):
                        # Extract valid samples for this timestep
                        valid_h_prev = h_prev[:, t]  # (n_valid, hidden_size)
                        valid_x_t = x_t[:, t]      # (n_valid, input_size)
                        valid_r_t = r_t[:, t]      # (n_valid, hidden_size)
                        valid_z_t = z_t[:, t]      # (n_valid, hidden_size)
                        valid_h_new_t = h_new_t[:, t]  # (n_valid, hidden_size)
                        
                        # Prepare transcoder inputs and targets
                        
                        # Update gate transcoder: input = [h_{t-1}, x_t], target = z_t
                        update_gate_input = torch.cat([valid_h_prev, valid_x_t], dim=1)
                        update_gate_target = valid_z_t
                        
                        # Hidden context transcoder: input = [r_t ⊙ h_{t-1}, x_t], target = h̃_t
                        gated_hidden = valid_r_t * valid_h_prev  # Element-wise multiplication
                        hidden_context_input = torch.cat([gated_hidden, valid_x_t], dim=1)
                        hidden_context_target = valid_h_new_t
                        
                        # Collect data
                        all_update_inputs.append(update_gate_input)
                        all_update_targets.append(update_gate_target)
                        all_hidden_inputs.append(hidden_context_input)
                        all_hidden_targets.append(hidden_context_target)
        
        # Concatenate all collected data
        dataset = {
            'update_gate_inputs': torch.cat(all_update_inputs, dim=0),
            'update_gate_targets': torch.cat(all_update_targets, dim=0),
            'hidden_context_inputs': torch.cat(all_hidden_inputs, dim=0),
            'hidden_context_targets': torch.cat(all_hidden_targets, dim=0)
        }
        
        print(f"Generated transcoder dataset with {dataset['update_gate_inputs'].shape[0]} samples")
        update_dataset = {"input": dataset["update_gate_inputs"],
                          "output": dataset["update_gate_targets"]}
        hidden_dataset = {"input": dataset["hidden_context_inputs"],
                          "output": dataset["hidden_context_targets"]}
        update_dataset = StackDataset(**update_dataset)
        hidden_dataset = StackDataset(**hidden_dataset)
        return update_dataset, hidden_dataset
    
    def _generate_unique_sequences(self, n_tokens: int, n_sequences: int, 
                                 max_len: int, min_len: int):
        """Generate unique sequences similar to generate_unique_test_set"""
        seen = set()
        sequences = []
        masks = []
        
        pbar = tqdm(total=n_sequences)
        while len(sequences) < n_sequences:
            seq = torch.randint(0, n_tokens, (max_len,))
            seq_len = torch.randint(min_len, max_len+1, (1,))
            mask = torch.arange(max_len) < seq_len
            seq = seq * mask  # zero out pads if needed
            tup = tuple(seq.tolist())
            
            if tup not in seen:
                seen.add(tup)
                sequences.append(seq)
                masks.append(mask)
            pbar.update(1)
        
        sequences = torch.stack(sequences)
        masks = torch.stack(masks)
        sequences_one_hot = torch.nn.functional.one_hot(sequences, num_classes=n_tokens).float()
        
        return sequences_one_hot, masks

def create_transcoder_dataloaders(dataset: StackDataset,
                                batch_size: int = 256,
                                train_split: float = 0.9,
                                shuffle: bool = True) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train/val dataloaders for transcoder training
    
    Args:
        dataset: Output from generate_transcoder_dataset
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training
        shuffle: Whether to shuffle the data
    
    Returns:
        train_loader, val_loader
    """
    
    # Determine dataset size
    n_samples = len(dataset)
    n_train = int(n_samples * train_split)
    
    # Create indices for train/val splits
    indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=192, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate Dataset for Transcoders")

    parser.add_argument("--n_tokens", type=int, required=True, help="Vocab Size")
    parser.add_argument("--n_sequences", type=int, required=True, help="Number of sequences to generate")
    parser.add_argument("--max_len", type=int, required=True, help="Max length seq")
    parser.add_argument("--min_len", type=int, default=3, help="Min length seq")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of dataset")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--model_path", required=True)

    args = parser.parse_args()
    rnn = RNN(input_size=args.n_tokens+1, hidden_size=args.hidden_size,
               out_size=args.n_tokens,
                out_act=lambda x: x, use_gru=True, learn_init=False)
    rnn.load_state_dict(torch.load(args.model_path))
    generator = TranscoderDataGenerator(rnn)
    update_dataset, hidden_dataset = generator.generate_transcoder_dataset(n_tokens=args.n_tokens,
                                                    n_sequences=args.n_sequences,
                                                    batch_size=args.batch_size,
                                                    max_len=args.max_len,
                                                    min_len=args.min_len)
    os.makedirs("data/copy_transcoder/", exist_ok=True)
    torch.save(update_dataset, f"data/copy_transcoder/{args.dataset_name}_update_gate.pt")
    torch.save(hidden_dataset, f"data/copy_transcoder/{args.dataset_name}_hctx.pt")
