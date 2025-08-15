from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import string

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from models.rnn import RNN
from models.transcoders import Transcoder
from training.train_copy import inference_generate, add_delimiter_dimension
class FeatureActivationAnalyzer:
    """Analyze feature activations across sequences for RNN transcoders"""
    
    def __init__(self, 
                 rnn_model: nn.Module,
                 update_transcoder: nn.Module,
                 hidden_transcoder: nn.Module,
                 device: str = 'cuda'):
        """
        Args:
            rnn_model: Trained RNN model
            update_transcoder: Trained update gate transcoder
            hidden_transcoder: Trained hidden context transcoder
            device: Device to run analysis on
        """
        self.rnn_model = rnn_model.to(device)
        self.update_transcoder = update_transcoder.to(device)
        self.hidden_transcoder = hidden_transcoder.to(device)
        self.device = device
        
        # Set models to eval mode
        self.rnn_model.eval()
        self.update_transcoder.eval()
        self.hidden_transcoder.eval()
        
        # Create token mapping (26 letters + 4 numbers + 1 delimiter)
        self.tokens = list(string.ascii_lowercase) + ['0', '1', '2', '3'] + ['<DEL>']
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        
        # Storage for feature activations
        self.feature_activations = {
            'update': defaultdict(defaultdict(list)),  # feature_idx -> list of (sequence, positions, magnitudes)
            'hidden': defaultdict(defaultdict(list))   # feature_idx -> list of (sequence, positions, magnitudes)
        }
        
    def one_hot_to_tokens(self, one_hot_seq: torch.Tensor, mask: torch.Tensor) -> List[str]:
        """Convert one-hot sequence to readable tokens"""
        indices = one_hot_seq.argmax(dim=-1)
        tokens = []
        for i, idx in enumerate(indices):
            if mask[i]:  # Only include non-padded positions
                tokens.append(self.idx_to_token[idx.item()])
        return tokens
        
    def generate_sequences_by_length(self, 
                                   n_tokens: int,
                                   sequences_per_length: int,
                                   min_len: int = 2,
                                   max_len: int = 9) -> Dict[int, List[Tuple[torch.Tensor, torch.Tensor, List[str]]]]:
        """
        Generate sequences organized by length
        
        Returns:
            Dict mapping length -> list of (one_hot_seq, mask, token_list) tuples
        """
        sequences_by_length = defaultdict(list)
        
        for seq_len in range(min_len, max_len + 1):
            print(f"Generating {sequences_per_length} sequences of length {seq_len}...")
            
            generated = 0
            seen_sequences = set()
            
            while generated < sequences_per_length:
                # Generate random sequence of exact length
                indices = torch.randint(0, n_tokens, (seq_len,))
                
                # Create mask (all True for this length)
                mask = torch.ones(seq_len, dtype=torch.bool)
                
                # Convert to tuple for set membership
                seq_tuple = tuple(indices.tolist())
                
                if seq_tuple not in seen_sequences:
                    seen_sequences.add(seq_tuple)
                    
                    # Create one-hot encoding
                    one_hot = torch.nn.functional.one_hot(indices, num_classes=n_tokens).float()
                    
                    # Convert to token list
                    tokens = tuple(self.idx_to_token[idx.item()] for idx in indices)
                    
                    sequences_by_length[seq_len].append((one_hot, mask, tokens))
                    generated += 1
                    
        return sequences_by_length
        
    def analyze_batch_activations(self, 
                                batch_sequences: torch.Tensor,
                                batch_masks: torch.Tensor,
                                batch_tokens: List[List[str]]) -> None:
        """
        Analyze feature activations for a batch of sequences of the same length
        
        Args:
            batch_sequences: (batch_size, seq_len, n_tokens) one-hot sequences
            batch_masks: (batch_size, seq_len) masks indicating valid positions
            batch_tokens: List of token sequences for each batch item
        """
        batch_size, seq_len, _ = batch_sequences.shape
        batch_sequences = batch_sequences.to(self.device)
        
        with torch.no_grad():
            # Run RNN to get internal states
            outs, _, r_records, z_records, h_new_records, h_records = inference_generate(self.rnn_model, batch_sequences,
                                discrete=True, record_gates=True)
            
            # Extract data from single layer (index 0)
            r_t = r_records[0]  # (batch_size, seq_len, hidden_size)
            z_t = z_records[0]  # (batch_size, seq_len, hidden_size)  
            h_new_t = h_new_records[0]  # (batch_size, seq_len, hidden_size)
            h_prev = h_records[0]  # (batch_size, seq_len, hidden_size)
            outs = add_delimiter_dimension(outs, concat_del=False)
            batch_sequences = add_delimiter_dimension(batch_sequences)
            x_t = torch.cat([batch_sequences, outs],
                                dim=1)   # (batch_size, 2*seq_len, input_size)
            
            # Process each timestep
            for t in range(2*seq_len):
                valid_h_prev = h_prev[:, t]  # (n_valid, hidden_size)
                valid_x_t = x_t[:, t]      # (n_valid, input_size)
                valid_r_t = r_t[:, t]      # (n_valid, hidden_size)
                
                # Prepare transcoder inputs
                update_gate_input = torch.cat([valid_h_prev, valid_x_t], dim=1)
                gated_hidden = valid_r_t * valid_h_prev
                hidden_context_input = torch.cat([gated_hidden, valid_x_t], dim=1)
                
                # Get feature activations from transcoders
                _, update_activations, _ = self.update_transcoder(update_gate_input)
                nonzero_update_acts = torch.nonzero(update_activations, as_tuple=False)
                
                # Hidden context transcoder  
                _, hidden_activations, _ = self.hidden_transcoder(hidden_context_input)  # (n_valid, n_features)
                nonzero_hidden_acts = torch.nonzero(hidden_activations, as_tuple=False)
                
                # Store activations for each sequence and feature
                for idx in range(nonzero_update_acts.shape[0]):
                    batch_idx, feat_idx = nonzero_update_acts[idx, 0].item(), nonzero_update_acts[idx, 1].item()
                    activation_magnitude = update_activations[batch_idx, feat_idx].item()
                    sequence_tokens = batch_tokens[batch_idx]
                    self.feature_activations['update'][feat_idx][sequence_tokens].append({
                                    'positions': [t],
                                    'magnitudes': [activation_magnitude]
                                })
                for idx in range(nonzero_hidden_acts.shape[0]):
                    batch_idx, feat_idx = nonzero_hidden_acts[idx, 0].item(), nonzero_hidden_acts[idx, 1].item()
                    activation_magnitude = hidden_activations[batch_idx, feat_idx].item()
                    sequence_tokens = batch_tokens[batch_idx]
                    self.feature_activations['hidden'][feat_idx][sequence_tokens].append({
                                    'positions': [t],
                                    'magnitudes': [activation_magnitude]
                                })
                                
    def analyze_all_sequences(self,
                            n_tokens: int = 31,  # 26 + 4 + 1
                            sequences_per_length: int = 1000,
                            min_len: int = 2,
                            max_len: int = 9,
                            batch_size: int = 32) -> None:
        """
        Analyze feature activations across all sequence lengths
        
        Args:
            n_tokens: Total number of tokens in vocabulary
            sequences_per_length: Number of sequences to generate per length
            min_len: Minimum sequence length
            max_len: Maximum sequence length
            batch_size: Batch size for processing
        """
        print("Generating sequences by length...")
        sequences_by_length = self.generate_sequences_by_length(
            n_tokens, sequences_per_length, min_len, max_len
        )
        
        print("Analyzing feature activations...")
        for seq_len in range(min_len, max_len + 1):
            sequences = sequences_by_length[seq_len]
            print(f"Processing {len(sequences)} sequences of length {seq_len}...")
            
            # Process in batches
            for batch_start in tqdm(range(0, len(sequences), batch_size), 
                                  desc=f"Length {seq_len}"):
                batch_end = min(batch_start + batch_size, len(sequences))
                batch = sequences[batch_start:batch_end]
                
                # Extract batch components
                batch_sequences = torch.stack([seq[0] for seq in batch])
                batch_masks = torch.stack([seq[1] for seq in batch])
                batch_tokens = [seq[2] for seq in batch]
                
                # Analyze this batch
                self.analyze_batch_activations(batch_sequences, batch_masks, batch_tokens)
                
        print("Feature activation analysis complete!")
        
    def get_feature_summary(self, transcoder_type: str, feature_idx: int) -> Dict:
        """
        Get summary statistics for a specific feature
        
        Args:
            transcoder_type: 'update' or 'hidden'
            feature_idx: Index of the feature to analyze
            
        Returns:
            Dictionary with feature statistics
        """
        if feature_idx not in self.feature_activations[transcoder_type]:
            return {"n_sequences": 0, "n_activations": 0}
            
        activations = self.feature_activations[transcoder_type][feature_idx]
        
        total_activations = sum(len(entry['positions']) for entry in activations)
        total_sequences = len(activations)
        
        # Get all positions where this feature activates
        all_positions = []
        all_magnitudes = []
        for entry in activations:
            all_positions.extend(entry['positions'])
            all_magnitudes.extend(entry['magnitudes'])
            
        return {
            "n_sequences": total_sequences,
            "n_activations": total_activations,
            "avg_activations_per_sequence": total_activations / max(total_sequences, 1),
            "position_distribution": np.bincount(all_positions).tolist(),
            "magnitude_stats": {
                "mean": np.mean(all_magnitudes),
                "std": np.std(all_magnitudes), 
                "min": np.min(all_magnitudes),
                "max": np.max(all_magnitudes)
            } if all_magnitudes else None
        }
        
    def get_most_active_features(self, transcoder_type: str, top_k: int = 10) -> List[Tuple[int, int]]:
        """
        Get the most active features by total number of activations
        
        Args:
            transcoder_type: 'update' or 'hidden'
            top_k: Number of top features to return
            
        Returns:
            List of (feature_idx, total_activations) tuples
        """
        feature_counts = []
        
        for feature_idx, activations in self.feature_activations[transcoder_type].items():
            total_activations = sum(len(entry['positions']) for entry in activations)
            feature_counts.append((feature_idx, total_activations))
            
        # Sort by total activations (descending)
        feature_counts.sort(key=lambda x: x[1], reverse=True)
        
        return feature_counts[:top_k]

# Example usage
if __name__ == "__main__":
    # Assume you have trained models
    rnn_model = RNN(input_size=31, hidden_size=64, use_gru=True, num_layers=1)
    update_transcoder = Transcoder(input_size=95, out_size=64, n_feats=512)
    hidden_transcoder = Transcoder(input_size=95, out_size=64, n_feats=512)
    
    # Create analyzer
    analyzer = FeatureActivationAnalyzer(
        rnn_model=rnn_model,
        update_transcoder=update_transcoder, 
        hidden_transcoder=hidden_transcoder,
        device='cuda'
    )
    
    # Run analysis
    analyzer.analyze_all_sequences(
        n_tokens=31,
        sequences_per_length=200,
        min_len=3,
        max_len=8,
        batch_size=64
    )
    
    # Get most active features
    top_update_features = analyzer.get_most_active_features('update', top_k=20)
    top_hidden_features = analyzer.get_most_active_features('hidden', top_k=20)
    
    print("Top Update Gate Features:")
    for feature_idx, count in top_update_features:
        summary = analyzer.get_feature_summary('update', feature_idx)
        print(f"Feature {feature_idx}: {count} activations across {summary['n_sequences']} sequences")