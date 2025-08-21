from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import string
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, StackDataset
import numpy as np
from tqdm import tqdm

from models.rnn import RNN
from models.transcoders import Transcoder
torch.serialization.add_safe_globals([StackDataset])

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
        
        self.rnn_model.eval()
        self.update_transcoder.eval()
        self.hidden_transcoder.eval()
        
        # Create token mapping (26 letters + 4 numbers + 1 delimiter)
        self.tokens = list(string.ascii_lowercase) + ['0', '1', '2', '3'] + ['<DEL>']
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        
        self.feature_activations = {
            'update': defaultdict(lambda:defaultdict(lambda: defaultdict(list))),  # feature_idx -> sequence -> list of (positions, magnitudes)
            'hidden': defaultdict(lambda:defaultdict(lambda: defaultdict(list)))   # feature_idx -> sequence -> list of (positions, magnitudes)
        }
        
    def one_hot_to_tokens(self, one_hot_seq: torch.Tensor, mask: torch.Tensor) -> List[str]:
        """Convert one-hot sequence to readable tokens"""
        indices = one_hot_seq.argmax(dim=-1)
        tokens = []
        for i, idx in enumerate(indices):
            if mask[i]:  # Only include non-padded positions
                tokens.append(self.idx_to_token[idx.item()])
        return tokens
    
    def make_tokens(self, batch: Dict[str, torch.Tensor]) -> None:
        ins = batch["inputs"]
        tokens = []
        for inp in ins:
            tokens.append([self.idx_to_token[one_hot.argmax().item()] for one_hot in inp])
        return tokens
    
    def analyze_batch_activations(self, 
                                 batch: Dict[str, torch.Tensor]) -> None:
        """
        Analyze feature activations for a batch of sequences of the same length
        
        Args:
            batch_sequences: (batch_size, seq_len, n_tokens) one-hot sequences
            batch_masks: (batch_size, seq_len) masks indicating valid positions
            batch_tokens: List of token sequences for each batch item
        """
        with torch.no_grad():
            seq_len = batch["inputs"].size(1)
            for t in range(seq_len):
                valid_h_prev = batch["h_prevs"][:, t].to(self.device)
                valid_x_t = batch["inputs"][:, t].to(self.device)
                valid_r_t = batch["r_ts"][:, t].to(self.device)
                
                # Prepare transcoder inputs
                update_gate_input = torch.cat([valid_h_prev, valid_x_t], dim=1)
                gated_hidden = valid_r_t * valid_h_prev
                hidden_context_input = torch.cat([gated_hidden, valid_x_t], dim=1)
                
                # Get feature activations from transcoders
                _, update_activations, _ = self.update_transcoder(update_gate_input)
                nonzero_update_acts = torch.nonzero(update_activations,
                                                     as_tuple=False)
                
                # Hidden context transcoder  
                _, hidden_activations, _ = self.hidden_transcoder(hidden_context_input)  # (n_valid, n_features)
                nonzero_hidden_acts = torch.nonzero(hidden_activations,
                                                     as_tuple=False)
                batch_keys = {}
                # Store activations for each sequence and feature
                for idx in range(nonzero_update_acts.shape[0]):
                    batch_idx, feat_idx = nonzero_update_acts[idx, 0].item(), nonzero_update_acts[idx, 1].item()
                    activation_magnitude = update_activations[batch_idx, feat_idx].item()
                    if batch_idx not in batch_keys:
                        batch_keys[batch_idx] = tuple(batch["inputs"][batch_idx].argmax(-1).tolist()) 
                    sequence_tokens = batch_keys[batch_idx]
                    self.feature_activations['update'][feat_idx][sequence_tokens]["positions"].append(t)
                    self.feature_activations['update'][feat_idx][sequence_tokens]["magnitudes"].append(activation_magnitude)
                for idx in range(nonzero_hidden_acts.shape[0]):
                    batch_idx, feat_idx = nonzero_hidden_acts[idx, 0].item(), nonzero_hidden_acts[idx, 1].item()
                    activation_magnitude = hidden_activations[batch_idx, feat_idx].item()
                    if batch_idx not in batch_keys:
                        batch_keys[batch_idx] = tuple(batch["inputs"][batch_idx].argmax(-1).tolist()) 
                    sequence_tokens = batch_keys[batch_idx]
                    self.feature_activations['hidden'][feat_idx][sequence_tokens]["positions"].append(t)
                    self.feature_activations['hidden'][feat_idx][sequence_tokens]["magnitudes"].append(activation_magnitude)
                                
    def analyze_all_sequences(self,
                            n_tokens: int = 30,  # 26 + 4 (delim added later)
                            sequences_per_length: int = 1000,
                            min_len: int = 2,
                            max_len: int = 9,
                            batch_size: int = 8192,
                            cache=[]) -> None:
        """
        Analyze feature activations across all sequence lengths
        
        Args:
            n_tokens: Total number of tokens in vocabulary
            sequences_per_length: Number of sequences to generate per length
            min_len: Minimum sequence length
            max_len: Maximum sequence length
            batch_size: Batch size for processing
        """
        print("Accumulating sequences...")
        sequences_by_length = {}
        for idx, path in enumerate(cache):
            sequences_by_length[idx] = torch.load(path) 

        print("Analyzing Features...")
        for seq_len in range(min_len, max_len + 1):
            sequences = sequences_by_length[seq_len - min_len]
            dataloader = DataLoader(sequences, batch_size, 
                                    num_workers=len(os.sched_getaffinity(0)), persistent_workers=True)
            for batch in tqdm(dataloader, desc=f"Length {seq_len}"):
                self.analyze_batch_activations(batch)
                
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
        
        total_activations = sum(len(activations[entry]['positions']) for entry in activations)
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
            total_activations = sum(len(activations[entry]['positions']) for entry in activations)
            feature_counts.append((feature_idx, total_activations))
            
        feature_counts.sort(key=lambda x: x[1], reverse=True)
        
        return feature_counts[:top_k]

def collate_fn(batch):
  keys = list(batch[0].keys())
  return {
      k: torch.stack([x[k] for x in batch]) for k in keys
}

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_feats_hidden", type=int, required=True)
    parser.add_argument("--n_feats_update", type=int, required=True)
    parser.add_argument("--update_transcoder_path", required=True)
    parser.add_argument("--hidden_transcoder_path", required=True)
    parser.add_argument("--rnn_path", required=True)
    parser.add_argument("--cached_sentences", nargs="+", default=None)

    args = parser.parse_args()
    rnn_model = RNN(input_size=31, hidden_size=128, out_size=30, 
                    use_gru=True, num_layers=1)
    update_transcoder = Transcoder(input_size=159, out_size=128, 
                                   n_feats=args.n_feats_update)
    hidden_transcoder = Transcoder(input_size=159, out_size=128, 
                                   n_feats=args.n_feats_hidden)
    
    rnn_model.load_state_dict(torch.load(args.rnn_path))
    update_transcoder.load_state_dict(torch.load(args.update_transcoder_path)["transcoder"])
    hidden_transcoder.load_state_dict(torch.load(args.hidden_transcoder_path)["transcoder"])
    
    analyzer = FeatureActivationAnalyzer(
        rnn_model=rnn_model,
        update_transcoder=update_transcoder, 
        hidden_transcoder=hidden_transcoder,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    
    analyzer.analyze_all_sequences(
        n_tokens=30,
        sequences_per_length=200,
        min_len=3,
        max_len=9,
        batch_size=4096,
        cache=args.cached_sentences if args.cached_sentences else []
    )
    # try:
    #     top_update_features = analyzer.get_most_active_features('update', top_k=20)
    #     top_hidden_features = analyzer.get_most_active_features('hidden', top_k=20)
        
    #     print("Top Update Gate Features:")
    #     for feature_idx, count in top_update_features:
    #         summary = analyzer.get_feature_summary('update', feature_idx)
    #         print(f"Feature {feature_idx}: {count} activations across {summary['n_sequences']} sequences")
    # except Exception as e:
    #     print(e)
    #     print("gotta work on this")

    # save feature data
    os.makedirs("/w/nobackup/436/lambda/data/copy_transcoder_features/", exist_ok=True)
    with open(f"/w/nobackup/436/lambda/data/copy_transcoder_features/h{args.n_feats_hidden}_u{args.n_feats_update}_features.p"):
        pickle.dump(analyzer.feature_activations)