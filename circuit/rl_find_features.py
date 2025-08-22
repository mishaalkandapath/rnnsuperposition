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

class RLFeatureActivationAnalyzer:
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
        
        self.feature_activations = {
            'update': defaultdict(lambda:defaultdict(lambda: defaultdict(list))),  # feature_idx -> sequence -> list of (positions, magnitudes)
            'hidden': defaultdict(lambda:defaultdict(lambda: defaultdict(list)))   # feature_idx -> sequence -> list of (positions, magnitudes)
        }
        self.cur_type=None

    def convert_sequence_to_text(self, 
                                 inputs: torch.Tensor, 
                                 outputs: torch.Tensor) -> List[str]:
        assert inputs.size(0) >= 2
        ret_list = ["prev. reward", "took action", "reached state"] * inputs.size(0)
        types = ["commonp", "common_p", "uncommonp", "uncommon_p"]
        starts = ["high_first", "low_first"]
        for i in range(inputs.size(0)):
            ret_list[3*i]+=f" {inputs[i, 0, -2]}"
            ret_list[3*i + 1]+=f" {inputs[i, 2, 3:6].argmax().item()} w value {outputs[i, 1, -1].item()}"
            ret_list[3*i + 2]+=f" {inputs[i, 2, :3].argmax().item()}"
        ret_list = [f"{starts[self.cur_type[1]]}_{types[self.cur_type[0]]}"] + ret_list
        return tuple(ret_list)

    
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
            seq_len = batch["inputs"].size(1)*3
            for t in range(seq_len):
                valid_h_prev = batch["h_prevs"][:, t//3, t%3].to(self.device)
                valid_x_t = batch["inputs"][:, t//3, t%3].to(self.device)
                valid_r_t = batch["r_ts"][:, t//3, t%3].to(self.device)
                
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
                        
                        batch_keys[batch_idx] = self.convert_sequence_to_text(batch["inputs"][batch_idx], batch["outputs"][batch_idx])
                    sequence_tokens = batch_keys[batch_idx]
                    self.feature_activations['update'][feat_idx][sequence_tokens]["positions"].append(t)
                    self.feature_activations['update'][feat_idx][sequence_tokens]["magnitudes"].append(activation_magnitude)
                for idx in range(nonzero_hidden_acts.shape[0]):
                    batch_idx, feat_idx = nonzero_hidden_acts[idx, 0].item(), nonzero_hidden_acts[idx, 1].item()
                    activation_magnitude = hidden_activations[batch_idx, feat_idx].item()
                    if batch_idx not in batch_keys:
                        batch_keys[batch_idx] = self.convert_sequence_to_text(batch["inputs"][batch_idx], batch["outputs"][batch_idx])
                    sequence_tokens = batch_keys[batch_idx]
                    self.feature_activations['hidden'][feat_idx][sequence_tokens]["positions"].append(t)
                    self.feature_activations['hidden'][feat_idx][sequence_tokens]["magnitudes"].append(activation_magnitude)
                                
    def analyze_all_sequences(self,
                            batch_size: int = 8192,
                            cache=[]) -> None:
        print("Accumulating sequences...")
        sequences_by_type_by_length = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for idx, path in enumerate(cache):
            pattern = (idx%4, idx//4)
            data = torch.load(path, map_location=torch.device("cpu")) 
            for sequence in data:
                n_zero = 0
                assert sequence["inputs"].size(0) == 5
                while torch.all(sequence["inputs"][n_zero].long() == 5):
                    n_zero+=1

                assert n_zero <= 3
                for key in sequence:
                    sequences_by_type_by_length[pattern][5 - n_zero][key].append(sequence[key][n_zero:])

        for seq_typ in sequences_by_type_by_length:
            self.cur_type=seq_typ
            sequences_by_length = sequences_by_type_by_length[seq_typ]
            sequences_by_length = {length: {k: torch.stack(sequences_by_length[length][k]) for k in sequences_by_length[length]} for length in sequences_by_length}
            sequences_by_length = sorted([(length, StackDataset(**sequences_by_length[length])) for length in sequences_by_length], key=lambda x: x[0])
            sequences_by_length = [s[1] for s in sequences_by_length]

            print("Analyzing Features...")
            for seq_len in range(2,6):
                sequences = sequences_by_length[seq_len - 2]
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
            all_positions.extend(activations[entry]['positions'])
            all_magnitudes.extend(activations[entry]['magnitudes'])
            
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
    parser.add_argument("--cached_sequences", nargs="+", default=None)

    args = parser.parse_args()
    rnn_model = RNN(input_size=8, hidden_size=48, out_size=4, 
                    use_gru=True, num_layers=1, learn_init=True)
    update_transcoder = Transcoder(input_size=56, out_size=48, 
                                   n_feats=args.n_feats_update)
    hidden_transcoder = Transcoder(input_size=56, out_size=48, 
                                   n_feats=args.n_feats_hidden)
    
    rnn_model.load_state_dict(torch.load(args.rnn_path))
    update_transcoder.load_state_dict(torch.load(args.update_transcoder_path)["transcoder"])
    hidden_transcoder.load_state_dict(torch.load(args.hidden_transcoder_path)["transcoder"])
    
    analyzer = RLFeatureActivationAnalyzer(
        rnn_model=rnn_model,
        update_transcoder=update_transcoder, 
        hidden_transcoder=hidden_transcoder,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    
    analyzer.analyze_all_sequences(
        batch_size=4096,
        cache=args.cached_sequences if args.cached_sequences else []
    )
    top_update_features = analyzer.get_most_active_features('update', top_k=20)
    top_hidden_features = analyzer.get_most_active_features('hidden', top_k=20)
    
    print("Top Update Gate Features:")
    for feature_idx, count in top_update_features:
        summary = analyzer.get_feature_summary('update', feature_idx)
        print(f"Feature {feature_idx}: {count} activations across {summary['n_sequences']} sequences")

    # save feature data
    new_dict = {}
    for typ in analyzer.feature_activations:
        new_dict[typ] = {}
        for feature in analyzer.feature_activations[typ]:
            new_dict[typ][feature] = {}
            for sequence in analyzer.feature_activations[typ][feature]:
                new_dict[typ][feature][sequence] = {"positions": analyzer.feature_activations[typ][feature][sequence]["positions"],
                                                    "magnitudes":analyzer.feature_activations[typ][feature][sequence]["magnitudes"]}
    os.makedirs("/w/nobackup/436/lambda/data/rl_transcoder_features/", exist_ok=True)
    with open(f"/w/nobackup/436/lambda/data/rl_transcoder_features/h{args.n_feats_hidden}_u{args.n_feats_update}_features.p", "wb") as f:
        pickle.dump(new_dict, f)