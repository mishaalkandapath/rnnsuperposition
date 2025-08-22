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
from circuit.copy_find_features import FeatureActivationAnalyzer, collate_fn, convert_dict
torch.serialization.add_safe_globals([StackDataset])

class RLFeatureActivationAnalyzer(FeatureActivationAnalyzer):
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
        super().__init__(rnn_model, update_transcoder, hidden_transcoder, device)
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
                
                self.analyze_activations(
                                         batch,
                                         valid_h_prev, 
                                         valid_x_t,
                                         valid_r_t,
                                         t)        
                        
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
    new_dict_features = convert_dict(analyzer.feature_activations, "positions",
                                     "magnitudes")
    new_dict_sequences = convert_dict(analyzer.sequence_activations, "features",
                                      "magnitudes")
    os.makedirs("/w/nobackup/436/lambda/data/rl_transcoder_features/", exist_ok=True)
    with open(f"/w/nobackup/436/lambda/data/rl_transcoder_features/h{args.n_feats_hidden}_u{args.n_feats_update}_features.p", "wb") as f:
        pickle.dump(new_dict_features, f)
    with open(f"/w/nobackup/436/lambda/data/rl_transcoder_features/h{args.n_feats_hidden}_u{args.n_feats_update}_sequences.p", "wb") as f:
        pickle.dump(new_dict_sequences, f)