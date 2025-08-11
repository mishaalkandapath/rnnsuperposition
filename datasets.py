import torch
from torch.utils.data import Dataset
import math

def generate_sparse_copyset(n_features, feature_prob, copy_length, batch_size):
    """Generate sparse copy dataset"""
    feature_prob = 1 - math.pow(1 - feature_prob, 1/copy_length)
    batch_shape = (batch_size, copy_length, n_features)
    feat_mag = torch.rand(batch_shape)
    feat_seeds = torch.rand(batch_shape)
    return torch.where(feat_seeds <= feature_prob, feat_mag, 0.0)

def generate_token_copyset(n_tokens, batch_size, max_len, min_len=2):
    sequence_indices = torch.randint(0, n_tokens, 
                                            (batch_size, max_len))
    sequence_one_hot = torch.nn.functional.one_hot(sequence_indices, 
                                                    num_classes=n_tokens).float()
    sequence_lengths = torch.randint(min_len, max_len+1, (batch_size,))
    length_mask = torch.arange(max_len).unsqueeze(0) < sequence_lengths.unsqueeze(1)
    sequence_one_hot *= length_mask.unsqueeze(-1)
    return sequence_one_hot, length_mask