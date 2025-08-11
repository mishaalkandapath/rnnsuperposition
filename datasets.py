import torch
from torch.utils.data import TensorDataset
torch.serialization.add_safe_globals([TensorDataset])
import math

def generate_sparse_copyset(n_features, feature_prob, copy_length, batch_size):
    """Generate sparse copy dataset"""
    feature_prob = 1 - math.pow(1 - feature_prob, 1/copy_length)
    batch_shape = (batch_size, copy_length, n_features)
    feat_mag = torch.rand(batch_shape)
    feat_seeds = torch.rand(batch_shape)
    return torch.where(feat_seeds <= feature_prob, feat_mag, 0.0)

def generate_token_copyset(n_tokens, batch_size, max_len, min_len=2):
    batch_size, max_len = int(batch_size), int(max_len)
    sequence_indices = torch.randint(0, n_tokens, 
                                            (batch_size, max_len))
    sequence_one_hot = torch.nn.functional.one_hot(sequence_indices, 
                                                    num_classes=n_tokens).float()
    sequence_lengths = torch.randint(min_len, max_len+1, (batch_size,))
    length_mask = torch.arange(max_len).unsqueeze(0) < sequence_lengths.unsqueeze(1)
    sequence_one_hot *= length_mask.unsqueeze(-1)
    return sequence_one_hot, length_mask

def generate_unique_test_set(n_tokens, test_size, 
                             max_len, min_len, train_indices):
    # Convert train sequences (int form) to a set for fast lookup
    train_set = {tuple(row.tolist()) for row in train_indices}
    seen = set(train_set)  # start with all train sequences

    test_indices = []
    test_masks = []

    while len(test_indices) < test_size:
        seq = torch.randint(0, n_tokens, (max_len,))
        seq_len = torch.randint(min_len, max_len+1, (1,))
        mask = torch.arange(max_len) < seq_len
        seq = seq * mask  # zero out pads if needed
        tup = tuple(seq.tolist())

        if tup not in seen:
            seen.add(tup)
            test_indices.append(seq)
            test_masks.append(mask)

    test_indices = torch.stack(test_indices)
    test_masks = torch.stack(test_masks)
    test_one_hot = torch.nn.functional.one_hot(test_indices, num_classes=n_tokens).float()

    return test_one_hot, test_masks

def generate_token_copy_dataset(n_tokens, train_size, test_size, 
                                max_len, min_len=2):
    train_seq_one_hot, train_loss_mask = generate_token_copyset(n_tokens,
                                                                train_size,
                                                                max_len,
                                                                min_len=min_len
                                                                )
    train_seq_indices = train_seq_one_hot.argmax(-1)
    test_seq_one_hot, test_loss_mask = generate_unique_test_set(n_tokens,
                                                                test_size,
                                                                max_len,
                                                                min_len, 
                                                                train_seq_indices)
    
    return TensorDataset(train_seq_one_hot, train_loss_mask), TensorDataset(test_seq_one_hot, test_loss_mask)
