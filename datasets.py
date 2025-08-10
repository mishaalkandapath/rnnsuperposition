import torch
from torch.utils.data import Dataset

def generate_sparse_copyset(n_features, feature_prob, copy_length, batch_size):
    """Generate sparse copy dataset"""
    feature_prob = 1 - math.pow(1 - feature_prob, 1/copy_length)
    batch_shape = (batch_size, copy_length, n_features)
    feat_mag = torch.rand(batch_shape)
    feat_seeds = torch.rand(batch_shape)
    return torch.where(feat_seeds <= feature_prob, feat_mag, 0.0)