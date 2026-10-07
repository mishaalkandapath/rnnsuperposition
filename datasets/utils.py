import os
from typing import Optional, Tuple
from collections import defaultdict
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader, Subset

class IndexedDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices.tolist()
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx]
    
class ConsolidatedStackDataset(Dataset):
    def __init__(self, datasets):
        self.all_data = []
        
        for dataset in datasets:
            for data in dataset:
                    self.all_data.append(data)

    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        return self.all_data[idx]
    
def row_sequence_ids(dataset) -> Optional[torch.Tensor]:
    """Per-row source-sequence ids, or None if any part of the dataset lacks them.

    Reads the stored tensors directly (StackDataset keeps keyword datasets in
    ``.datasets``) instead of indexing millions of rows one at a time.
    """
    parts = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    ids = []
    for part in parts:
        columns = getattr(part, "datasets", None)
        if not isinstance(columns, dict) or "seq_id" not in columns:
            return None
        ids.append(torch.as_tensor(columns["seq_id"]))
    return torch.cat(ids)


def create_transcoder_dataloaders(dataset: ConsolidatedStackDataset,
                                batch_size: int = 256,
                                train_split: float = 0.9,
                                shuffle: bool = True,
                                num_workers: int = None,
                                split_seed: int = None,
                                return_val_sequence_ids: bool = False):
    """
    Create train/val dataloaders for transcoder training

    Args:
        dataset: Output from generate_transcoder_dataset
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training
        shuffle: Whether to shuffle the data
        return_val_sequence_ids: Also return the held-out sequence ids
            (None when the split had to fall back to individual rows)

    Returns:
        train_loader, val_loader[, val_sequence_ids]
    """

    n_samples = len(dataset)
    print(f"-- Dataset is of length {n_samples}---")

    split_generator = None
    if split_seed is not None:
        split_generator = torch.Generator().manual_seed(split_seed)

    # Split by source sequence when rows record one: all timesteps of a
    # held-out sequence go to validation, so val measures unseen sequences
    # rather than neighbouring timesteps of training sequences.
    sequence_ids = row_sequence_ids(dataset)
    val_sequence_ids = None
    if sequence_ids is not None:
        unique_ids = torch.unique(sequence_ids)
        order = (torch.randperm(len(unique_ids), generator=split_generator)
                 if shuffle else torch.arange(len(unique_ids)))
        n_train_sequences = int(len(unique_ids) * train_split)
        val_sequence_ids = unique_ids[order[n_train_sequences:]]
        is_val = torch.isin(sequence_ids, val_sequence_ids)
        train_indices = torch.nonzero(~is_val).squeeze(1)
        val_indices = torch.nonzero(is_val).squeeze(1)
        print(f"-- Split by sequence: {n_train_sequences} train / "
              f"{len(val_sequence_ids)} val sequences")
    else:
        print("-- WARNING: dataset has no seq_id column; splitting by row, so "
              "validation shares sequences with training. Regenerate the "
              "dataset to split by sequence.")
        n_train = int(n_samples * train_split)
        indices = (torch.randperm(n_samples, generator=split_generator)
                   if shuffle else torch.arange(n_samples))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    if num_workers is None:
        num_workers = min(64, len(os.sched_getaffinity(0)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=num_workers, persistent_workers=num_workers > 0
    )
    print(f"-- Using {num_workers} DataLoader workers")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=num_workers > 0
    )

    if return_val_sequence_ids:
        return train_loader, val_loader, val_sequence_ids
    return train_loader, val_loader
