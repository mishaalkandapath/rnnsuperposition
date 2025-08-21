import os
from typing import Tuple
from collections import defaultdict
import torch 
from torch.utils.data import Dataset, DataLoader, Subset

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
    
def create_transcoder_dataloaders(dataset: ConsolidatedStackDataset,
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
    
    n_samples = len(dataset)
    print(f"-- Dataset is of length {n_samples}---")
    n_train = int(n_samples * train_split)
    
    indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=min(64, len(os.sched_getaffinity(0))), persistent_workers=True
    )
    print(f"-- Asked for {min(64, len(os.sched_getaffinity(0)))} workers")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader