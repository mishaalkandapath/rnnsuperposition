# signal_manager.py
import os
import sys
import signal
from typing import Optional, Any, Literal
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

class SignalManager:
    """Centralized signal handler for graceful interruption handling."""
    
    def __init__(self):
        self.interrupted = False
        self.train_obj_global: Optional[Any] = None
        self.run_name_global: Optional[str] = None
        self._handler_registered = False
    
    def register_handler(self):
        """Register the signal handler for SIGINT (Ctrl+C)."""
        if not self._handler_registered:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self._handler_registered = True
    
    def set_training_context(self, train_obj: Any, run_name: str):
        """Set the training object and run name for saving."""
        self.train_obj_global = train_obj
        self.run_name_global = run_name
    
    def clear_training_context(self):
        """Clear the training context."""
        self.train_obj_global = None
        self.run_name_global = None
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signal (Ctrl+C)."""
        print("\n\nReceived interrupt signal (Ctrl+C). Saving model and exiting gracefully...")
        self.interrupted = True
        
        if self.train_obj_global is not None and self.run_name_global is not None:
            try:
                os.makedirs(self.run_name_global, exist_ok=True)
                save_path = f"{self.run_name_global}/interrupted_{self.run_name_global.split('/')[-1]}.ckpt"
                torch.save(self.train_obj_global.state_dict(), save_path)
                print(f"Model saved to: {save_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
        
        print("Exiting...")
        sys.exit(0)


@dataclass
class CopySuperPosConfig:
    n_inst: int
    n_features: int = 5
    d_hidden: int = 2
    copy_length: int = 3
    batch_size: int = 32
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0
    feat_mag_distn: Literal["unif", "normal"] = "unif"
    gru: bool = False

@dataclass
class CopyConfig:
    run_name: str
    n_tokens: int = 30
    d_hidden: int = 2
    max_len: int = 9
    min_len: int = 3
    batch_size: int = 32
    n_layers: int = 1
    gru: bool = False
    ctd_from: str = None
    data_path: str = None


def normalize_batch(x, eps=1e-8):
    """Normalize per batch to zero mean, unit variance."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(eps)
    return (x - mean) / std

# Learning rate schedulers
def linear_lr(step, steps):
    return 1 - (step / steps)

def constant_lr(*_):
    return 1.0

def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))

def mse_loss(pred, target, importances=1):
    """MSE loss function"""
    return (importances * (pred - target) ** 2).mean()

def cross_entropy_loss(pred, target, importances=1):
    return nn.functional.cross_entropy(pred.transpose(1, 2), target,
                                       ignore_index=-100)