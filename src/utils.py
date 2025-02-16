# src/utils.py
import numpy as np
import tensorflow as tf
import random

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    print(f"Seed set to {seed}")
