"""
This file tests the MNIST training routine to ensure that the training
process runs correctly, logs necessary metrics, and handle the config
for epochs.
"""

import pytest
from src import train, data_loader

def test_training_run():
    """
    Ensures that:
    1. The training function returns a valid history object with 'loss' and 'accuracy'.
    2. The number of epochs in the history matches the config.
    """
    # Run the training (which loads MNIST, builds a model, and trains)
    history = train.train()
    
    # 1. Check that training history includes 'loss' and 'accuracy' keys
    assert 'loss' in history.history, "Training history missing 'loss' key"
    assert 'accuracy' in history.history, "Training history missing 'accuracy' key"
    
    # 2. Ensure the correct number of epochs were run, based on config
    config = data_loader.load_config()
    expected_epochs = config['training'].get('epochs', 10)
    assert len(history.history['loss']) == expected_epochs, (
        "Number of epochs in history doesn't match config"
    )
