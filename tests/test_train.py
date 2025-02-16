# tests/test_train.py
import pytest
from src import train, data_loader

def test_training_run():
    history = train.train()
    # Check that training history includes loss and accuracy
    assert 'loss' in history.history
    assert 'accuracy' in history.history
    # Ensure training ran for the specified number of epochs
    config = data_loader.load_config()
    assert len(history.history['loss']) == config['training']['epochs']
