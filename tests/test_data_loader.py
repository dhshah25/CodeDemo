# tests/test_data_loader.py
import pytest
from src import data_loader

def test_load_cifar10_data():
    config = data_loader.load_config()
    x_train, y_train, x_test, y_test = data_loader.load_cifar10_data(config)
    # Check that training and test sets are non-empty
    assert x_train.shape[0] > 0
    assert x_test.shape[0] > 0
    # Ensure labels are one-hot encoded (10 classes)
    assert y_train.shape[1] == 10
