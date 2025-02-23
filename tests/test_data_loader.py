"""
This file tests the MNIST data loading process to ensure it returns the
correct shapes and label formats. We expect 28×28 grayscale images (with
an extra dimension for channels) and one-hot labels for 10 classes.
"""

import pytest
from src import data_loader

def test_load_mnist_data():
    """
    Verifies that MNIST data is loaded and preprocessed properly:
    1. The training and test sets should not be empty.
    2. Each image should be reshaped to (28, 28, 1).
    3. Labels should be one-hot encoded with 10 possible classes.
    """
    config = data_loader.load_config()
    (x_train, y_train), (x_test, y_test) = data_loader.load_mnist_data(config)
    
    # 1. Check that training and test sets are non-empty
    assert x_train.shape[0] > 0, "Training set is empty"
    assert x_test.shape[0] > 0, "Test set is empty"

    # 2. Verify the shape (batch_size, 28, 28, 1)
    assert x_train.shape[1:] == (28, 28, 1), "Training images are not (28,28,1)"
    assert x_test.shape[1:] == (28, 28, 1), "Test images are not (28,28,1)"

    # 3. Check one-hot encoding (10 classes)
    assert y_train.shape[1] == 10, "Training labels should have 10 one-hot classes"
    assert y_test.shape[1] == 10, "Test labels should have 10 one-hot classes"
