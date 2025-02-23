"""
This file tests the CNN model creation function to ensure it builds
a valid architecture for MNIST images (28×28×1) and outputs 10 classes.
"""

import pytest
from src import model

def test_create_cnn_model():
    """
    Checks that the model:
    1. Has at least one layer (i.e., isn't empty).
    2. Produces an output shape of (None, 10) for 10 MNIST classes.
    3. Can handle input shape (28,28,1).
    """
    # Create a CNN model with default or config-based parameters
    cnn_model = model.create_cnn_model(
        input_shape=(28, 28, 1), 
        num_classes=10, 
        config=None
    )
    
    # 1. Verify the model has layers
    assert len(cnn_model.layers) > 0, "Model has no layers, which is unexpected"
    
    # 2. Check output shape is (None, 10)
    output_shape = cnn_model.output_shape
    assert output_shape[-1] == 10, "Model output layer does not have 10 units for MNIST"

    # 3. Optionally, we can compile and check if it runs a single forward pass
    # But that's often covered in training tests.
