# tests/test_model.py
import pytest
from src import model

def test_create_cnn_model():
    cnn_model = model.create_cnn_model()
    # Verify that the model has layers and an output layer with 10 units for CIFAR-10
    assert len(cnn_model.layers) > 0
    output_shape = cnn_model.output_shape
    assert output_shape[-1] == 10
