import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import yaml
from . import utils

def load_config(config_path='src/config.yaml'):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_cifar10_data(config=None):
    """Load and preprocess the CIFAR-10 dataset."""
    if config is None:
        config = load_config()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if config['data']['normalize']:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    config = load_config()
    x_train, y_train, x_test, y_test = load_cifar10_data(config)
    print("Training data shape:", x_train.shape)
