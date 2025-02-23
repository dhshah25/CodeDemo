import tensorflow as tf
from tensorflow.keras.datasets import mnist
import yaml

def load_config(config_path='src/config.yaml'):
    """
    This will reads and returns the project configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML file containing configuration settings.
        
    Returns:
        dict: A dictionary of configuration parameters.

    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_mnist_data(config=None):
    """
    Loads and prepares the MNIST dataset for training and testing.
    
    The MNIST consists of 70,000 grayscale images of handwritten digits (0–9),
    each with size of 28×28. This function reshapes them into the form (28, 28, 1)
    for convolutional networks, scales pixel values to [0, 1], and converts
    labels to one-hot vectors.

    Args:
        config (dict, optional): A configuration dictionary. If None, the default
            configuration is loaded from 'src/config.yaml'. Currently, this function
            doesn't use config for MNIST, but it can be extended to do so if needed.

    Returns:
        (x_train, y_train): Training images and labels, ready for a CNN model.
        (x_test, y_test): Test images and labels, for final evaluation.
    """
    if config is None:
        config = load_config()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape to (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)
