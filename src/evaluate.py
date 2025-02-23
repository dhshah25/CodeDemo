# src/evaluate.py

import tensorflow as tf
# We import load_config and load_mnist_data to retrieve settings and data
from src.data_loader import load_config, load_mnist_data

def evaluate():
    """
    Loads the saved MNIST model from 'models/mnist_model.h5' and evaluates
    its performance on the test set. This function:
      1. Retrieves the configuration (which might contain seed or other info).
      2. Loads the MNIST test data (along with training data, though only test is used here).
      3. Loads the pre-trained CNN model from disk.
      4. Prints out the final loss and accuracy on the test set.
      
    It's a simple way to confirm how well your model generalizes after training.
    """
    # Load the configuration settings
    config = load_config()
    
    # Load MNIST data. We'll only need x_test, y_test for final evaluation,
    # but the function returns x_train, y_train as well.
    (x_train, y_train), (x_test, y_test) = load_mnist_data(config)

    # Load the trained model from the 'models' folder
    mnist_model = tf.keras.models.load_model("models/mnist_model.h5")
    
    # Evaluate the model on the test set
    loss, accuracy = mnist_model.evaluate(x_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
