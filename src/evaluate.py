# src/evaluate.py
import tensorflow as tf
import matplotlib.pyplot as plt
from src import data_loader, utils
import yaml

def load_config(config_path='src/config.yaml'):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate():
    config = load_config()
    x_train, y_train, x_test, y_test = data_loader.load_cifar10_data(config)
    
    # Load the saved model
    cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
    loss, accuracy = cnn_model.evaluate(x_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    
    # Predict and visualize a few test images with predictions
    predictions = cnn_model.predict(x_test)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_test[i])
        plt.title("Pred: {}".format(predictions[i].argmax()))
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    evaluate()
