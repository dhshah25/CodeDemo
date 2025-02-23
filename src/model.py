# src/model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, Activation
)

def create_cnn_model(input_shape=(28, 28, 1), num_classes=10, config=None):
    """
    This is to build a simple Convolutional Neural Network (CNN) designed for image data, the
    configuration for dropout rates and filter sizes is optional.
    
    By default, this setup works well for MNIST-like images of shape (28, 28, 1).
    We are using:
      - Two convolutional blocks (Conv + BatchNorm + ReLU + MaxPool + Dropout)
      - A final set of Dense layers for classification
    
    Args:
        input_shape (tuple): Dimensions of the input data (height, width, channels).
            For MNIST, this is (28, 28, 1).
        num_classes (int): Number of output categories. MNIST has 10 digits (0–9).
        config (dict, optional): A dictionary that can override default hyperparameters
            like dropout_rate and num_filters. Example keys:
                - config['model']['dropout_rate']
                - config['model']['num_filters']
    
    Returns:
        tensorflow.keras.Model: A compiled CNN model ready for training.
    """
    if config is None:
        dropout_rate = 0.5
        num_filters = [32, 64]
    else:
        # Read user-defined dropout and filter sizes from config,
        # or use the default one, if they're not present.
        dropout_rate = config['model'].get('dropout_rate', 0.5)
        num_filters = config['model'].get('num_filters', [32, 64])

    model = Sequential()

    # Convolutional Block 1
    # First convolutional layer: extracts basic features (edges, shapes)
    model.add(Conv2D(num_filters[0], (3, 3), padding='same', input_shape=input_shape))
    # Normalize activations to help with faster, more stable training
    model.add(BatchNormalization())
    # Non-linear activation to introduce complexity
    model.add(Activation('relu'))
    # Downsample spatial dimensions, reducing computations
    model.add(MaxPooling2D((2, 2)))
    # Randomly drop neurons to reduce overfitting
    model.add(Dropout(dropout_rate))

    # Convolutional Block 2
    # Second conv layer: learns more complex features
    model.add(Conv2D(num_filters[1], (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))

    # Dense Layers
    # It will flattens the feature maps to feed into a fully connected layer
    model.add(Flatten())
    # A small Dense layer for further feature processing
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))

    # Here, the final classification layer with softmax for multi-class output
    model.add(Dense(num_classes, activation='softmax'))

    return model

if __name__ == "__main__":
    from src.data_loader import load_config
    config = load_config()
    cnn_model = create_cnn_model(config=config)
    cnn_model.summary()
