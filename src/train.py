from src.data_loader import load_mnist_data, load_config
from src import model
import tensorflow as tf
import mlflow
import mlflow.tensorflow

def train():
    """
    Trains a Convolutional Neural Network (CNN) on the MNIST dataset using parameters
    from a configuration file. Automatically logs key metrics and parameters to MLflow.
    
    This function will:
      1. Loads MNIST images (28×28, single-channel) and their labels.
      2. Builds a CNN based on config-driven hyperparameters (dropout, filters, etc.).
      3. Compiles the model with Adam and a specified learning rate.
      4. Trains the model for a given number of epochs and batch size.
      5. Logs training details to MLflow for easy experiment tracking.
      6. Saves the final model to 'models/mnist_model.h5'.

    Returns:
        history (tf.keras.callbacks.History): A record of training loss and accuracy
        over each epoch.
    """
    # Read hyperparameters and settings from config.yaml
    config = load_config()

    # Load the MNIST dataset (handwritten digits) into training and test sets
    (x_train, y_train), (x_test, y_test) = load_mnist_data(config)

    # Build a CNN model suitable for single-channel, 28×28 images
    cnn_model = model.create_cnn_model(
        input_shape=(28, 28, 1),
        num_classes=10,
        config=config
    )

    # Extract training parameters from config, providing defaults if missing
    initial_lr = config['training'].get('learning_rate', 0.001)
    epochs = config['training'].get('epochs', 10)
    batch_size = config['training'].get('batch_size', 32)

    # Compile the model with Adam and our chosen learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    cnn_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Enable automatic logging of training metrics (loss, accuracy) to MLflow
    mlflow.tensorflow.autolog()
    mlflow.start_run()
    mlflow.log_param("dataset", "mnist")  # Indicate we're using MNIST
    mlflow.log_param("initial_lr", initial_lr)  # Track the learning rate used

    # Train the model with the given batch size, epochs, and validation data
    history = cnn_model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test)
    )

    # End the MLflow run so metrics and parameters are saved
    mlflow.end_run()

    # Save the trained model for future inference or evaluation
    cnn_model.save("models/mnist_model.h5")
    return history
