import mlflow
import mlflow.tensorflow
from src.data_loader import load_cifar10_data, load_config
from src.model import create_cnn_model
from src import utils
import tensorflow as tf
import yaml

def train():
    config = load_config()
    utils.set_seed(config['training']['seed'])
    if mlflow.active_run() is not None:
        mlflow.end_run()
    # Load data
    x_train, y_train, x_test, y_test = load_cifar10_data(config)
    
    # Create model
    cnn_model = create_cnn_model(config=config)
    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    
    # End any active MLflow run before starting a new one
    

    # Start MLflow run with auto-logging enabled
    mlflow.tensorflow.autolog()
    mlflow.start_run()
    history = cnn_model.fit(
        x_train, y_train,
        epochs=config['training']['epochs'],
        batch_size=config['data']['batch_size'],
        validation_split=0.2
    )
    mlflow.end_run()
    
    # Optionally, save your model
    cnn_model.save("models/cnn_model.h5")
    return history

if __name__ == "__main__":
    train()
