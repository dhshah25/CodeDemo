# src/model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_cnn_model(input_shape=(32, 32, 3), num_classes=10, config=None):
    """Create a simple CNN model for CIFAR-10 classification."""
    if config is None:
        dropout_rate = 0.5
        num_filters = [32, 64]
    else:
        dropout_rate = config['model'].get('dropout_rate', 0.5)
        num_filters = config['model'].get('num_filters', [32, 64])
    
    model = Sequential()
    model.add(Conv2D(num_filters[0], (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(num_filters[1], (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    return model

if __name__ == "__main__":
    from src.data_loader import load_config
    config = load_config()
    cnn_model = create_cnn_model(config=config)
    cnn_model.summary()
