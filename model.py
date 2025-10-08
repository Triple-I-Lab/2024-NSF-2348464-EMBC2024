"""
Stacked CNN model for ECG signal classification
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.metrics import Precision, Recall


def create_stacked_cnn(input_shape, learning_rate=0.001):
    """
    Create stacked CNN architecture for ECG classification
    
    Args:
        input_shape: tuple (timesteps, channels)
        learning_rate: optimizer learning rate
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Conv Block 1
        layers.Conv1D(256, 3, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # Conv Block 2
        layers.Conv1D(128, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # Conv Block 3
        layers.Conv1D(64, 2, activation='relu'),
        layers.BatchNormalization(),
        
        # Conv Block 4
        layers.Conv1D(32, 2, activation='selu'),
        layers.GlobalAveragePooling1D(),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', Precision(), Recall()]
    )
    
    return model