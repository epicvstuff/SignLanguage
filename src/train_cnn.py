"""
CNN Model Training Module
Train a Convolutional Neural Network on hand images for ASL alphabet recognition.

Uses MobileNetV2 with transfer learning for efficient training.

For Google Colab usage:
```python
# Install dependencies
!pip install tensorflow kagglehub

# Download dataset
import kagglehub
path = kagglehub.dataset_download("grassknoted/asl-alphabet")

# Train
from train_cnn import train_cnn_model
model, history = train_cnn_model(path, "asl_cnn_model.keras", epochs=20)

# Download model
from google.colab import files
files.download("asl_cnn_model.keras")
```
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')


# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 26  # A-Z


def find_dataset_path(base_path):
    """
    Find the actual training data folder in the Kaggle dataset.
    
    Args:
        base_path: Base path to the downloaded dataset
        
    Returns:
        path: Path to the training data folder
    """
    possible_paths = [
        os.path.join(base_path, "asl_alphabet_train", "asl_alphabet_train"),
        os.path.join(base_path, "asl_alphabet_train"),
        base_path
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "A")):
            return path
    
    raise FileNotFoundError(
        f"Could not find training data in {base_path}\n"
        f"Expected folder structure: asl_alphabet_train/asl_alphabet_train/A, B, C, ..."
    )


def create_data_generators(dataset_path, validation_split=0.2):
    """
    Create training and validation data generators with augmentation.
    
    Args:
        dataset_path: Path to the training data folder
        validation_split: Fraction of data to use for validation
        
    Returns:
        train_generator: Training data generator
        val_generator: Validation data generator
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        horizontal_flip=False,  # Don't flip - ASL signs are hand-specific
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Get only A-Z folders (exclude 'del', 'nothing', 'space')
    classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    print(f"Loading training data from: {dataset_path}")
    print(f"Classes: {classes}")
    
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=classes,
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=classes,
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator


def build_cnn_model(num_classes=NUM_CLASSES, fine_tune_layers=0):
    """
    Build a CNN model using MobileNetV2 transfer learning.
    
    Args:
        num_classes: Number of output classes
        fine_tune_layers: Number of top layers to fine-tune (0 = freeze all)
        
    Returns:
        model: Compiled Keras model
    """
    # Load MobileNetV2 pre-trained on ImageNet
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Optionally unfreeze top layers for fine-tuning
    if fine_tune_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
    
    # Build the model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Preprocessing for MobileNetV2
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, train_generator, val_generator, epochs=20, model_path="asl_cnn_model.keras"):
    """
    Train the CNN model.
    
    Args:
        model: Keras model to train
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs: Number of training epochs
        model_path: Path to save the best model
        
    Returns:
        history: Training history
    """
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print(f"\nTraining for up to {epochs} epochs...")
    print(f"Model will be saved to: {model_path}")
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, val_generator):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained Keras model
        val_generator: Validation data generator
        
    Returns:
        results: Evaluation results (loss, accuracy)
    """
    print("\nEvaluating model...")
    results = model.evaluate(val_generator, verbose=1)
    print(f"Validation Loss: {results[0]:.4f}")
    print(f"Validation Accuracy: {results[1]:.4f}")
    return results


def train_cnn_model(dataset_path, model_path="asl_cnn_model.keras", epochs=20, fine_tune=False):
    """
    Complete training pipeline for CNN model.
    
    Args:
        dataset_path: Path to the Kaggle ASL Alphabet dataset
        model_path: Where to save the trained model
        epochs: Number of training epochs
        fine_tune: Whether to fine-tune the last layers of MobileNetV2
        
    Returns:
        model: Trained model
        history: Training history
    """
    print("="*60)
    print("ASL Alphabet CNN Training")
    print("="*60)
    
    # Find dataset
    train_path = find_dataset_path(dataset_path)
    
    # Create data generators
    print("\nPreparing data generators...")
    train_gen, val_gen = create_data_generators(train_path)
    
    print(f"\nTraining samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Classes: {list(train_gen.class_indices.keys())}")
    
    # Build model
    print("\nBuilding CNN model (MobileNetV2 + custom head)...")
    fine_tune_layers = 20 if fine_tune else 0
    model = build_cnn_model(fine_tune_layers=fine_tune_layers)
    model.summary()
    
    # Train
    history = train_model(model, train_gen, val_gen, epochs, model_path)
    
    # Evaluate
    evaluate_model(model, val_gen)
    
    # Save class indices for inference
    class_indices = train_gen.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    
    # Save class names alongside model
    import json
    class_path = model_path.replace('.keras', '_classes.json').replace('.h5', '_classes.json')
    with open(class_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Class names saved to: {class_path}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to: {model_path}")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.2%}")
    print("="*60)
    
    return model, history


def load_cnn_model(model_path="asl_cnn_model.keras"):
    """
    Load a trained CNN model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        model: Loaded Keras model
        class_names: Dictionary mapping class indices to letter names
    """
    import json
    
    model = keras.models.load_model(model_path)
    
    # Load class names
    class_path = model_path.replace('.keras', '_classes.json').replace('.h5', '_classes.json')
    if os.path.exists(class_path):
        with open(class_path, 'r') as f:
            class_names = json.load(f)
        # Convert string keys back to int
        class_names = {int(k): v for k, v in class_names.items()}
    else:
        # Default to A-Z
        class_names = {i: chr(ord('A') + i) for i in range(26)}
    
    return model, class_names


def predict_image(model, image, class_names):
    """
    Predict the ASL letter from an image.
    
    Args:
        model: Trained CNN model
        image: Input image (numpy array, any size)
        class_names: Dictionary mapping class indices to letter names
        
    Returns:
        letter: Predicted letter
        confidence: Prediction confidence
    """
    import cv2
    
    # Preprocess image
    if image.shape[:2] != (IMG_SIZE, IMG_SIZE):
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Predict
    predictions = model.predict(image, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    
    letter = class_names.get(class_idx, '?')
    
    return letter, confidence


def main():
    """Main training script for command line usage."""
    import sys
    
    # Default paths
    dataset_path = None
    model_path = "models/asl_cnn_model.keras"
    epochs = 20
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    if len(sys.argv) > 3:
        epochs = int(sys.argv[3])
    
    if dataset_path is None:
        print("Usage: python train_cnn.py <dataset_path> [model_path] [epochs]")
        print("\nExample:")
        print("  python train_cnn.py /path/to/asl-alphabet models/asl_cnn.keras 20")
        print("\nTo use with Kaggle dataset:")
        print("  import kagglehub")
        print("  path = kagglehub.dataset_download('grassknoted/asl-alphabet')")
        print("  python train_cnn.py $path")
        return
    
    # Ensure model directory exists
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    # Train
    train_cnn_model(dataset_path, model_path, epochs)


if __name__ == "__main__":
    main()

