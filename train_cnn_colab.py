"""
ASL Sign Language Translator - CNN Training Script for Google Colab

This script trains a MobileNetV2-based CNN on the Kaggle ASL Alphabet dataset.
Optimized for Google Colab with GPU support.

USAGE IN COLAB:
===============
1. Create a new Colab notebook
2. Enable GPU: Runtime > Change runtime type > GPU
3. Copy each cell section below into separate cells and run them

Or upload this file and run: !python train_cnn_colab.py
"""

# ============================================================
# CELL 1: Install Dependencies and Check GPU
# ============================================================
# !pip install tensorflow kagglehub

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

if tf.config.list_physical_devices('GPU'):
    print("GPU is available! Training will be faster.")
else:
    print("No GPU detected. Training will use CPU (slower).")

# ============================================================
# CELL 2: Download Dataset
# ============================================================

import kagglehub
import os

print("Downloading ASL Alphabet dataset from Kaggle...")
dataset_path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print(f"Dataset downloaded to: {dataset_path}")

# Find training folder
possible_paths = [
    os.path.join(dataset_path, "asl_alphabet_train", "asl_alphabet_train"),
    os.path.join(dataset_path, "asl_alphabet_train"),
    dataset_path
]

train_path = None
for path in possible_paths:
    if os.path.exists(path) and os.path.exists(os.path.join(path, "A")):
        train_path = path
        break

if train_path:
    print(f"Training data found at: {train_path}")
    # List classes
    classes = sorted([d for d in os.listdir(train_path) 
                     if os.path.isdir(os.path.join(train_path, d)) and len(d) == 1])
    print(f"Classes found: {classes}")
else:
    raise FileNotFoundError("Could not find training data!")

# ============================================================
# CELL 3: Setup Data Generators
# ============================================================

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 32

# Only use A-Z letters
classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    horizontal_flip=False,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("Creating data generators...")

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=classes,
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=classes,
    subset='validation',
    shuffle=False
)

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Number of classes: {len(classes)}")

# ============================================================
# CELL 4: Build the Model
# ============================================================

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

print("Building MobileNetV2 model with transfer learning...")

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

# Build model
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Preprocessing
x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255)

# Base model
x = base_model(x, training=False)

# Classification head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(26, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print(f"\nTotal parameters: {model.count_params():,}")
print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")

# ============================================================
# CELL 5: Train the Model
# ============================================================

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

MODEL_PATH = "asl_cnn_model.keras"
EPOCHS = 20

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH,
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

print(f"Training for up to {EPOCHS} epochs...")
print(f"Model will be saved to: {MODEL_PATH}")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# CELL 6: Evaluate and Visualize
# ============================================================

import matplotlib.pyplot as plt

# Evaluate
print("\n" + "="*50)
print("Final Evaluation")
print("="*50)

results = model.evaluate(val_generator, verbose=1)
print(f"Validation Loss: {results[0]:.4f}")
print(f"Validation Accuracy: {results[1]:.4f}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Training')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss
axes[1].plot(history.history['loss'], label='Training')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()

print(f"\nBest validation accuracy: {max(history.history['val_accuracy']):.2%}")

# ============================================================
# CELL 7: Save Class Names and Test Prediction
# ============================================================

import json
import numpy as np

# Save class names
class_names = {v: k for k, v in train_generator.class_indices.items()}
with open('asl_cnn_model_classes.json', 'w') as f:
    json.dump(class_names, f)

print("Class names saved to: asl_cnn_model_classes.json")

# Test prediction on a sample
print("\nTesting prediction on sample images...")

# Get a batch of validation images
sample_images, sample_labels = next(val_generator)

# Predict
predictions = model.predict(sample_images[:5], verbose=0)

print("\nSample predictions:")
for i in range(5):
    true_label = class_names[np.argmax(sample_labels[i])]
    pred_label = class_names[np.argmax(predictions[i])]
    confidence = np.max(predictions[i])
    status = "✓" if true_label == pred_label else "✗"
    print(f"  {status} True: {true_label}, Predicted: {pred_label} ({confidence:.1%})")

# ============================================================
# CELL 8: Download the Model (Colab only)
# ============================================================
# Uncomment and run in Colab to download the trained model

# from google.colab import files
# files.download('asl_cnn_model.keras')
# files.download('asl_cnn_model_classes.json')
# files.download('training_history.png')

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print(f"Model saved to: {MODEL_PATH}")
print(f"Final accuracy: {results[1]:.2%}")
print("\nTo download in Colab, run:")
print("  from google.colab import files")
print("  files.download('asl_cnn_model.keras')")
print("  files.download('asl_cnn_model_classes.json')")
print("="*50)

# ============================================================
# For command-line execution
# ============================================================
if __name__ == "__main__":
    print("\nScript completed successfully!")

