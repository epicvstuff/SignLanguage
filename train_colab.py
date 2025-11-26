"""
ASL Sign Language Translator - Google Colab Training Script

This script is designed to be run directly in Google Colab cells.
It downloads the Kaggle ASL Alphabet dataset and trains a classifier.

USAGE:
======
Copy and paste each section into separate Colab cells.

Alternatively, upload this file to Colab and run:
    !python train_colab.py
"""

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
# !pip install mediapipe scikit-learn kagglehub opencv-python

# ============================================================
# CELL 2: Import Libraries and Download Dataset
# ============================================================

import os
import numpy as np
import pickle
import cv2
from collections import Counter

# For Kaggle dataset download
import kagglehub

# For hand landmark extraction
import mediapipe as mp

# For training
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

# Download the dataset
print("\nDownloading ASL Alphabet dataset from Kaggle...")
dataset_path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print(f"Dataset downloaded to: {dataset_path}")

# ============================================================
# CELL 3: Define Helper Classes
# ============================================================

class HandLandmarkExtractor:
    """Extracts hand landmarks from images using MediaPipe."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    
    def extract(self, image):
        """Extract normalized landmarks from an image."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            return None
        
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        
        landmarks = np.array(landmarks)
        
        # Normalize: center around wrist, scale by hand size
        wrist = landmarks[0]
        centered = landmarks - wrist
        hand_size = max(np.linalg.norm(centered[12][:2]), 0.001)
        normalized = centered / hand_size
        
        return normalized.flatten()
    
    def close(self):
        self.hands.close()


def load_kaggle_data(dataset_path, max_images_per_class=500):
    """Load and process images from the Kaggle ASL dataset."""
    
    # Find the training folder
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
    
    if train_path is None:
        raise FileNotFoundError(f"Could not find training data in {dataset_path}")
    
    print(f"Training data path: {train_path}")
    
    # Initialize extractor
    extractor = HandLandmarkExtractor()
    
    X = []
    y = []
    
    # Load A-Z letters
    letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    for letter in letters:
        letter_path = os.path.join(train_path, letter)
        
        if not os.path.exists(letter_path):
            print(f"  Warning: Folder not found for '{letter}'")
            continue
        
        image_files = [f for f in os.listdir(letter_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files = image_files[:max_images_per_class]
        
        success_count = 0
        
        for img_file in image_files:
            try:
                image = cv2.imread(os.path.join(letter_path, img_file))
                if image is None:
                    continue
                
                features = extractor.extract(image)
                if features is not None:
                    X.append(features)
                    y.append(letter)
                    success_count += 1
            except:
                pass
        
        print(f"  {letter}: {success_count} samples")
    
    extractor.close()
    
    return np.array(X), np.array(y)


def augment_data(X, y, factor=3):
    """Augment data with noise."""
    X_aug = [X]
    y_aug = [y]
    
    for _ in range(factor):
        noise = np.random.normal(0, 0.02, X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)
    
    X_aug = np.vstack(X_aug)
    y_aug = np.hstack(y_aug)
    
    # Shuffle
    idx = np.random.permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]


# ============================================================
# CELL 4: Load and Process Data
# ============================================================

print("="*60)
print("Loading images and extracting hand landmarks...")
print("="*60)

# Adjust max_images_per_class based on your Colab resources
# 500 works well for free Colab, increase for better accuracy
MAX_IMAGES_PER_CLASS = 500

X, y = load_kaggle_data(dataset_path, max_images_per_class=MAX_IMAGES_PER_CLASS)

print(f"\nTotal samples: {len(X)}")
print(f"Classes: {len(set(y))}")

# Show distribution
print("\nClass distribution:")
for letter, count in sorted(Counter(y).items()):
    print(f"  {letter}: {count}")

# ============================================================
# CELL 5: Train the Model
# ============================================================

print("\n" + "="*60)
print("Training the classifier...")
print("="*60)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Augment data
print("\nAugmenting data...")
X_aug, y_aug = augment_data(X, y_encoded, factor=3)
print(f"Augmented: {len(X)} -> {len(X_aug)} samples")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
)

print(f"Training set: {len(X_train)}")
print(f"Test set: {len(X_test)}")

# Train Random Forest
print("\nTraining Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\nTraining accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Cross-validation
print("\nCross-validation...")
cv_scores = cross_val_score(model, X_aug, y_aug, cv=5)
print(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Classification report
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ============================================================
# CELL 6: Save the Model
# ============================================================

MODEL_PATH = "asl_classifier.pkl"

model_data = {
    'model': model,
    'label_encoder': label_encoder
}

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nModel saved to {MODEL_PATH}")
print(f"Final test accuracy: {test_acc:.2%}")

# ============================================================
# CELL 7: Download the Model (Run in Colab)
# ============================================================
# from google.colab import files
# files.download("asl_classifier.pkl")

print("\n" + "="*60)
print("DONE! To download the model in Colab, run:")
print("  from google.colab import files")
print("  files.download('asl_classifier.pkl')")
print("="*60)


# ============================================================
# For command-line execution
# ============================================================
if __name__ == "__main__":
    print("\nScript completed successfully!")

