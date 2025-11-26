"""
Model Training Module
Train a classifier to recognize ASL alphabet letters from hand landmarks.

Supports two data sources:
1. Custom collected data (pickle file from data_collector.py)
2. Kaggle ASL Alphabet dataset (grassknoted/asl-alphabet)

For Google Colab usage with Kaggle dataset:
```python
# Install dependencies
!pip install mediapipe scikit-learn kagglehub

# Download dataset
import kagglehub
path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("Path to dataset files:", path)

# Clone or upload the src folder, then run:
from train_model import train_from_kaggle
train_from_kaggle(path, "asl_classifier.pkl", max_images_per_class=500)
```
"""

import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class HandLandmarkExtractor:
    """
    Extracts hand landmarks from images using MediaPipe.
    Used for processing image datasets.
    """
    
    def __init__(self):
        """Initialize MediaPipe hands."""
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    
    def extract_landmarks(self, image):
        """
        Extract normalized landmarks from an image.
        
        Args:
            image: BGR or RGB image (numpy array)
            
        Returns:
            features: Flattened normalized landmarks (63 values) or None
        """
        import cv2
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Process with MediaPipe
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Get first hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract coordinates
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        
        landmarks = np.array(landmarks)
        
        # Normalize: center around wrist, scale by hand size
        wrist = landmarks[0]
        centered = landmarks - wrist
        
        # Scale by distance from wrist to middle finger tip
        middle_tip = centered[12]
        hand_size = np.linalg.norm(middle_tip[:2])
        
        if hand_size < 0.001:
            hand_size = 0.001
        
        normalized = centered / hand_size
        
        return normalized.flatten()
    
    def close(self):
        """Release resources."""
        self.hands.close()


class KaggleDatasetLoader:
    """
    Loads and processes the Kaggle ASL Alphabet dataset.
    Dataset: grassknoted/asl-alphabet
    """
    
    def __init__(self, dataset_path, max_images_per_class=None):
        """
        Initialize the loader.
        
        Args:
            dataset_path: Path to the downloaded Kaggle dataset
            max_images_per_class: Maximum images to load per class (None for all)
        """
        self.dataset_path = dataset_path
        self.max_images_per_class = max_images_per_class
        self.extractor = HandLandmarkExtractor()
        
        # Find the actual training data folder
        self.train_path = self._find_train_folder()
    
    def _find_train_folder(self):
        """Find the training data folder in the dataset."""
        possible_paths = [
            os.path.join(self.dataset_path, "asl_alphabet_train", "asl_alphabet_train"),
            os.path.join(self.dataset_path, "asl_alphabet_train"),
            self.dataset_path
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                # Check if it contains letter folders
                if os.path.exists(os.path.join(path, "A")):
                    return path
        
        raise FileNotFoundError(
            f"Could not find training data in {self.dataset_path}\n"
            f"Expected folder structure: asl_alphabet_train/asl_alphabet_train/A, B, C, ..."
        )
    
    def load_data(self, letters=None, verbose=True):
        """
        Load and process images from the dataset.
        
        Args:
            letters: List of letters to load (default: A-Z)
            verbose: Whether to print progress
            
        Returns:
            X: Feature array (n_samples, 63)
            y: Label array (n_samples,)
        """
        import cv2
        
        if letters is None:
            # Default to A-Z only (skip 'del', 'nothing', 'space')
            letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        
        X = []
        y = []
        
        total_processed = 0
        total_failed = 0
        
        for letter in letters:
            letter_path = os.path.join(self.train_path, letter)
            
            if not os.path.exists(letter_path):
                if verbose:
                    print(f"Warning: Folder not found for '{letter}'")
                continue
            
            # Get image files
            image_files = [f for f in os.listdir(letter_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Limit number of images if specified
            if self.max_images_per_class is not None:
                image_files = image_files[:self.max_images_per_class]
            
            success_count = 0
            fail_count = 0
            
            for img_file in image_files:
                img_path = os.path.join(letter_path, img_file)
                
                try:
                    # Load image
                    image = cv2.imread(img_path)
                    if image is None:
                        fail_count += 1
                        continue
                    
                    # Extract landmarks
                    features = self.extractor.extract_landmarks(image)
                    
                    if features is not None:
                        X.append(features)
                        y.append(letter)
                        success_count += 1
                    else:
                        fail_count += 1
                        
                except Exception as e:
                    fail_count += 1
            
            total_processed += success_count
            total_failed += fail_count
            
            if verbose:
                print(f"  {letter}: {success_count} samples extracted, {fail_count} failed")
        
        self.extractor.close()
        
        X = np.array(X)
        y = np.array(y)
        
        if verbose:
            print(f"\nTotal: {total_processed} samples loaded, {total_failed} failed")
            print(f"Classes: {len(set(y))}")
        
        return X, y


class ASLTrainer:
    """
    Trains and evaluates a classifier for ASL alphabet recognition.
    """
    
    def __init__(self, data_path="data/asl_dataset.pkl", model_path="models/asl_classifier.pkl"):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to the collected training data (pickle file)
            model_path: Path to save the trained model
        """
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.label_encoder = LabelEncoder()
    
    def load_data(self):
        """
        Load and prepare the training data from pickle file.
        
        Returns:
            X: Feature array (n_samples, n_features)
            y: Label array (n_samples,)
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}\n"
                                   f"Please run data_collector.py first to collect training data.")
        
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        X = []
        y = []
        
        for letter, samples in data.items():
            for sample in samples:
                X.append(sample)
                y.append(letter)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Loaded {len(X)} samples across {len(set(y))} classes")
        return X, y
    
    def load_data_from_kaggle(self, dataset_path, max_images_per_class=None, letters=None):
        """
        Load training data from Kaggle ASL Alphabet dataset.
        
        Args:
            dataset_path: Path to the downloaded Kaggle dataset
            max_images_per_class: Maximum images to load per class
            letters: List of letters to load (default: A-Z)
            
        Returns:
            X: Feature array (n_samples, 63)
            y: Label array (n_samples,)
        """
        print("Loading data from Kaggle ASL Alphabet dataset...")
        print(f"Dataset path: {dataset_path}")
        
        if max_images_per_class:
            print(f"Max images per class: {max_images_per_class}")
        
        loader = KaggleDatasetLoader(dataset_path, max_images_per_class)
        return loader.load_data(letters=letters)
    
    def augment_data(self, X, y, augmentation_factor=3):
        """
        Augment data by adding slight perturbations.
        
        Args:
            X: Feature array
            y: Label array
            augmentation_factor: Number of augmented samples per original sample
            
        Returns:
            X_augmented: Augmented feature array
            y_augmented: Augmented label array
        """
        X_augmented = [X]
        y_augmented = [y]
        
        for _ in range(augmentation_factor):
            # Add small random noise
            noise = np.random.normal(0, 0.02, X.shape)
            X_noisy = X + noise
            X_augmented.append(X_noisy)
            y_augmented.append(y)
        
        X_augmented = np.vstack(X_augmented)
        y_augmented = np.hstack(y_augmented)
        
        # Shuffle the augmented data
        indices = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[indices]
        y_augmented = y_augmented[indices]
        
        print(f"Augmented data: {len(X)} -> {len(X_augmented)} samples")
        return X_augmented, y_augmented
    
    def train(self, X, y, augment=True):
        """
        Train the classifier.
        
        Args:
            X: Feature array
            y: Label array
            augment: Whether to use data augmentation
            
        Returns:
            accuracy: Test accuracy
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Augment data if requested
        if augment:
            X, y_encoded = self.augment_data(X, y_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train Random Forest classifier
        print("\nTraining Random Forest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        print(f"\nTraining accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X, y_encoded, cv=5)
        print(f"CV scores: {cv_scores}")
        print(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Classification report
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0
        ))
        
        return test_accuracy
    
    def save_model(self, model_path=None):
        """Save the trained model and label encoder."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        if model_path is not None:
            self.model_path = model_path
        
        # Ensure directory exists
        model_dir = os.path.dirname(self.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        
        # Save model and label encoder together
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to {self.model_path}")
    
    @staticmethod
    def load_model(model_path="models/asl_classifier.pkl"):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            model: Trained classifier
            label_encoder: Fitted label encoder
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        return model_data['model'], model_data['label_encoder']


def train_from_kaggle(dataset_path, model_path="asl_classifier.pkl", max_images_per_class=500, augment=True):
    """
    Convenience function to train from Kaggle dataset.
    Designed for easy use in Google Colab.
    
    Args:
        dataset_path: Path to downloaded Kaggle dataset
        model_path: Where to save the trained model
        max_images_per_class: Max images per letter (500 recommended for Colab)
        augment: Whether to augment data
        
    Returns:
        trainer: Trained ASLTrainer instance
        accuracy: Test accuracy
        
    Example usage in Colab:
    ```python
    # Install dependencies
    !pip install mediapipe scikit-learn kagglehub opencv-python
    
    # Download dataset
    import kagglehub
    path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    
    # Train (use the code directly or import)
    # Option 1: Copy train_model.py content to a cell and run:
    trainer, accuracy = train_from_kaggle(path, "asl_classifier.pkl", max_images_per_class=500)
    
    # Download the model
    from google.colab import files
    files.download("asl_classifier.pkl")
    ```
    """
    print("="*60)
    print("ASL Alphabet Classifier - Training from Kaggle Dataset")
    print("="*60)
    
    trainer = ASLTrainer(model_path=model_path)
    
    # Load data from Kaggle dataset
    X, y = trainer.load_data_from_kaggle(dataset_path, max_images_per_class)
    
    if len(X) == 0:
        raise ValueError("No samples were loaded. Check the dataset path.")
    
    # Check class distribution
    from collections import Counter
    class_counts = Counter(y)
    print(f"\nClass distribution:")
    for letter, count in sorted(class_counts.items()):
        print(f"  {letter}: {count}")
    
    # Train
    accuracy = trainer.train(X, y, augment=augment)
    
    # Save
    trainer.save_model()
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Final test accuracy: {accuracy:.2%}")
    print(f"Model saved to: {model_path}")
    print("="*60)
    
    return trainer, accuracy


def main():
    """Main training script for command line usage."""
    import sys
    
    # Get paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    data_path = os.path.join(parent_dir, "data/asl_dataset.pkl")
    model_path = os.path.join(parent_dir, "models/asl_classifier.pkl")
    
    # Allow custom paths from command line
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    
    print("="*50)
    print("ASL Alphabet Classifier Training")
    print("="*50)
    
    trainer = ASLTrainer(data_path, model_path)
    
    try:
        # Load data
        X, y = trainer.load_data()
        
        # Check minimum samples
        from collections import Counter
        class_counts = Counter(y)
        min_samples = min(class_counts.values())
        
        if min_samples < 5:
            print("\nWarning: Some classes have very few samples!")
            print("Class distribution:")
            for letter, count in sorted(class_counts.items()):
                print(f"  {letter}: {count}")
            print("\nConsider collecting more data before training.")
            
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
        
        # Train
        accuracy = trainer.train(X, y)
        
        # Save model
        trainer.save_model()
        
        print("\n" + "="*50)
        print("Training complete!")
        print(f"Final test accuracy: {accuracy:.2%}")
        print("="*50)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease collect training data first by running:")
        print("  python src/data_collector.py")


# ============================================================
# GOOGLE COLAB CELL - Copy everything below to a Colab cell
# ============================================================
"""
# Cell 1: Install dependencies
!pip install mediapipe scikit-learn kagglehub opencv-python

# Cell 2: Download the Kaggle dataset
import kagglehub
dataset_path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("Dataset path:", dataset_path)

# Cell 3: Copy the train_from_kaggle function and run training
# (Copy the train_from_kaggle function and its dependencies above)
# Then run:
trainer, accuracy = train_from_kaggle(dataset_path, "asl_classifier.pkl", max_images_per_class=500)

# Cell 4: Download the trained model
from google.colab import files
files.download("asl_classifier.pkl")
"""


if __name__ == "__main__":
    main()
