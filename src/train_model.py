"""
Model Training Module
Train a classifier to recognize ASL alphabet letters from hand landmarks.
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


class ASLTrainer:
    """
    Trains and evaluates a classifier for ASL alphabet recognition.
    """
    
    def __init__(self, data_path="data/asl_dataset.pkl", model_path="models/asl_classifier.pkl"):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to the collected training data
            model_path: Path to save the trained model
        """
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.label_encoder = LabelEncoder()
    
    def load_data(self):
        """
        Load and prepare the training data.
        
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
    
    def save_model(self):
        """Save the trained model and label encoder."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
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


def main():
    """Main training script."""
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


if __name__ == "__main__":
    main()

