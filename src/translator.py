"""
ASL Sign Language Translator
Real-time ASL alphabet recognition with text-to-speech output.
"""

import cv2
import numpy as np
import os
import time
from collections import deque

from hand_detector import HandDetector
from text_to_speech import TextToSpeech, SpeechBuffer
from train_model import ASLTrainer


class ASLTranslator:
    """
    Real-time ASL alphabet translator.
    Recognizes hand gestures and converts them to text and speech.
    """
    
    def __init__(self, model_path="models/asl_classifier.pkl"):
        """
        Initialize the translator.
        
        Args:
            model_path: Path to the trained classifier model
        """
        self.model_path = model_path
        self.detector = HandDetector()
        self.tts = TextToSpeech(rate=150)
        self.buffer = SpeechBuffer(self.tts)
        
        # Load the trained model
        self.model = None
        self.label_encoder = None
        self.load_model()
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=15)
        self.stable_prediction = None
        self.stable_start_time = None
        self.letter_added = False
        
        # Timing settings
        self.stability_threshold = 1.0  # Seconds to hold for stable prediction
        self.min_confidence = 0.6  # Minimum confidence threshold
        
        # UI colors (elegant dark theme)
        self.colors = {
            'bg': (30, 30, 35),
            'panel': (45, 45, 50),
            'text': (240, 240, 240),
            'accent': (100, 200, 255),
            'success': (100, 220, 130),
            'warning': (100, 200, 255),
            'error': (100, 100, 255),
            'progress': (100, 200, 255),
            'dim': (120, 120, 130)
        }
        
        # Window settings
        self.window_width = 1000
        self.window_height = 700
    
    def load_model(self):
        """Load the trained classifier model."""
        if not os.path.exists(self.model_path):
            print(f"Warning: Model not found at {self.model_path}")
            print("Please train a model first by running train_model.py")
            return False
        
        try:
            self.model, self.label_encoder = ASLTrainer.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Classes: {list(self.label_encoder.classes_)}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, features):
        """
        Predict the letter from hand features.
        
        Args:
            features: Normalized hand landmark features
            
        Returns:
            letter: Predicted letter
            confidence: Prediction confidence
        """
        if self.model is None or features is None:
            return None, 0.0
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Get prediction and probability
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = probabilities[prediction]
        
        # Decode label
        letter = self.label_encoder.inverse_transform([prediction])[0]
        
        return letter, confidence
    
    def get_stable_prediction(self, letter, confidence):
        """
        Get a stable prediction by tracking consistent predictions over time.
        
        Args:
            letter: Current predicted letter
            confidence: Current prediction confidence
            
        Returns:
            stable_letter: Stable letter prediction or None
            progress: Progress towards stability (0.0 to 1.0)
        """
        current_time = time.time()
        
        if confidence < self.min_confidence:
            self.prediction_history.clear()
            self.stable_prediction = None
            self.stable_start_time = None
            return None, 0.0
        
        # Add to history
        self.prediction_history.append(letter)
        
        # Check if all recent predictions are the same
        if len(self.prediction_history) < 5:
            return None, 0.0
        
        # Get most common prediction
        from collections import Counter
        counts = Counter(self.prediction_history)
        most_common, count = counts.most_common(1)[0]
        
        # Check if prediction is consistent (>80% of history)
        consistency = count / len(self.prediction_history)
        
        if consistency > 0.8:
            if self.stable_prediction != most_common:
                # New stable prediction started
                self.stable_prediction = most_common
                self.stable_start_time = current_time
                self.letter_added = False
            
            # Calculate progress
            elapsed = current_time - self.stable_start_time
            progress = min(1.0, elapsed / self.stability_threshold)
            
            if progress >= 1.0 and not self.letter_added:
                self.letter_added = True
                return most_common, 1.0
            
            return None, progress
        else:
            self.stable_prediction = None
            self.stable_start_time = None
            return None, 0.0
    
    def draw_ui(self, frame, prediction=None, confidence=0.0, progress=0.0, hand_detected=False):
        """
        Draw the translator UI.
        
        Args:
            frame: Video frame (will be resized)
            prediction: Current predicted letter
            confidence: Prediction confidence
            progress: Stability progress
            hand_detected: Whether a hand is detected
            
        Returns:
            canvas: Complete UI canvas
        """
        # Create canvas
        canvas = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        canvas[:] = self.colors['bg']
        
        # Resize frame to fit in left panel
        frame_height = 480
        frame_width = 640
        
        # Ensure frame is valid and has correct format
        if frame is None or frame.size == 0:
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Convert to BGR if needed (handle grayscale or BGRA)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        frame_resized = cv2.resize(frame, (frame_width, frame_height))
        
        # Place frame
        frame_x = 20
        frame_y = 100
        canvas[frame_y:frame_y+frame_height, frame_x:frame_x+frame_width] = frame_resized
        
        # Draw frame border
        border_color = self.colors['success'] if hand_detected else self.colors['dim']
        cv2.rectangle(canvas, (frame_x-2, frame_y-2), 
                     (frame_x+frame_width+2, frame_y+frame_height+2), 
                     border_color, 2)
        
        # Title
        cv2.putText(canvas, "ASL Sign Language Translator", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors['text'], 2)
        
        # Status indicator
        status_x = 20
        status_y = 80
        status_text = "Hand Detected" if hand_detected else "No Hand Detected"
        status_color = self.colors['success'] if hand_detected else self.colors['error']
        cv2.circle(canvas, (status_x + 8, status_y - 5), 6, status_color, -1)
        cv2.putText(canvas, status_text, (status_x + 25, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Right panel - prediction display
        panel_x = 680
        panel_y = 100
        panel_width = 300
        panel_height = 200
        
        # Draw panel background
        cv2.rectangle(canvas, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     self.colors['panel'], -1)
        cv2.rectangle(canvas, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     self.colors['dim'], 1)
        
        cv2.putText(canvas, "Current Sign", (panel_x + 20, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['dim'], 1)
        
        if prediction and self.model is not None:
            # Large letter display
            letter_size = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_SIMPLEX, 4, 4)[0]
            letter_x = panel_x + (panel_width - letter_size[0]) // 2
            letter_y = panel_y + 120
            cv2.putText(canvas, prediction, (letter_x, letter_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 4, self.colors['accent'], 4)
            
            # Confidence bar
            bar_x = panel_x + 20
            bar_y = panel_y + 150
            bar_width = panel_width - 40
            bar_height = 15
            
            cv2.rectangle(canvas, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height),
                         self.colors['dim'], 1)
            
            conf_width = int(bar_width * confidence)
            conf_color = self.colors['success'] if confidence > 0.8 else (
                self.colors['warning'] if confidence > 0.6 else self.colors['error']
            )
            cv2.rectangle(canvas, (bar_x + 1, bar_y + 1), 
                         (bar_x + conf_width - 1, bar_y + bar_height - 1),
                         conf_color, -1)
            
            cv2.putText(canvas, f"{confidence:.0%}", (bar_x + bar_width + 10, bar_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            
            # Progress indicator
            if progress > 0:
                prog_y = panel_y + 175
                cv2.putText(canvas, "Hold steady:", (bar_x, prog_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['dim'], 1)
                
                prog_bar_x = bar_x + 80
                prog_width = bar_width - 80
                cv2.rectangle(canvas, (prog_bar_x, prog_y - 10), 
                             (prog_bar_x + prog_width, prog_y + 2),
                             self.colors['dim'], 1)
                
                fill_width = int(prog_width * progress)
                cv2.rectangle(canvas, (prog_bar_x + 1, prog_y - 9), 
                             (prog_bar_x + fill_width - 1, prog_y + 1),
                             self.colors['progress'], -1)
        else:
            # No model or no prediction
            msg = "No model loaded" if self.model is None else "Show hand gesture"
            msg_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            msg_x = panel_x + (panel_width - msg_size[0]) // 2
            cv2.putText(canvas, msg, (msg_x, panel_y + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['dim'], 1)
        
        # Text output panel
        text_panel_y = 320
        text_panel_height = 120
        
        cv2.rectangle(canvas, (panel_x, text_panel_y), 
                     (panel_x + panel_width, text_panel_y + text_panel_height),
                     self.colors['panel'], -1)
        cv2.rectangle(canvas, (panel_x, text_panel_y), 
                     (panel_x + panel_width, text_panel_y + text_panel_height),
                     self.colors['dim'], 1)
        
        cv2.putText(canvas, "Translated Text", (panel_x + 20, text_panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['dim'], 1)
        
        # Display buffer text (wrap if needed)
        buffer_text = self.buffer.get_text()
        if buffer_text:
            # Simple word wrap
            max_chars = 20
            lines = []
            words = buffer_text.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= max_chars:
                    current_line += (" " if current_line else "") + word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            for i, line in enumerate(lines[-3:]):  # Show last 3 lines
                cv2.putText(canvas, line, (panel_x + 20, text_panel_y + 55 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
        else:
            cv2.putText(canvas, "(empty)", (panel_x + 20, text_panel_y + 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['dim'], 1)
        
        # Instructions panel
        inst_y = 460
        inst_height = 120
        
        cv2.rectangle(canvas, (panel_x, inst_y), 
                     (panel_x + panel_width, inst_y + inst_height),
                     self.colors['panel'], -1)
        cv2.rectangle(canvas, (panel_x, inst_y), 
                     (panel_x + panel_width, inst_y + inst_height),
                     self.colors['dim'], 1)
        
        cv2.putText(canvas, "Controls", (panel_x + 20, inst_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['dim'], 1)
        
        controls = [
            ("SPACE", "Add space"),
            ("BACKSPACE", "Delete last"),
            ("ENTER", "Speak text"),
            ("C", "Clear all"),
            ("Q", "Quit")
        ]
        
        for i, (key, action) in enumerate(controls):
            y = inst_y + 50 + i * 18
            cv2.putText(canvas, key, (panel_x + 20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['accent'], 1)
            cv2.putText(canvas, action, (panel_x + 120, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        # Bottom status bar
        bar_y = self.window_height - 30
        cv2.rectangle(canvas, (0, bar_y), (self.window_width, self.window_height),
                     self.colors['panel'], -1)
        
        if self.tts.is_speaking:
            cv2.putText(canvas, "Speaking...", (20, bar_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['success'], 1)
        
        cv2.putText(canvas, "ASL Alphabet Translator", 
                   (self.window_width - 200, bar_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['dim'], 1)
        
        return canvas
    
    def run(self):
        """Run the translator application."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Read a test frame to get camera dimensions
        ret, test_frame = cap.read()
        if not ret:
            print("Error: Could not read from webcam")
            cap.release()
            return
        
        print("\n" + "="*50)
        print("ASL Sign Language Translator")
        print("="*50)
        print(f"Camera resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
        
        if self.model is None:
            print("\nWARNING: No trained model found!")
            print("The translator will run but won't recognize signs.")
            print("Please train a model first:")
            print("  1. Run: python src/data_collector.py")
            print("  2. Run: python src/train_model.py")
        
        print("\nControls:")
        print("  SPACE     - Add space to text")
        print("  BACKSPACE - Delete last character")
        print("  ENTER     - Speak the text")
        print("  C         - Clear all text")
        print("  Q         - Quit")
        print("="*50 + "\n")
        
        cv2.namedWindow("ASL Translator", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ASL Translator", self.window_width, self.window_height)
        
        # Force window to front on macOS
        cv2.waitKey(1)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Warning: Failed to read frame")
                    continue
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Get hand features and results
                features, results = self.detector.get_features(frame)
                hand_detected = features is not None
                
                # Draw hand landmarks on frame
                frame = self.detector.draw_landmarks(frame, results)
                
                # Predict letter
                prediction = None
                confidence = 0.0
                progress = 0.0
                
                if features is not None and self.model is not None:
                    prediction, confidence = self.predict(features)
                    
                    # Check for stable prediction
                    stable_letter, progress = self.get_stable_prediction(prediction, confidence)
                    
                    if stable_letter:
                        self.buffer.add_letter(stable_letter)
                        print(f"Added: {stable_letter} | Text: {self.buffer.get_text()}")
                
                # Draw UI
                try:
                    canvas = self.draw_ui(frame, prediction, confidence, progress, hand_detected)
                except Exception as e:
                    print(f"UI Error: {e}")
                    canvas = frame  # Fallback to raw frame
                
                # Display
                cv2.imshow("ASL Translator", canvas)
                
                # Handle key presses - use longer wait for macOS stability
                key = cv2.waitKey(10) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord(' '):
                    self.buffer.add_space()
                    print(f"Added space | Text: {self.buffer.get_text()}")
                elif key == 8 or key == 127:  # Backspace
                    self.buffer.backspace()
                    print(f"Deleted | Text: {self.buffer.get_text()}")
                elif key == 13 or key == 10:  # Enter
                    text = self.buffer.get_text()
                    if text.strip():
                        print(f"Speaking: {text}")
                        self.buffer.speak_buffer()
                elif key == ord('c') or key == ord('C'):
                    self.buffer.clear()
                    print("Cleared text")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            print("Cleaning up...")
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()
            self.tts.stop()


def main():
    """Main entry point."""
    import sys
    
    # Get model path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    model_path = os.path.join(parent_dir, "models/asl_classifier.pkl")
    
    # Allow custom model path from command line
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    translator = ASLTranslator(model_path)
    translator.run()


if __name__ == "__main__":
    main()

