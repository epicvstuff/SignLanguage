"""
Data Collection Tool
Interactive tool to collect hand landmark data for ASL alphabet training.
"""

import cv2
import numpy as np
import pickle
import os
from datetime import datetime
from hand_detector import HandDetector


class DataCollector:
    """
    Interactive data collection tool for ASL alphabet.
    Press A-Z to record samples, Q to quit and save.
    """
    
    def __init__(self, data_path="data/asl_dataset.pkl"):
        """
        Initialize the data collector.
        
        Args:
            data_path: Path to save the collected data
        """
        self.data_path = data_path
        self.detector = HandDetector()
        
        # Data storage: {letter: [feature_arrays]}
        self.data = {chr(i): [] for i in range(ord('A'), ord('Z') + 1)}
        
        # Load existing data if available
        self.load_existing_data()
        
        # UI colors
        self.colors = {
            'text': (255, 255, 255),
            'success': (0, 255, 0),
            'warning': (0, 255, 255),
            'error': (0, 0, 255),
            'info': (255, 200, 100),
            'bg': (40, 40, 40),
            'panel': (60, 60, 60)
        }
    
    def load_existing_data(self):
        """Load existing data if available."""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'rb') as f:
                    self.data = pickle.load(f)
                print(f"Loaded existing data from {self.data_path}")
                self.print_data_summary()
            except Exception as e:
                print(f"Could not load existing data: {e}")
    
    def save_data(self):
        """Save collected data to disk."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
        with open(self.data_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Data saved to {self.data_path}")
        self.print_data_summary()
    
    def print_data_summary(self):
        """Print a summary of collected data."""
        total = 0
        print("\n=== Data Summary ===")
        for letter in sorted(self.data.keys()):
            count = len(self.data[letter])
            total += count
            if count > 0:
                print(f"  {letter}: {count} samples")
        print(f"  Total: {total} samples")
        print("====================\n")
    
    def draw_ui(self, frame, current_letter=None, last_recorded=None, hand_detected=False):
        """
        Draw the UI overlay on the frame.
        
        Args:
            frame: The video frame
            current_letter: Currently selected letter (if any)
            last_recorded: Last recorded letter and timestamp
            hand_detected: Whether a hand is currently detected
        """
        h, w = frame.shape[:2]
        
        # Draw semi-transparent panel at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), self.colors['panel'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "ASL Data Collector", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2)
        
        # Hand status
        status_color = self.colors['success'] if hand_detected else self.colors['error']
        status_text = "Hand Detected" if hand_detected else "No Hand Detected"
        cv2.putText(frame, status_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Instructions
        cv2.putText(frame, "Press A-Z to record | Q to quit & save", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['info'], 1)
        
        # Draw sample counts panel on right side
        panel_x = w - 180
        cv2.rectangle(overlay, (panel_x - 10, 0), (w, h), self.colors['panel'], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, "Samples:", (panel_x, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Show sample counts in columns
        y_offset = 50
        col1_letters = [chr(i) for i in range(ord('A'), ord('N'))]
        col2_letters = [chr(i) for i in range(ord('N'), ord('Z') + 1)]
        
        for i, letter in enumerate(col1_letters):
            count = len(self.data[letter])
            color = self.colors['success'] if count >= 50 else (
                self.colors['warning'] if count >= 20 else self.colors['text']
            )
            cv2.putText(frame, f"{letter}:{count:3d}", (panel_x, y_offset + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        for i, letter in enumerate(col2_letters):
            count = len(self.data[letter])
            color = self.colors['success'] if count >= 50 else (
                self.colors['warning'] if count >= 20 else self.colors['text']
            )
            cv2.putText(frame, f"{letter}:{count:3d}", (panel_x + 80, y_offset + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Show recording feedback
        if last_recorded:
            letter, timestamp = last_recorded
            elapsed = (datetime.now() - timestamp).total_seconds()
            if elapsed < 0.5:  # Show for 0.5 seconds
                # Flash effect
                cv2.rectangle(frame, (w//2 - 100, h//2 - 50), (w//2 + 100, h//2 + 50),
                             self.colors['success'], 3)
                cv2.putText(frame, f"Recorded: {letter}", (w//2 - 80, h//2 + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['success'], 2)
        
        return frame
    
    def run(self):
        """Run the data collection loop."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\n" + "="*50)
        print("ASL Data Collector")
        print("="*50)
        print("Instructions:")
        print("  - Position your hand in the camera view")
        print("  - Press A-Z to record a sample for that letter")
        print("  - Collect at least 50 samples per letter")
        print("  - Press Q to quit and save data")
        print("="*50 + "\n")
        
        last_recorded = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get hand features
            features, results = self.detector.get_features(frame)
            hand_detected = features is not None
            
            # Draw hand landmarks
            frame = self.detector.draw_landmarks(frame, results)
            
            # Draw UI
            frame = self.draw_ui(frame, last_recorded=last_recorded, hand_detected=hand_detected)
            
            # Display
            cv2.imshow("ASL Data Collector", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                # Quit and save
                self.save_data()
                break
            elif ord('a') <= key <= ord('z') or ord('A') <= key <= ord('Z'):
                # Record sample for pressed letter
                letter = chr(key).upper()
                
                if features is not None:
                    self.data[letter].append(features)
                    last_recorded = (letter, datetime.now())
                    print(f"Recorded sample for '{letter}' - Total: {len(self.data[letter])}")
                else:
                    print(f"Cannot record '{letter}' - No hand detected!")
        
        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()


def main():
    """Main entry point."""
    import sys
    
    # Allow custom data path
    data_path = "data/asl_dataset.pkl"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    # Get absolute path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_path = os.path.join(parent_dir, data_path)
    
    collector = DataCollector(data_path)
    collector.run()


if __name__ == "__main__":
    main()



