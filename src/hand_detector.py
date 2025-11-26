"""
Hand Detector Module
Uses MediaPipe to detect hands and extract normalized landmark features.
"""

import cv2
import numpy as np
import mediapipe as mp


class HandDetector:
    """
    Detects hands using MediaPipe and extracts normalized landmark features.
    
    The 21 hand landmarks are normalized relative to the hand's bounding box
    to make the features position and scale invariant.
    """
    
    def __init__(self, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        Initialize the hand detector.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Number of landmarks per hand (MediaPipe provides 21 landmarks)
        self.num_landmarks = 21
        # Features per landmark (x, y, z)
        self.features_per_landmark = 3
        # Total features
        self.num_features = self.num_landmarks * self.features_per_landmark
    
    def process_frame(self, frame):
        """
        Process a frame and detect hands.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            results: MediaPipe hand detection results
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame)
    
    def extract_landmarks(self, results):
        """
        Extract raw landmark coordinates from detection results.
        
        Args:
            results: MediaPipe hand detection results
            
        Returns:
            landmarks: List of (x, y, z) tuples for each landmark, or None if no hand detected
        """
        if not results.multi_hand_landmarks:
            return None
        
        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        
        return landmarks
    
    def normalize_landmarks(self, landmarks):
        """
        Normalize landmarks to be position and scale invariant.
        
        The landmarks are normalized by:
        1. Centering around the wrist (landmark 0)
        2. Scaling based on the hand size (distance from wrist to middle finger tip)
        
        Args:
            landmarks: List of (x, y, z) tuples
            
        Returns:
            normalized: Flattened numpy array of normalized features
        """
        if landmarks is None:
            return None
        
        landmarks = np.array(landmarks)
        
        # Center around wrist (landmark 0)
        wrist = landmarks[0]
        centered = landmarks - wrist
        
        # Calculate hand size (distance from wrist to middle finger tip - landmark 12)
        middle_tip = centered[12]
        hand_size = np.linalg.norm(middle_tip[:2])  # Use x, y for scale
        
        # Avoid division by zero
        if hand_size < 0.001:
            hand_size = 0.001
        
        # Scale the landmarks
        normalized = centered / hand_size
        
        # Flatten to 1D array
        return normalized.flatten()
    
    def get_features(self, frame):
        """
        Extract normalized hand features from a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            features: Numpy array of 63 normalized features, or None if no hand detected
        """
        results = self.process_frame(frame)
        landmarks = self.extract_landmarks(results)
        return self.normalize_landmarks(landmarks), results
    
    def draw_landmarks(self, frame, results):
        """
        Draw hand landmarks on the frame.
        
        Args:
            frame: BGR image to draw on
            results: MediaPipe hand detection results
            
        Returns:
            frame: Frame with landmarks drawn
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return frame
    
    def get_hand_bbox(self, results, frame_width, frame_height):
        """
        Get the bounding box of the detected hand.
        
        Args:
            results: MediaPipe hand detection results
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            bbox: (x, y, w, h) tuple or None if no hand detected
        """
        if not results.multi_hand_landmarks:
            return None
        
        hand_landmarks = results.multi_hand_landmarks[0]
        
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        
        x_min = int(min(x_coords) * frame_width)
        x_max = int(max(x_coords) * frame_width)
        y_min = int(min(y_coords) * frame_height)
        y_max = int(max(y_coords) * frame_height)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame_width, x_max + padding)
        y_max = min(frame_height, y_max + padding)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def close(self):
        """Release resources."""
        self.hands.close()


if __name__ == "__main__":
    # Test the hand detector
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    
    print("Hand Detector Test - Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Get features
        features, results = detector.get_features(frame)
        
        # Draw landmarks
        frame = detector.draw_landmarks(frame, results)
        
        # Display feature info
        if features is not None:
            cv2.putText(frame, f"Features: {len(features)} values", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Hand Detected!", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Hand Detector Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

