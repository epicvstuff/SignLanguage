# ASL Sign Language Translator

A real-time American Sign Language (ASL) alphabet translator that uses MediaPipe for hand tracking, a trained classifier for letter recognition, and text-to-speech for audio output.

## Features

- **Real-time Hand Tracking**: Uses MediaPipe Hands to detect and track 21 hand landmarks
- **ASL Alphabet Recognition**: Recognizes letters A-Z from hand gestures
- **Text-to-Speech**: Speaks accumulated text using pyttsx3
- **Data Collection Tool**: Built-in tool to collect training data for custom gestures
- **Visual Feedback**: Live webcam feed with hand skeleton overlay and confidence display

## Architecture

```
Webcam → MediaPipe Hands → 21 Hand Landmarks → Trained Classifier → Letter Prediction → Text Display + TTS
```

## Installation

1. Clone or navigate to this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Collect Training Data

Run the data collection tool to record hand landmarks for each letter:

```bash
python src/data_collector.py
```

**Controls:**
- Press **A-Z** keys to record samples for each letter
- Press **Q** to quit and save data
- Each key press records the current hand position

Collect at least 50-100 samples per letter for best results.

### Step 2: Train the Model

Train the classifier on your collected data:

```bash
python src/train_model.py
```

This will create `models/asl_classifier.pkl` with the trained model.

### Step 3: Run the Translator

Launch the real-time translator:

```bash
python src/translator.py
```

**Controls:**
- **SPACE**: Add a space to the text
- **BACKSPACE**: Delete the last character
- **ENTER**: Speak the accumulated text
- **C**: Clear all text
- **Q**: Quit the application

## Project Structure

```
SignLanguage/
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── src/
│   ├── __init__.py
│   ├── data_collector.py     # Tool to collect training data
│   ├── hand_detector.py      # MediaPipe hand landmark extraction
│   ├── train_model.py        # Train the classifier
│   ├── translator.py         # Main real-time translator app
│   └── text_to_speech.py     # TTS utility
├── models/
│   └── asl_classifier.pkl    # Trained model (generated)
└── data/
    └── asl_dataset.pkl       # Collected hand landmark data
```

## Tips for Better Recognition

1. **Good Lighting**: Ensure your hand is well-lit and visible
2. **Plain Background**: A simple background helps with hand detection
3. **Consistent Hand Position**: Keep your hand at a comfortable distance from the camera
4. **Multiple Angles**: When collecting data, vary hand angles slightly for robustness
5. **Stable Predictions**: The translator requires a letter to be held stable for 1 second before adding it

## Dependencies

- **OpenCV**: Webcam capture and image display
- **MediaPipe**: Hand landmark detection
- **NumPy**: Numerical operations
- **scikit-learn**: Machine learning classifier
- **pyttsx3**: Text-to-speech synthesis

## License

MIT License

