# ASL Sign Language Translator

A real-time American Sign Language (ASL) alphabet translator that uses MediaPipe for hand tracking, a trained classifier for letter recognition, and text-to-speech for audio output.

## Features

- **Real-time Hand Tracking**: Uses MediaPipe Hands to detect and track 21 hand landmarks
- **ASL Alphabet Recognition**: Recognizes letters A-Z from hand gestures
- **Text-to-Speech**: Speaks accumulated text using pyttsx3
- **Data Collection Tool**: Built-in tool to collect training data for custom gestures
- **Visual Feedback**: Live webcam feed with hand skeleton overlay and confidence display
- **Kaggle Dataset Support**: Train on the ASL Alphabet dataset from Kaggle

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

## Training Options

You have two options for training the model:

### Option A: Train on Kaggle Dataset (Recommended for Quick Start)

Train on the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle. This is the easiest way to get a working model.

#### Using Google Colab

1. Open a new Google Colab notebook
2. Run these cells:

```python
# Cell 1: Install dependencies
!pip install mediapipe scikit-learn kagglehub opencv-python

# Cell 2: Download dataset
import kagglehub
dataset_path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("Dataset path:", dataset_path)

# Cell 3: Upload train_colab.py or copy its contents, then:
# (Upload the train_colab.py file from this repo)
!python train_colab.py

# Cell 4: Download the trained model
from google.colab import files
files.download("asl_classifier.pkl")
```

3. Copy the downloaded `asl_classifier.pkl` to the `models/` folder

#### Using Local Python

```python
from src.train_model import train_from_kaggle
import kagglehub

# Download dataset
path = kagglehub.dataset_download("grassknoted/asl-alphabet")

# Train (adjust max_images_per_class for speed vs accuracy)
trainer, accuracy = train_from_kaggle(path, "models/asl_classifier.pkl", max_images_per_class=500)
```

### Option B: Collect Your Own Training Data

This gives you a model trained on your specific hand and environment.

#### Step 1: Collect Training Data

Run the data collection tool:

```bash
python src/data_collector.py
```

**Controls:**
- Press **A-Z** keys to record samples for each letter
- Press **Q** to quit and save data
- Each key press records the current hand position

Collect at least 50-100 samples per letter for best results.

#### Step 2: Train the Model

```bash
python src/train_model.py
```

This will create `models/asl_classifier.pkl` with the trained model.

## Running the Translator

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
├── train_colab.py            # Standalone Colab training script
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
- **kagglehub** (optional): For downloading Kaggle datasets

## License

MIT License
