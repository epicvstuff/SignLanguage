# ASL Sign Language Translator

A real-time American Sign Language (ASL) alphabet translator that uses MediaPipe for hand tracking, machine learning for letter recognition, and text-to-speech for audio output.

## Features

- **Real-time Hand Tracking**: Uses MediaPipe Hands to detect and track 21 hand landmarks
- **ASL Alphabet Recognition**: Recognizes letters A-Z from hand gestures
- **Two Model Options**: 
  - Random Forest on landmarks (fast, lightweight)
  - CNN on images (more accurate, GPU-accelerated)
- **Text-to-Speech**: Speaks accumulated text
- **Data Collection Tool**: Built-in tool to collect training data for custom gestures
- **Visual Feedback**: Live webcam feed with hand skeleton overlay and confidence display

## Model Comparison

| Feature | Random Forest (Landmarks) | CNN (Images) |
|---------|--------------------------|--------------|
| Input | 21 hand landmarks (63 features) | Hand image (224x224) |
| Speed | Very fast | Moderate |
| Accuracy | Good (~85%) | Better (~95%) |
| GPU Required | No | Recommended |
| Model Size | ~5 MB | ~15 MB |
| Best For | Real-time on CPU | Maximum accuracy |

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
4. For CNN model, also install TensorFlow:
   ```bash
   pip install tensorflow
   ```

## Training Options

### Option 1: CNN Model (Recommended - Best Accuracy)

Train a MobileNetV2-based CNN on hand images for the best accuracy.

#### Using Google Colab (Recommended - Free GPU)

1. Open a new Google Colab notebook
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. Run these cells:

```python
# Cell 1: Install dependencies
!pip install tensorflow kagglehub

# Cell 2: Download dataset and train
import kagglehub
dataset_path = kagglehub.dataset_download("grassknoted/asl-alphabet")

# Cell 3: Clone repo and run training
!git clone https://github.com/epicvstuff/SignLanguage.git
%cd SignLanguage
!python train_cnn_colab.py

# Cell 4: Download the trained model
from google.colab import files
files.download("asl_cnn_model.keras")
files.download("asl_cnn_model_classes.json")
```

4. Move downloaded files to the `models/` folder

#### Using Local Python

```python
from src.train_cnn import train_cnn_model
import kagglehub

# Download dataset
path = kagglehub.dataset_download("grassknoted/asl-alphabet")

# Train CNN (uses MobileNetV2 transfer learning)
model, history = train_cnn_model(path, "models/asl_cnn_model.keras", epochs=20)
```

### Option 2: Random Forest Model (Faster Training)

Train a Random Forest classifier on hand landmark features.

#### Using Google Colab

```python
# Cell 1: Install dependencies
!pip install mediapipe scikit-learn kagglehub opencv-python

# Cell 2: Download dataset
import kagglehub
dataset_path = kagglehub.dataset_download("grassknoted/asl-alphabet")

# Cell 3: Clone and train
!git clone https://github.com/epicvstuff/SignLanguage.git
%cd SignLanguage
!python train_colab.py

# Cell 4: Download model
from google.colab import files
files.download("asl_classifier.pkl")
```

#### Using Local Python

```python
from src.train_model import train_from_kaggle
import kagglehub

path = kagglehub.dataset_download("grassknoted/asl-alphabet")
trainer, accuracy = train_from_kaggle(path, "models/asl_classifier.pkl", max_images_per_class=500)
```

### Option 3: Collect Your Own Training Data

For a model trained on your specific hand and environment:

```bash
# Step 1: Collect data (press A-Z to record, Q to quit)
python src/data_collector.py

# Step 2: Train Random Forest model
python src/train_model.py
```

## Running the Translator

Launch the real-time translator:

```bash
cd src
python translator.py
```

The translator automatically uses the CNN model if available, otherwise falls back to Random Forest.

**To specify a model:**
```bash
python translator.py ../models/asl_cnn_model.keras    # Use CNN
python translator.py ../models/asl_classifier.pkl     # Use Random Forest
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
├── train_colab.py            # Colab script for Random Forest
├── train_cnn_colab.py        # Colab script for CNN
├── src/
│   ├── __init__.py
│   ├── data_collector.py     # Tool to collect training data
│   ├── hand_detector.py      # MediaPipe hand landmark extraction
│   ├── train_model.py        # Random Forest training
│   ├── train_cnn.py          # CNN training (MobileNetV2)
│   ├── translator.py         # Main real-time translator app
│   └── text_to_speech.py     # TTS utility
├── models/
│   ├── asl_classifier.pkl        # Random Forest model (generated)
│   ├── asl_cnn_model.keras       # CNN model (generated)
│   └── asl_cnn_model_classes.json # CNN class mappings
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
- **scikit-learn**: Random Forest classifier
- **TensorFlow**: CNN model (optional, for better accuracy)
- **pyttsx3**: Text-to-speech synthesis
- **kagglehub**: For downloading Kaggle datasets

## License

MIT License
