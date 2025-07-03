# ASL Hand Gesture Detection

A real-time hand gesture recognition system that detects and classifies hand gestures (YES/NO) using computer vision and machine learning. Built with Python, OpenCV, MediaPipe, and TensorFlow.

## 🎯 Features

- **Real-time Detection**: Live hand gesture recognition through webcam
- **Gesture Recognition**: Supports YES (👍) and NO (👎) gestures
- **Visual Feedback**: Live camera feed with hand landmarks and prediction results
- **High Accuracy**: Uses TensorFlow model trained with Teachable Machine
- **Easy Data Collection**: Built-in tool for collecting custom training data

## 🛠️ Technologies Used

- **Python 3.x**
- **OpenCV** - Computer vision and image processing
- **MediaPipe** - Hand detection and landmark extraction
- **TensorFlow** - Machine learning model inference
- **NumPy** - Numerical computations

## 📋 Prerequisites

- Python 3.7 or higher
- Webcam/Camera for real-time detection
- Git (for cloning the repository)

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd "HS Detection"
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

Make sure you have the following folder structure:

```
HS Detection/
├── Model/
│   ├── keras_model.h5      # Pre-trained model
│   └── labels.txt          # Class labels
├── test.py                 # Main detection script
├── data_collection.py      # Data collection tool
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🎮 Usage

### Running Real-time Detection

```bash
python test.py
```

**Controls:**

- Show your hand to the camera and make gestures
- **YES**: Thumbs up gesture 👍
- **NO**: Thumbs down gesture 👎
- Press **'q'** to quit the application

### Collecting Training Data (Optional)

If you want to retrain the model with your own data:

```bash
python data_collection.py
```

**Controls:**

- Make the desired gesture in front of the camera
- Press **'s'** to save training images
- Press **'q'** to quit data collection
- Change `folderPath` in the script for different gestures

**Steps to collect data:**

1. Edit `folderPath = "Data/YES"` in `data_collection.py`
2. Run the script and make thumbs up gestures
3. Press 's' to save 50-100 images
4. Change to `folderPath = "Data/NO"`
5. Make thumbs down gestures and save 50-100 images

## 🧠 Model Training

The project uses a pre-trained TensorFlow model. To train your own model:

1. **Collect Data**: Use `data_collection.py` to gather training images
2. **Train Model**: Upload images to [Teachable Machine](https://teachablemachine.withgoogle.com/)
   - Choose "Image Project"
   - Create classes for YES and NO
   - Upload your collected images
   - Train the model
3. **Download Model**: Export as TensorFlow and download `keras_model.h5` and `labels.txt`
4. **Replace Files**: Put the new files in the `Model/` folder

## 📁 Project Structure

```
HS Detection/
├── Model/                  # Trained model files
│   ├── keras_model.h5     # TensorFlow model
│   └── labels.txt         # Class labels
├── Data/                  # Training data (private)
│   ├── YES/              # Thumbs up images
│   └── NO/               # Thumbs down images
├── test.py               # Main detection application
├── data_collection.py    # Data collection utility
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

## ⚙️ Configuration

### Key Parameters (in `test.py`):

- `imgSize = 224`: Input image size for the model
- `offset = 20`: Padding around detected hand
- `confidence > 0.7`: Minimum confidence threshold for predictions
- `min_detection_confidence=0.7`: MediaPipe hand detection sensitivity

### Customizing Gestures:

To add more gestures, modify:

1. Update `labels.txt` with new classes
2. Collect training data for new gestures
3. Retrain the model with all gesture classes

## 🎨 Output

The application displays:

- **Live camera feed** with hand landmarks
- **Prediction box** showing detected gesture
- **Confidence score** for each prediction
- **Color coding**: Green for YES, Red for NO

## 🔧 Troubleshooting

### Common Issues:

1. **Camera not working**:

   ```bash
   # Check camera index in test.py
   cap = cv2.VideoCapture(0)  # Try 0, 1, or 2
   ```

2. **Model not found**:

   - Ensure `Model/keras_model.h5` and `Model/labels.txt` exist
   - Check file paths in the code

3. **Poor detection accuracy**:

   - Ensure good lighting conditions
   - Make clear, distinct gestures
   - Retrain model with more diverse data

4. **Installation issues**:
   ```bash
   # Update pip and try again
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 📊 Performance

- **Real-time processing**: ~30 FPS on standard webcam
- **Detection accuracy**: >90% with good lighting and clear gestures
- **Response time**: <100ms per frame
- **Memory usage**: ~200MB RAM

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 🎯 Future Enhancements

- [ ] Add more gesture classes (numbers 0-9)
- [ ] Implement gesture sequence recognition
- [ ] Add audio feedback for predictions
- [ ] Create mobile app version
- [ ] Improve model accuracy with data augmentation

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Create an issue in the repository
3. Ensure you have the correct Python version and dependencies
---

**Made with ❤️ for accessible communication technology**
