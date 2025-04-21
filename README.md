# CrimeSceneAI

A machine learning project for detecting bloodstains in images using computer vision and deep learning techniques.

## Project Overview

CrimeSceneAI is designed to assist in crime scene analysis by automatically detecting bloodstains in images. The system uses a Convolutional Neural Network (CNN) trained on a dataset of bloodstain and non-bloodstain images.

## Features

- Automated bloodstain detection in images
- High-accuracy CNN model
- Easy-to-use image processing pipeline
- Support for various image formats
- Real-time detection capabilities

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/marksabuto/CrimeSceneAI.git
cd CrimeSceneAI
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Setup

1. Get an Unsplash Access Key:
   - Visit https://unsplash.com/developers
   - Register as a developer
   - Copy your Access Key

2. Update the script with your Access Key:
   - Open `src/download_images.py`
   - Replace `'Client-ID YOUR_ACCESS_KEY'` with your actual Unsplash Access Key

## Usage

### Downloading Training Images

To download images for training the model:
```bash
python src/download_images.py
```

This will:
- Create `data/bloodstain` and `data/no_bloodstain` directories
- Download bloodstain and non-bloodstain images from Unsplash
- Preprocess and save the images in the appropriate directories

### Training the Model

To train the bloodstain detection model:
```bash
python src/train.py
```

This will:
- Load and preprocess the training images
- Train the CNN model
- Save the trained model to `models/bloodstain_detector.pth`

## Project Structure

```
CrimeSceneAI/
├── data/                  # Training data
│   ├── bloodstain/       # Images containing bloodstains
│   └── no_bloodstain/    # Images without bloodstains
├── models/               # Trained models
├── src/                  # Source code
│   ├── download_images.py # Image download script
│   └── train.py          # Model training script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Model Architecture

The bloodstain detection model uses a CNN with the following architecture:
- 3 convolutional layers with increasing depth
- Max pooling layers for dimensionality reduction
- Dropout for regularization
- Fully connected layers for classification

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Images sourced from Unsplash
- Built with PyTorch and OpenCV
- Inspired by forensic science applications

