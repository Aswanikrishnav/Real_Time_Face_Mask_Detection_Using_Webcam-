# Face Mask Detection with Live Alert System

## Objective

Detect if people are wearing face masks in real-time using a webcam. The system alerts users if no mask is detected.

## Features
- Real-time face mask detection using webcam
- Live alerts when mask is not worn
- Deployed with Flask web app
- Uses pre-trained MobileNetV2 with fine-tuned layers


## Installation
1. **Clone the repository**
```bash
git clone <MY_REPO_URL>

2. **Navigate to project folder**  
cd face_mask_app

3. **Create and activate a virtual environment**

python -m venv venv
.\venv\Scripts\activate  # Windows

4. **Install dependencies**

pip install -r requirements.txt

```bash
Run the flask app

python flask_app.py

- Open your browser at: http://127.0.0.1:5000

- Webcam feed will start, and mask detection begins automatically.

- Press 'q'(Ctrl+c in terminal) in the video window to quit.

## Model Training
- Model used: MobileNetV2 (pretrained on ImageNet)
- Fine-tuned for binary mask/no-mask classification
- Dataset: Kaggle Face Mask Dataset
- Notebook for training: `train_mask_detection.ipynb`

## Demo
Demo video showing live detection is included: `demo.mp4`



