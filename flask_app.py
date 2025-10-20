# app.py
from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model

app = Flask(__name__)

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = os.path.join(os.getcwd(), "face_mask_model_balanced.h5")
HAAR_PATH = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")
INPUT_SIZE = (224, 224)   # change if your model expects another shape
THRESHOLD = 0.5           # adjust if needed

# -------------------------
# Load model and cascade
# -------------------------
model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(HAAR_PATH)

# Open webcam (0 = default camera)
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Could not start camera. Check camera access and ID.")

last_alert = 0.0

def gen_frames():
    global last_alert
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            # Preprocess to your model's requirement
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, INPUT_SIZE)
            face_input = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_input, axis=0)

            preds = model.predict(face_input)
            # model outputs: sigmoid (1) or softmax (2). Try both guards.
            if preds.shape[-1] == 1:
                prob_no_mask = preds[0][0]
                label = "No Mask" if prob_no_mask >= THRESHOLD else "Mask"
                confidence = prob_no_mask if label == "No Mask" else 1 - prob_no_mask
            else:
                # Assume [mask_prob, no_mask_prob]
                mask_prob, no_mask_prob = float(preds[0][0]), float(preds[0][1])
                label = "No Mask" if no_mask_prob >= mask_prob else "Mask"
                confidence = no_mask_prob if label == "No Mask" else mask_prob

            color = (0, 0, 255) if label == "No Mask" else (0, 255, 0)
            text = f"{label}: {confidence*100:.1f}%"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # On-screen big warning when No Mask
            if label == "No Mask":
                cv2.putText(frame, "⚠ PLEASE WEAR A MASK!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                # Console log no more often than every 2 seconds
                now = time.time()
                if now - last_alert > 2.0:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ALERT: No mask detected (confidence {confidence:.2f})")
                    last_alert = now

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Use host='0.0.0.0' to allow other devices on the LAN to access (optional)
    app.run(host='0.0.0.0', port=5000, debug=True)
