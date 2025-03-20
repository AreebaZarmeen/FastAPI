import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Define YOLO model file paths
pretrained_model_path = "yolo11s.pt"
custom_model_path = "customYOLO11s.pt"

print(f"🔍 Checking for YOLO model files...")
print(f"➡️ Pretrained model path: {pretrained_model_path}")
print(f"➡️ Custom model path: {custom_model_path}")

# Load YOLO models only if files exist
pretrained_model, custom_model = None, None

if os.path.exists(pretrained_model_path):
    print("✅ Pretrained model found, loading...")
    pretrained_model = YOLO(pretrained_model_path)
else:
    print(f"❌ Error: {pretrained_model_path} not found. Upload this file to Railway.")

if os.path.exists(custom_model_path):
    print("✅ Custom model found, loading...")
    custom_model = YOLO(custom_model_path)
else:
    print(f"❌ Error: {custom_model_path} not found. Upload this file to Railway.")

@app.route("/")
def home():
    return "✅ Server is running"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"].read()
    np_img = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image format"}), 400

    if pretrained_model is None and custom_model is None:
        return jsonify({"error": "YOLO models not loaded"}), 500

    predictions = []

    # Perform YOLO inference using the pretrained model (if available)
    if pretrained_model:
        results_pretrained = pretrained_model.predict(img)
        for result in results_pretrained:
            for box in result.boxes:
                x1, y1, x2, y2 = [round(coord, 3) for coord in box.xyxy[0].tolist()]
                label = result.names[int(box.cls[0])]
                predictions.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "model": "pretrained"})

    # Perform YOLO inference using the custom model (if available)
    if custom_model:
        results_custom = custom_model.predict(img)
        for result in results_custom:
            for box in result.boxes:
                x1, y1, x2, y2 = [round(coord, 3) for coord in box.xyxy[0].tolist()]
                label = result.names[int(box.cls[0])]
                predictions.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "model": "custom"})

    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Railway's PORT or default to 8080
    app.run(host="0.0.0.0", port=port, debug=False)
