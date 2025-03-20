{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "07ef7812-ff65-4705-b4f7-dfa02e638080",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading Pretrained YOLO model from: yolo11s.pt\n",
      "Loading Custom YOLO model from: customYOLO11s.pt\n",
      "✅ Both YOLO models loaded successfully.\n",
      " * Serving Flask app '__main__'\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on all addresses (0.0.0.0)\n",
      " * Running on http://127.0.0.1:5000\n",
      " * Running on http://192.168.0.101:5000\n",
      "Press CTRL+C to quit\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "from flask import Flask, request, jsonify\n",
    "import cv2\n",
    "import numpy as np\n",
    "from ultralytics import YOLO\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "# Load YOLO models with absolute paths\n",
    "pretrained_model_path = (\"yolo11s.pt\")\n",
    "custom_model_path = (\"customYOLO11s.pt\")\n",
    "\n",
    "print(f\"Loading Pretrained YOLO model from: {pretrained_model_path}\")\n",
    "print(f\"Loading Custom YOLO model from: {custom_model_path}\")\n",
    "\n",
    "try:\n",
    "    pretrained_model = YOLO(pretrained_model_path)  # Load Pretrained YOLO model\n",
    "    custom_model = YOLO(custom_model_path)  # Load Custom YOLO model\n",
    "    print(\"✅ Both YOLO models loaded successfully.\")\n",
    "except Exception as e:\n",
    "    print(f\"❌ Error loading YOLO models: {e}\")\n",
    "    pretrained_model = None\n",
    "    custom_model = None\n",
    "\n",
    "@app.route(\"/\")\n",
    "def home():\n",
    "    return \"Server is running\"\n",
    "\n",
    "@app.route(\"/predict\", methods=[\"POST\"])\n",
    "def predict():\n",
    "    if \"image\" not in request.files:\n",
    "        return jsonify({\"error\": \"No image uploaded\"}), 400\n",
    "\n",
    "    file = request.files[\"image\"].read()\n",
    "    np_img = np.frombuffer(file, np.uint8)\n",
    "    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)\n",
    "\n",
    "    if img is None:\n",
    "        return jsonify({\"error\": \"Invalid image format\"}), 400\n",
    "\n",
    "    if pretrained_model is None or custom_model is None:\n",
    "        return jsonify({\"error\": \"YOLO models not loaded\"}), 500\n",
    "\n",
    "    # Perform YOLO inference using both models\n",
    "    results_pretrained = pretrained_model.predict(img)\n",
    "    results_custom = custom_model.predict(img)\n",
    "\n",
    "    predictions = []\n",
    "\n",
    "    # Process results from the pretrained model\n",
    "    for result in results_pretrained:\n",
    "        for box in result.boxes:\n",
    "            x1, y1, x2, y2 = [round(coord, 3) for coord in box.xyxy[0].tolist()]\n",
    "            label = result.names[int(box.cls[0])]\n",
    "            predictions.append({\"label\": label, \"x1\": x1, \"y1\": y1, \"x2\": x2, \"y2\": y2, \"model\": \"pretrained\"})\n",
    "\n",
    "    # Process results from the custom model\n",
    "    for result in results_custom:\n",
    "        for box in result.boxes:\n",
    "            x1, y1, x2, y2 = [round(coord, 3) for coord in box.xyxy[0].tolist()]\n",
    "            label = result.names[int(box.cls[0])]\n",
    "            predictions.append({\"label\": label, \"x1\": x1, \"y1\": y1, \"x2\": x2, \"y2\": y2, \"model\": \"custom\"})\n",
    "\n",
    "    return jsonify({\n",
    "        \"predictions\": predictions\n",
    "    })\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run(host=\"0.0.0.0\", port=5000, debug=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8e54fc38-4cae-43bf-a728-d901253599d3",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[NbConvertApp] Converting notebook app.ipynb to script\n",
      "[NbConvertApp] Writing 2968 bytes to app.py\n"
     ]
    }
   ],
   "source": [
    "!jupyter nbconvert --to script app.ipynb\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "60ac76ea-b643-4826-b560-ad83008a0deb",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
