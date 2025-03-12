
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import torch
import base64
from io import BytesIO

app = Flask(__name__)

# Load YOLOv8 (this downloads weights if not available locally)
# You can specify a path or URL to a custom model if needed
model = YOLO("yolov8n.pt")

# If you want to force CPU usage or move to GPU if available, you can:
# model.to("cuda")  # Uncomment if using a GPU instance on Render
# model.to("cpu")   # Force CPU

@app.route("/", methods=["GET"])
def index():
    return "Welcome to the YOLOv8 inference API!"

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]

    # Load image via PIL
    image = Image.open(file.stream).convert("RGB")

    # Run YOLOv8 inference
    # results is a list of Results objects (one per image)
    results_list = model.predict(image)

    if not results_list:
        return jsonify({"error": "No inference results"}), 500

    # Get the first Results object (we only sent one image)
    results = results_list[0]

    # The .plot() method returns a numpy array (with bounding boxes drawn)
    annotated_np = results.plot()

    # Convert the numpy array to a PIL Image
    annotated_img = Image.fromarray(annotated_np)

    # Convert to bytes
    img_io = BytesIO()
    annotated_img.save(img_io, format="JPEG")
    img_io.seek(0)

    # Encode as base64 string
    b64_string = base64.b64encode(img_io.read()).decode("utf-8")

    # Return JSON containing the base64-encoded annotated image
    return jsonify({"boxed_image": b64_string})

if __name__ == "__main__":
    # For local testing; on Render you'll specify this in the Start Command
    app.run(host="0.0.0.0", port=5000)
