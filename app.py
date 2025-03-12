import torch
from ultralytics import YOLO
from flask import Flask, request, jsonify
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

# 2) Load the YOLOv8 model
model = YOLO("train3/weights/best.pt")  # or your custom .pt file
# If Render has a GPU plan, you can do model.to("cuda")

@app.route("/", methods=["GET"])
def index():
    return "Welcome to YOLOv8!"

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")

    # 3) Run inference
    results_list = model.predict(image)
    if not results_list:
        return jsonify({"error": "No inference results"}), 500

    # 4) Annotate image
    results = results_list[0]
    num_bboxes = len(results.boxes)
    annotated_np = results.plot()  # draws bounding boxes on the image
    annotated_img = Image.fromarray(annotated_np)

    # 5) Convert to base64 and return
    img_io = BytesIO()
    annotated_img.save(img_io, format="JPEG")
    img_io.seek(0)
    b64_string = base64.b64encode(img_io.read()).decode("utf-8")

    return jsonify({"boxed_image": b64_string})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
