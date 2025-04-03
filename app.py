import os
import numpy as np
import torch
import tensorflow as tf
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoTokenizer, TFAutoModel

# -------------------
# Flask Configuration
# -------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# -------------------
# Load Pretrained Models
# -------------------
print("Loading models...")

# Image Classification Model
image_processor = AutoImageProcessor.from_pretrained("GauravGrow/vit-base-hate-meme")
image_model = AutoModelForImageClassification.from_pretrained("GauravGrow/vit-base-hate-meme")

# Text Analysis Model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFAutoModel.from_pretrained("bert-base-uncased")

# CNN Model (Feature Extraction)
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))

print("Models loaded successfully!")

# -------------------
# Utility Functions
# -------------------
def allowed_file(filename):
    """Check if the file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_hate_meme(image_path):
    """Run hate meme detection using image and text analysis."""
    
    # Process image for CNN
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((224, 224))
    image_array = img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    # OCR to extract text
    extracted_text = pytesseract.image_to_string(image).strip() or "[NO TEXT]"

    # Tokenize the extracted text
    encoding = tokenizer(extracted_text, padding="max_length", truncation=True, max_length=128, return_tensors="tf")
    
    # Predict using the image classifier
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = image_model(**inputs)
    logits = outputs.logits
    image_prediction = torch.argmax(logits).item()

    # Predict using text model (Optional Fusion)
    text_features = bert_model(encoding["input_ids"], attention_mask=encoding["attention_mask"]).pooler_output
    image_features = resnet_model(image_array)
    combined_features = np.concatenate((image_features.numpy(), text_features.numpy()), axis=1)

    # Combine results (simplified decision logic)
    final_prediction = "Hate Meme Detected" if image_prediction == 1 else "Not a Hate Meme"
    
    return final_prediction

# -------------------
# Flask Routes
# -------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Run hate meme detection
        result = predict_hate_meme(file_path)
        return jsonify({"prediction": result})

    return jsonify({"error": "Invalid file format"}), 400

if __name__ == "_main_":
    app.run(debug=True)