# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('saved_model/nutrition_model.h5')

# Create upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Prediction function
def predict_nutrition(image_path):
    img = Image.open(image_path).resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)
    prediction = model.predict(img_array)[0]
    return {
        "Calories": round(float(prediction[0]), 2),
        "Protein": round(float(prediction[1]), 2),
        "Fat": round(float(prediction[2]), 2),
        "Carbs": round(float(prediction[3]), 2)
    }

@app.route('/')
def home():
    return "🍎 Nutrition Estimator API is running!"

@app.route('/predict', methods=['POST'])
def upload_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Predict
    nutrition_data = predict_nutrition(filepath)
    return jsonify({
        "filename": file.filename,
        "nutrition_estimation": nutrition_data
    })

if __name__ == '__main__':
    app.run(debug=True)

