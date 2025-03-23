# src/api.py
from flask import Flask, request, jsonify
from inference import FashionPredictor
import cv2
import os

app = Flask(__name__)

# Initialize the predictor with your model and data paths
predictor = FashionPredictor(
    model_path='D:/projects/codemonk/models/best_model.pth',
    images_csv_path='D:/projects/codemonk/archive/fashion-dataset/images.csv',
    styles_csv_path='D:/projects/codemonk/archive/fashion-dataset/styles.csv'
)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    # Check if the filename is empty
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Save the temporary image
    temp_path = 'temp_image.jpg'
    file.save(temp_path)
    
    # Make prediction using the saved image
    predictions = predictor.predict(temp_path)
    
    # Clean up the temporary image
    os.remove(temp_path)
    
    # Return the predictions as JSON
    return jsonify(predictions)

if __name__ == '__main__':
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000)