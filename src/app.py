import streamlit as st
from inference import FashionPredictor
import cv2
import numpy as np
import tempfile
import os

# Initialize predictor with your model and data paths
predictor = FashionPredictor(
    model_path='D:/projects/codemonk/models/best_model.pth',
    images_csv_path='D:/projects/codemonk/archive/fashion-dataset/images.csv',
    styles_csv_path='D:/projects/codemonk/archive/fashion-dataset/styles.csv'
)

# Set up the Streamlit app
st.title("Fashion Product Classifier")
st.write("Upload an image of a fashion product and get predictions about its attributes.")

# File upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_image_path = tmp_file.name

    # Display the uploaded image
    image = cv2.imread(temp_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    with st.spinner('Making predictions...'):
        try:
            predictions = predictor.predict(temp_image_path)
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            os.remove(temp_image_path)
            raise

    # Clean up the temporary file
    os.remove(temp_image_path)

    # Display predictions
    st.subheader("Predictions:")
    st.write(f"Color: {predictions['color']}")
    st.write(f"Product Type: {predictions['product_type']}")
    st.write(f"Season: {predictions['season']}")
    st.write(f"Gender: {predictions['gender']}")