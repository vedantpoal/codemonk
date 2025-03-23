Fashion Product Classifier

Table of Contents
Project Overview
Dataset
Installation
Usage
Model Architecture
Training
API
Streamlit App
Contributing

Project Overview
This project implements a deep learning-based fashion product classifier that can predict the following attributes from an image:

Color
Product Type (e.g., T-shirt, shoes)
Season (e.g., Summer, Winter)
Gender (Men, Women, Unisex)

The model uses a multi-task learning approach with a ResNet-50 backbone.
Dataset
The project uses the Fashion Product Images Dataset which contains:
44,424 product images
Metadata including product type, color, season, and gender

Installation
Clone the repository:
bash
git clone https://github.com/vedantpoal/fashion-product-classifier.git
cd fashion-product-classifier

Create a virtual environment (recommended):
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
bash
pip install -r requirements.txt

Usage
Training the Model
bash
python src/train.py

Making Predictions
Python
from src.inference import FashionPredictor

predictor = FashionPredictor(
    model_path='D:/projects/codemonk/models/best_model.pth',
    images_csv_path='D:/projects/codemonk/archive/fashion-dataset/images.csv',
    styles_csv_path='D:/projects/codemonk/archive/fashion-dataset/styles.csv'
)

predictions = predictor.predict('path/to/your/image.jpg')
print(predictions)

Using the API
Start the API server:
bash
python src/api.py

Send a POST request with an image:
bash
curl -X POST -F "image=@path/to/your/image.jpg" http://localhost:5000/predict

Using the Streamlit App
Start the Streamlit app:
bash
streamlit run src/app.py

Upload an image through the web interface to get predictions.
Model Architecture
The model uses a multi-task learning approach with a ResNet-50 backbone:
Python
Copy
class MultiTaskModel(nn.Module):
    def __init__(self, num_colors, num_product_types, num_seasons, num_genders):
        super(MultiTaskModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        # Custom heads for each task
        self.color_head = nn.Sequential(...)
        self.product_type_head = nn.Sequential(...)
        self.season_head = nn.Sequential(...)
        self.gender_head = nn.Sequential(...)
        
Training
To train the model:
Ensure your dataset is properly structured
Run the training script:
bash
python src/train.py

API
The API provides a simple endpoint for predictions:
POST /predict: Accepts an image file and returns predictions

Streamlit App
The Streamlit app provides a user-friendly interface to:
Upload images
Display predictions
Visualize results
