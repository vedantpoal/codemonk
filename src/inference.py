import cv2
import torch
import numpy as np
from data import FashionDataset  # Changed from "src.data" to "data"
from model import MultiTaskModel  # Changed from "src.model" to "model"
from torchvision import transforms

class FashionPredictor:
    def __init__(self, model_path='D:/projects/codemonk/models/best_model.pth', 
                 images_csv_path='D:/projects/codemonk/archive/fashion-dataset/images.csv', 
                 styles_csv_path='D:/projects/codemonk/archive/fashion-dataset/styles.csv'):
        # Load dataset to get label mappings
        self.dataset = FashionDataset(images_csv_path, styles_csv_path)
        
        # Create model
        self.model = MultiTaskModel(
            num_colors=len(self.dataset.color_labels),
            num_product_types=len(self.dataset.product_type_labels),
            num_seasons=len(self.dataset.season_labels),
            num_genders=len(self.dataset.gender_labels)
        )
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.229], std=[0.224, 0.225, 0.406])
        ])
    
    def predict(self, image_path):
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Process predictions
        predictions = {}
        
        # Color prediction
        _, predicted_color_idx = torch.max(outputs['color'], 1)
        predictions['color'] = self.dataset.color_labels[predicted_color_idx.item()]
        
        # Product type prediction
        _, predicted_product_type_idx = torch.max(outputs['product_type'], 1)
        predictions['product_type'] = self.dataset.product_type_labels[predicted_product_type_idx.item()]
        
        # Season prediction
        _, predicted_season_idx = torch.max(outputs['season'], 1)
        predictions['season'] = self.dataset.season_labels[predicted_season_idx.item()]
        
        # Gender prediction
        _, predicted_gender_idx = torch.max(outputs['gender'], 1)
        predictions['gender'] = self.dataset.gender_labels[predicted_gender_idx.item()]
        
        return predictions