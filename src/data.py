import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FashionDataset(Dataset):
    def __init__(self, images_csv_path, styles_csv_path, transform=None):
        # Read images CSV with enhanced parameters
        self.images_df = pd.read_csv(images_csv_path, 
                                     sep=',', 
                                     header=0, 
                                     encoding='utf-8', 
                                     engine='python',
                                     on_bad_lines='skip',
                                     quotechar='"',
                                     quoting=1)  # QUOTE_ALL
        
        # Read styles CSV with enhanced parameters
        self.styles_df = pd.read_csv(styles_csv_path, 
                                     sep=',', 
                                     header=0, 
                                     encoding='utf-8', 
                                     engine='python',
                                     on_bad_lines='skip',
                                     quotechar='"',
                                     quoting=1)  # QUOTE_ALL
        
        # Debugging: Display first few rows of each DataFrame
        print("Images DataFrame head:")
        print(self.images_df.head())
        print("\nStyles DataFrame head:")
        print(self.styles_df.head())
        
        # Clean filename column by removing file extensions
        self.images_df['filename'] = self.images_df['filename'].str.split('.').str[0]
        
        # Ensure both columns are string type
        self.images_df['filename'] = self.images_df['filename'].astype(str)
        self.styles_df['id'] = self.styles_df['id'].astype(str)
        
        # Debugging: Check unique values in merge columns
        print("\nUnique 'filename' values in images_df:", self.images_df['filename'].nunique())
        print("Unique 'id' values in styles_df:", self.styles_df['id'].nunique())
        print("\nCommon values between 'filename' and 'id':", 
              len(set(self.images_df['filename']).intersection(set(self.styles_df['id']))))
        
        # Drop rows with missing values
        self.images_df = self.images_df.dropna()
        self.styles_df = self.styles_df.dropna()

        # Reset index to avoid any index-related issues
        self.images_df = self.images_df.reset_index(drop=True)
        self.styles_df = self.styles_df.reset_index(drop=True)
        
        # Merge the two datasets on filename/id
        self.merged_df = pd.merge(self.images_df, self.styles_df, 
                                 left_on='filename', right_on='id', how='inner')
        
        # Debugging: Check merged DataFrame
        print("\nMerged DataFrame shape:", self.merged_df.shape)
        if self.merged_df.empty:
            raise ValueError("Merged DataFrame is empty. Check the merge columns and data.")
        
        # Filter out rows with missing images
        valid_indices = []
        for idx in range(len(self.merged_df)):
            filename = self.merged_df.iloc[idx]['filename']
            img_path = os.path.join('D:/projects/codemonk/archive/fashion-dataset/images', filename + '.jpg')
            if os.path.exists(img_path):
                valid_indices.append(idx)
            else:
                print(f"Missing image: {img_path}")
        
        self.merged_df = self.merged_df.iloc[valid_indices].reset_index(drop=True)
        print(f"Dataset contains {len(self.merged_df)} valid samples after filtering")
        
        # Create label mappings
        self.color_labels = sorted(self.merged_df['baseColour'].unique())
        self.color_to_idx = {color: idx for idx, color in enumerate(self.color_labels)}
        
        self.product_type_labels = sorted(self.merged_df['articleType'].unique())
        self.product_type_to_idx = {ptype: idx for idx, ptype in enumerate(self.product_type_labels)}
        
        self.season_labels = sorted(self.merged_df['season'].unique())
        self.season_to_idx = {season: idx for idx, season in enumerate(self.season_labels)}
        
        self.gender_labels = sorted(self.merged_df['gender'].unique())
        self.gender_to_idx = {gender: idx for idx, gender in enumerate(self.gender_labels)}
        
        self.transform = transform
        
    def __len__(self):
        return len(self.merged_df)
    
    def __getitem__(self, idx):
        row = self.merged_df.iloc[idx]
        
        # Construct image path
        img_path = os.path.join('D:/projects/codemonk/archive/fashion-dataset/images', row['filename'] + '.jpg')
        
        # Verify image exists
        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            return None
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        color_label = self.color_to_idx[row['baseColour']]
        product_type_label = self.product_type_to_idx[row['articleType']]
        season_label = self.season_to_idx[row['season']]
        gender_label = self.gender_to_idx[row['gender']]
        
        return {
            'image': image,
            'color': torch.tensor(color_label, dtype=torch.long),
            'product_type': torch.tensor(product_type_label, dtype=torch.long),
            'season': torch.tensor(season_label, dtype=torch.long),
            'gender': torch.tensor(gender_label, dtype=torch.long)
        }

def get_data_loaders(batch_size=32, image_size=224):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.229], std=[0.224, 0.225, 0.406])
    ])
    
    # Create dataset
    dataset = FashionDataset(
        images_csv_path='D:/projects/codemonk/archive/fashion-dataset/images.csv',
        styles_csv_path='D:/projects/codemonk/archive/fashion-dataset/styles.csv',
        transform=transform
    )
    
    # Debugging: Check dataset length
    print("Dataset length:", len(dataset))
    
    # Split dataset
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check data loading and merging process.")
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, dataset