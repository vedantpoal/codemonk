import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskModel(nn.Module):
    def __init__(self, num_colors, num_product_types, num_seasons, num_genders):
        super(MultiTaskModel, self).__init__()
        
        # Load pre-trained ResNet-50
        self.backbone = models.resnet50(pretrained=True)
        
        # Replace the last fully connected layer
        num_features = self.backbone.fc.in_features
        
        # Create separate heads for each task
        self.color_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_colors)
        )
        
        self.product_type_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_product_types)
        )
        
        self.season_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_seasons)
        )
        
        self.gender_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_genders)
        )
        
        # Freeze backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last few layers
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        # Get features from backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Get predictions for each task
        color_pred = self.color_head(x)
        product_type_pred = self.product_type_head(x)
        season_pred = self.season_head(x)
        gender_pred = self.gender_head(x)
        
        return {
            'color': color_pred,
            'product_type': product_type_pred,
            'season': season_pred,
            'gender': gender_pred
        }