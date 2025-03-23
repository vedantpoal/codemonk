import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model import MultiTaskModel
from src.data import get_data_loaders

def train_model(num_epochs=10, batch_size=32, image_size=224, learning_rate=1e-4):
    # Get data loaders
    train_loader, val_loader, test_loader, dataset = get_data_loaders(batch_size, image_size)
    
    # Create model
    model = MultiTaskModel(
        num_colors=len(dataset.color_labels),
        num_product_types=len(dataset.product_type_labels),
        num_seasons=len(dataset.season_labels),
        num_genders=len(dataset.gender_labels)
    )
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define loss functions
    criterion = {
        'color': nn.CrossEntropyLoss(),
        'product_type': nn.CrossEntropyLoss(),
        'season': nn.CrossEntropyLoss(),
        'gender': nn.CrossEntropyLoss()
    }
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Train)')
        for batch in progress_bar:
            images = batch['image'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = (
                criterion['color'](outputs['color'], batch['color'].to(device)) +
                criterion['product_type'](outputs['product_type'], batch['product_type'].to(device)) +
                criterion['season'](outputs['season'], batch['season'].to(device)) +
                criterion['gender'](outputs['gender'], batch['gender'].to(device))
            )
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Val)'):
                images = batch['image'].to(device)
                
                outputs = model(images)
                
                loss = (
                    criterion['color'](outputs['color'], batch['color'].to(device)) +
                    criterion['product_type'](outputs['product_type'], batch['product_type'].to(device)) +
                    criterion['season'](outputs['season'], batch['season'].to(device)) +
                    criterion['gender'](outputs['gender'], batch['gender'].to(device))
                )
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f'Saved model with val loss {val_loss:.4f}')
    
    # Evaluate on test set
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()
    test_loss = 0.0
    correct = {'color': 0, 'product_type': 0, 'season': 0, 'gender': 0}
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images = batch['image'].to(device)
            outputs = model(images)
            
            loss = (
                criterion['color'](outputs['color'], batch['color'].to(device)) +
                criterion['product_type'](outputs['product_type'], batch['product_type'].to(device)) +
                criterion['season'](outputs['season'], batch['season'].to(device)) +
                criterion['gender'](outputs['gender'], batch['gender'].to(device))
            )
            
            test_loss += loss.item()
            
            # Calculate accuracy
            _, predicted_color = torch.max(outputs['color'], 1)
            _, predicted_product_type = torch.max(outputs['product_type'], 1)
            _, predicted_season = torch.max(outputs['season'], 1)
            _, predicted_gender = torch.max(outputs['gender'], 1)
            
            correct['color'] += (predicted_color == batch['color'].to(device)).sum().item()
            correct['product_type'] += (predicted_product_type == batch['product_type'].to(device)).sum().item()
            correct['season'] += (predicted_season == batch['season'].to(device)).sum().item()
            correct['gender'] += (predicted_gender == batch['gender'].to(device)).sum().item()
            
            total += batch['color'].size(0)
    
    test_loss /= len(test_loader)
    accuracy = {k: 100 * v / total for k, v in correct.items()}
    
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy - Color: {accuracy["color"]:.2f}%')
    print(f'Test Accuracy - Product Type: {accuracy["product_type"]:.2f}%')
    print(f'Test Accuracy - Season: {accuracy["season"]:.2f}%')
    print(f'Test Accuracy - Gender: {accuracy["gender"]:.2f}%')
    
    return model, dataset