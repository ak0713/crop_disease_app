
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import json
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# 1. Hardware Acceleration (Uses Mac MPS if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

# 2. Data Augmentation and Normalization
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load Data from the filtered paths
train_dir = 'filtered_dataset/train'
val_dir = 'filtered_dataset/val'

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

# Save class mapping to JSON for later use in the FastAPI server
class_names = train_data.classes
with open('class_indices.json', 'w') as f:
    json.dump(class_names, f)

# 4. Initialize ResNet50 Transfer Learning
model = models.resnet50(weights='IMAGENET1K_V1')

# Freeze base layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer to match our number of plant classes (38)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

# 5. Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 6. The Training Loop
epochs = 5
best_accuracy = 0.0

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    print("-" * 10)
    
    # Training phase
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(train_data)
    
    # Validation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_accuracy = 100 * correct / total
    print(f"Train Loss: {epoch_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")
    
    # Save the best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_crop_model.pth')
        print(">> Saved new best model!")
