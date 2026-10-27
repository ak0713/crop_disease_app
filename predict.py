import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import sys

# 1. Setup Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 2. Load the Class Names
with open('class_indices.json', 'r') as f:
    classes = json.load(f)

# 3. Load Model Architecture and Custom Weights
model = models.resnet50(weights=None) # We don't need the internet weights anymore
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load('best_crop_model.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval() # Lock the weights for testing

# 4. Image Preprocessing and Prediction
def predict_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return
        
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device) 
    
    # 5. Run Inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    
    print(f"\nAnalysis for: {image_path}")
    print("-" * 30)
    for i in range(top3_prob.size(0)):
        class_name = classes[top3_catid[i]]
        confidence = top3_prob[i].item() * 100
        print(f"{i+1}. {class_name}: {confidence:.2f}%")
    print("\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_image(sys.argv[1])
    else:
        print("Usage: python predict.py <path_to_image.jpg>")
