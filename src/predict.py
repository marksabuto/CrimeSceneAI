import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load Model
model = BloodstainModel()  # Reuse the class from train.py
model.load_state_dict(torch.load("models/bloodstain_model.pth"))
model.eval()

# Prediction Function
def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        return "Bloodstain" if predicted == 0 else "No Bloodstain"

# Test
result = predict("test_image.jpg")  # Replace with your image
print(f"Prediction: {result}")