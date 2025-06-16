import torch
from torchvision import transforms
from PIL import Image

# CIFAR-10 class labels
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Preprocessing pipeline
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension
