import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from utils import preprocess_image, CLASS_NAMES
import sys
import os

# Add the parent folder (project root) to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.cnn_model import CNNModel

@st.cache_resource
def load_model():
    model = CNNModel()
    model.load_state_dict(torch.load("saved_model/best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

st.title("Image Classifier - CIFAR10")
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(1).item()

    st.success(f"Prediction: {CLASS_NAMES[prediction]}")

