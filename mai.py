import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


model = YOLO("yolov8n.pt")

# Preprocessing Function (Dehazing, Contrast Enhancement, Noise Reduction)
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    
   
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    
    return Image.fromarray(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))

# Load and Preprocess Image
image_path = "C:/Users/rohit/OneDrive/Desktop/kacha/dataset/images/train/foggy image.jpg"  
preprocessed_image = preprocess_image(image_path)


results = model(preprocessed_image)


for result in results:
    result.show()
    result.save("output.jpg")
    

def evaluate_model(model, dataset_path):
    # Placeholder: Compute mAP, IoU on dataset
    print("Evaluating model...")

evaluate_model(model, "dataset/val")

print("Detection Complete!")
