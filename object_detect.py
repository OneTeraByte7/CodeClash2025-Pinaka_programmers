import cv2
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Define dataset path
dataset_path = "frames"
categories = ["safe", "risky"]
labels = {"safe": 0, "risky": 1}  


CASCADE_PATH = "F:/CodeClash2025/haarcascade_car.xml"  
if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(f"Error: Missing '{CASCADE_PATH}'. Download it manually.")

car_cascade = cv2.CascadeClassifier(CASCADE_PATH)


for category in categories:
    folder = os.path.join(dataset_path, category)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Error: Required folder '{folder}' is missing!")

X, y = [], []


for category in categories:
    folder = os.path.join(dataset_path, category)
    images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if not images:
        print(f"Warning: No images found in '{folder}', skipping...")
        continue

    for file in images:
        file_path = os.path.join(folder, file)

        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Failed to load '{file_path}', skipping...")
                continue

            
            cars = car_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

            if len(cars) == 0:
                print(f"Skipping '{file_path}' (No cars detected).")
                continue  

        
            x, y_coord, w, h = cars[0]
            car_region = img[y_coord:y_coord + h, x:x + w]  

            car_region = cv2.resize(car_region, (64, 64))  
            car_region = car_region.flatten() / 255.0  
            X.append(car_region)
            y.append(labels[category])

        except Exception as e:
            print(f"Error loading image '{file_path}': {e}")


X, y = np.array(X), np.array(y)


if len(X) == 0:
    raise ValueError("Error: No valid car images found. Ensure 'frames/safe' and 'frames/risky' contain car images.")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print(" Model training complete!")


test_folder = os.path.join(dataset_path, "risky")  
test_images = [f for f in os.listdir(test_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))][:5] 

if not test_images:
    print("No test images found for prediction.")
else:
    print("\n Predictions on detected cars:")

    for test_file in test_images:
        test_path = os.path.join(test_folder, test_file)
        img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            cars = car_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

            if len(cars) == 0:
                print(f"Skipping '{test_file}' (No cars detected).")
                continue

            x, y_coord, w, h = cars[0]
            car_region = img[y_coord:y_coord + h, x:x + w]
            car_region = cv2.resize(car_region, (64, 64)).flatten().reshape(1, -1) / 255.0

            prediction = model.predict(car_region)
            print(f"📌 {test_file}: {'RISKY' if prediction[0] == 1 else 'SAFE'}")
        else:
            print(f"Error: Unable to read '{test_path}'.")

print("\n Model execution complete!")
