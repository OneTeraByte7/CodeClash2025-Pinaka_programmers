import cv2
import numpy as np
import os
import joblib
import time
import pandas as pd

MODEL_PATH = "car_risk_model.pkl"  
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(" Error: Trained model not found! Train and save the model first.")

model = joblib.load(MODEL_PATH)
print(" Model Loaded Successfully!")


CASCADE_PATH = "F:/CodeClash2025/haarcascade_car.xml"  
if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(f" Error: Missing '{CASCADE_PATH}'. Download it manually.")

car_cascade = cv2.CascadeClassifier(CASCADE_PATH)


CSV_FILE = "risk_analysis_data.csv"
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["Timestamp", "Distance (m)", "Risk Level", "Threat"]).to_csv(CSV_FILE, index=False)


def estimate_distance(w):
    """Estimate distance based on detected car width."""
    FOCAL_LENGTH = 850 
    REAL_WIDTH = 1.8  
    return round((REAL_WIDTH * FOCAL_LENGTH) / w, 2)


def predict_risk(car_region):
    """Predict risk level using detected car region only."""
    car_region = cv2.resize(car_region, (64, 64)).flatten().reshape(1, -1) / 255.0 
    return model.predict(car_region)[0] 


VIDEO_PATH = "./videos/today.mp4"  
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(" Error: Cannot open video file!")

print("Processing video... Press 'Q' to exit.")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(" End of video.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))  

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  
    detected_car = False  

    for (x, y, w, h) in cars:
        detected_car = True  
        car_region = gray[y:y + h, x:x + w]  
        risk_level = predict_risk(car_region)  
        distance = estimate_distance(w)  

        
        if risk_level == 1:
            if distance < 8:
                threat_level, color = "CRITICAL", (0, 0, 255)  
                cv2.putText(frame, "DANGER! COLLISION IMMINENT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            elif distance < 15:
                threat_level, color = "HIGH", (0, 140, 255) 
                cv2.putText(frame, "WARNING: HIGH RISK!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            else:
                threat_level, color = "MEDIUM", (0, 255, 255)  
                cv2.putText(frame, "Moderate Risk", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        else:
            threat_level, color = "LOW", (0, 255, 0)  #

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"Dist: {distance}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        
        risk_data = pd.DataFrame([[timestamp, distance, risk_level, threat_level]], 
                                 columns=["Timestamp", "Distance (m)", "Risk Level", "Threat"])
        risk_data.to_csv(CSV_FILE, mode='a', header=False, index=False)

    if not detected_car:
        cv2.putText(frame, "No Cars Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    
    cv2.imshow(" Car Risk Analysis", frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print(" Exiting...")
        break


cap.release()
cv2.destroyAllWindows()
print(f" Risk analysis data saved to {CSV_FILE}")
