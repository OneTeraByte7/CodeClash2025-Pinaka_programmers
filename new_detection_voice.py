import cv2
import numpy as np
import joblib
import time
import os
import pygame  # For audio alerts

# Load trained AI model
MODEL_PATH = "collision_risk_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(" Error: Trained model not found! Run training script first.")

model = joblib.load(MODEL_PATH)
print(" Model Loaded Successfully!")


CASCADE_PATH = "F:/CodeClash2025/haarcascade_car.xml"  
if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(f" Error: Missing '{CASCADE_PATH}'. Download it manually.")

car_cascade = cv2.CascadeClassifier(CASCADE_PATH)

pygame.mixer.init()
alert_sound = "alert.mp3"  

# Speed Estimation Variables
prev_time = time.time()
prev_distance = None
your_car_speed = 0 


def estimate_distance(w):
    """Estimate distance based on detected object width."""
    FOCAL_LENGTH = 850  
    REAL_WIDTH = 1.8  
    return round((REAL_WIDTH * FOCAL_LENGTH) / w, 2)

def estimate_speed(distance, prev_distance, time_diff):
    """Estimate speed based on distance change over time."""
    if prev_distance is None or time_diff == 0:
        return 0  
    speed = (prev_distance - distance) / time_diff * 3.6
    return round(abs(speed), 2)  


cap = cv2.VideoCapture("./videos/roads.mp4")  
if not cap.isOpened():
    raise FileNotFoundError("❌ Error: Cannot access webcam!")

print(" Webcam started... Press 'Q' to exit.")

# Enable fullscreen mode
cv2.namedWindow("Real-Time Car Risk Analysis", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Real-Time Car Risk Analysis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Webcam feed unavailable.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    
    current_time = time.time()
    time_diff = current_time - prev_time
    prev_time = current_time

    front_car_speed = 0
    distance = None

    for (x, y, w, h) in cars:
        distance = estimate_distance(w) 

    
        front_car_speed = estimate_speed(distance, prev_distance, time_diff)
        prev_distance = distance

        
        risk_features = np.array([[distance, your_car_speed, 0, front_car_speed]])
        risk_level = model.predict(risk_features)[0]

        
        if risk_level == 1 and distance < 25:
            risk_text, color = "CRITICAL!", (0, 0, 255) 
            pygame.mixer.music.load(alert_sound)
            pygame.mixer.music.play()
        elif risk_level == 1 and distance < 30:
            risk_text, color = "WARNING", (0, 140, 255) 
        else:
            risk_text, color = "SAFE", (0, 255, 0)  #


        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{risk_text} - {distance}m", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    
    cv2.putText(frame, f"Your Speed: {your_car_speed} km/h", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Front Car Speed: {front_car_speed} km/h", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    
    cv2.imshow("Real-Time Car Risk Analysis", frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting...")
        break


cap.release()
cv2.destroyAllWindows()
print("Webcam stream closed.")
