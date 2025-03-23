import cv2
import numpy as np
import time
import os
import pyttsx3  


engine = pyttsx3.init()

# Load OpenCV Haar cascade for car detection
CASCADE_PATH = "F:/CodeClash2025/haarcascade_car.xml" 
if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(f" Error: Missing '{CASCADE_PATH}'. Download it manually.")

car_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def estimate_distance(w):
    """Estimate distance based on detected object width."""
    FOCAL_LENGTH = 850  
    REAL_WIDTH = 1.8  
    return round((REAL_WIDTH * FOCAL_LENGTH) / w, 2)


prev_positions = {}
fps = 30  
METER_PER_PIXEL = 0.05  
frame_time = 1 / fps  

def estimate_speed(prev_x, new_x):
    """Calculate speed based on pixel movement per frame."""
    if prev_x is None:
        return 0
    speed = abs(new_x - prev_x) * METER_PER_PIXEL * fps * 3.6  
    return round(speed, 2)


cap = cv2.VideoCapture("./videos/roads.mp4")  

if not cap.isOpened():
    raise FileNotFoundError(" Error: Cannot access webcam or video!")

print(" Video started... Press 'Q' to exit.")


cv2.namedWindow(" Real-Time Car Risk Analysis", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(" Real-Time Car Risk Analysis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


my_car_speed = 50   

# Real-time processing loop
while True:
    start_time = time.time()  

    ret, frame = cap.read()
    if not ret:
        print("Video feed unavailable.")
        break

    
    frame = cv2.resize(frame, (1920, 1080))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    front_car_speed = 0
    alert_played = False

    for i, (x, y, w, h) in enumerate(cars):
        distance = estimate_distance(w) 
        car_id = i 

        # Calculate front car speed
        prev_x = prev_positions.get(car_id, None)
        front_car_speed = estimate_speed(prev_x, x)
        prev_positions[car_id] = x  

       
        if distance < 8:
            risk_text, color = "CRITICAL!", (0, 0, 255)  
            if not alert_played:
                engine.say("Danger! Collision risk detected!")
                engine.runAndWait()
                alert_played = True
        elif distance < 15:
            risk_text, color = "WARNING", (0, 140, 255)  
            if not alert_played:
                engine.say("Warning! Maintain distance.")
                engine.runAndWait()
                alert_played = True
        else:
            risk_text, color = "SAFE", (0, 255, 0)  

       
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{risk_text} - {distance}m", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    
    cv2.putText(frame, f"My Speed: {my_car_speed} km/h", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Front Car Speed: {front_car_speed} km/h", (30, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    
    cv2.imshow(" Real-Time Car Risk Analysis", frame)

   
    elapsed_time = time.time() - start_time
    sleep_time = max(0, frame_time - elapsed_time)
    time.sleep(sleep_time)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print(" Exiting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(" Video stream closed.")
