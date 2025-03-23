import cv2
import numpy as np
import time
import os
import platform  


if platform.system() == "Windows":
    import winsound

CASCADE_PATH = "F:/CodeClash2025/haarcascade_car.xml"  
if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(f"Error: Missing '{CASCADE_PATH}'. Download it manually.")

car_cascade = cv2.CascadeClassifier(CASCADE_PATH)


cap = cv2.VideoCapture("./videos/lane_change.mp4")  
if not cap.isOpened():
    raise FileNotFoundError(" Error: Cannot access video file!")


cv2.namedWindow(" Lane Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(" Lane Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


prev_x = None
lane_change_threshold = 50  

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Video feed unavailable.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    lane_change_detected = False

  
    height, width, _ = frame.shape
    left_lane_x = width // 3
    right_lane_x = 2 * (width // 3)

    cv2.line(frame, (left_lane_x, 0), (left_lane_x, height), (255, 0, 0), 3)  
    cv2.line(frame, (right_lane_x, 0), (right_lane_x, height), (255, 0, 0), 3)  

    for (x, y, w, h) in cars:
        
        car_center_x = x + w // 2

        
        if prev_x is not None and abs(car_center_x - prev_x) > lane_change_threshold:
            lane_change_detected = True

          
            if platform.system() == "Windows":
                winsound.Beep(1000, 300)  
            else:
                os.system('play -nq -t alsa synth 0.2 sine 440')  
        prev_x = car_center_x  

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Car Detected", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if lane_change_detected:
        cv2.putText(frame, " LANE CHANGE DETECTED!", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
   
    cv2.imshow("Lane Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print(" Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
print("!! Video stream closed.")
