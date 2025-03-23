import cv2
import numpy as np


left_lane_history = []
right_lane_history = []
max_history = 5  
def filter_white_yellow(image):
    """Filters white and yellow lane colors in the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

   
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=combined_mask)

def region_of_interest(image):
    """Applies a mask to keep only the region of interest (road lanes)."""
    height, width = image.shape[:2]
    mask = np.zeros_like(image)

    polygon = np.array([[
        (width * 0.1, height), 
        (width * 0.45, height * 0.6), 
        (width * 0.55, height * 0.6), 
        (width * 0.9, height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, (255, 255, 255))
    return cv2.bitwise_and(image, mask)

def adaptive_canny(image):
    """Applies adaptive Canny edge detection for dynamic thresholding."""
    median_val = np.median(image)
    lower = int(max(0, 0.66 * median_val))
    upper = int(min(255, 1.33 * median_val))
    return cv2.Canny(image, lower, upper)

def average_lines(lines):
    """Averages multiple detected lane lines for a stable display."""
    if len(lines) == 0:
        return None
    return np.mean(lines, axis=0).astype(int)

def detect_lanes(frame):
    """Detects lanes using improved filtering and line averaging."""
    global left_lane_history, right_lane_history

    roi = region_of_interest(frame)
    filtered = filter_white_yellow(roi)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = adaptive_canny(blur)

    lines = cv2.HoughLinesP(edges, 2, np.pi / 180, 50, minLineLength=80, maxLineGap=150)
    overlay = np.zeros_like(frame)

    left_lanes, right_lanes = [], []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            if -0.5 < slope < 0.5:  
                continue

            if slope < 0:  
                left_lanes.append([x1, y1, x2, y2])
            else:  
                right_lanes.append([x1, y1, x2, y2])

    
    if left_lanes:
        left_avg = average_lines(left_lanes)
        left_lane_history.append(left_avg)
        if len(left_lane_history) > max_history:
            left_lane_history.pop(0)
    if right_lanes:
        right_avg = average_lines(right_lanes)
        right_lane_history.append(right_avg)
        if len(right_lane_history) > max_history:
            right_lane_history.pop(0)

    left_final = average_lines(left_lane_history)
    right_final = average_lines(right_lane_history)

    
    if left_final is not None:
        cv2.line(overlay, (left_final[0], left_final[1]), (left_final[2], left_final[3]), (0, 255, 0), 5)
    if right_final is not None:
        cv2.line(overlay, (right_final[0], right_final[1]), (right_final[2], right_final[3]), (0, 255, 0), 5)

    
    alert_text = "Lane Detected"
    if left_final is None and right_final is None:
        alert_text = "ALERT: Both Lanes Lost! Stay Centered!"
        color = (0, 0, 255)  
    elif left_final is None:
        alert_text = "Warning: Left Lane Lost!"
        color = (0, 255, 255)  
    elif right_final is None:
        alert_text = "Warning: Right Lane Lost!"
        color = (0, 255, 255)  
    else:
        color = (0, 255, 0)  

    
    cv2.putText(frame, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)

    return cv2.addWeighted(frame, 0.8, overlay, 1, 0)

cap = cv2.VideoCapture("./videos/lane_change.mp4")


cv2.namedWindow("Lane Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Lane Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    lane_frame = detect_lanes(frame)
    cv2.imshow("Lane Detection", lane_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
