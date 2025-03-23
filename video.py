import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

# Load YOLOv8 Model
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("Model loaded successfully!")

def enhance_frame(frame):
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def process_video(input_video_path, output_video_path):
    print(f"Processing video: {input_video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_width}x{frame_height} at {fps} fps, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file: {output_video_path}")
        cap.release()
        return
    
    frame_count = 0
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Enhance frame
            enhanced_frame = enhance_frame(frame)
            
            # Run YOLO detection with increased confidence threshold
            results = model(enhanced_frame, conf=0.4)  # Increased confidence threshold
            
            # Get detection results
            result = results[0]
            
            # Create a copy of the frame for annotations
            annotated_frame = enhanced_frame.copy()
            
            # Count vehicles and draw custom boxes
            vehicle_counts = {vehicle: 0 for vehicle in vehicle_classes}
            
            for box in result.boxes:
                cls = result.names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                if cls in vehicle_classes and conf > 0.4:  # Confidence threshold
                    vehicle_counts[cls] += 1
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw box with custom style
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label with confidence
                    label = f"{cls} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add vehicle count text with background
            y_position = 30
            for vehicle, count in vehicle_counts.items():
                if count > 0:  # Only show if vehicles are detected
                    text = f"{vehicle.capitalize()}: {count}"
                    # Add background rectangle
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    cv2.rectangle(annotated_frame, (10, y_position-text_height-5),
                                (10+text_width, y_position+5), (0, 0, 0), -1)
                    # Add text
                    cv2.putText(annotated_frame, text, (10, y_position),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    y_position += 40
            
            # Add progress with background
            progress = (frame_count / total_frames) * 100
            progress_text = f"Progress: {progress:.1f}%"
            (text_width, text_height), _ = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(annotated_frame, (10, frame_height-50),
                         (10+text_width, frame_height-10), (0, 0, 0), -1)
            cv2.putText(annotated_frame, progress_text, (10, frame_height-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame
            out.write(annotated_frame)
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processing: {progress:.1f}% complete")
            
            frame_count += 1
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        print("Cleaning up...")
        cap.release()
        out.release()
        print(f"Processing complete. Output saved to: {output_video_path}")
        print(f"You can find the processed video at: {os.path.abspath(output_video_path)}")

def main():
    # Use the specific video path
    video_path = "C:/Users/rohit/Downloads/Vehicle detection in FOG conditions for Autonomous cars - YouTube - Google Chrome 2025-03-22 19-56-25.webm"
    
    # Process the video
    process_video(video_path, "output.mp4")
if __name__ == "__main__":
    main()
