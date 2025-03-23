import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import time
from collections import defaultdict
import json

class PerformanceEvaluator:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

    def evaluate_single_image(self, image_path, ground_truth=None):
        """Evaluate model performance on a single image"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error loading image: {image_path}")
            return None
            
        # Time the inference
        start_time = time.time()
        results = self.model(image, conf=self.conf_threshold)
        inference_time = time.time() - start_time
        
        # Get detections
        result = results[0]
        detections = []
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            detections.append({
                'class': result.names[cls],
                'confidence': conf,
                'bbox': xyxy.tolist()
            })
            
        return {
            'inference_time': inference_time,
            'detections': detections,
            'fps': 1.0 / inference_time
        }

    def evaluate_dataset(self, normal_dir, foggy_dir):
        """Compare performance between normal and foggy conditions"""
        conditions = {
            'normal': Path(normal_dir ),
            'foggy': Path(foggy_dir)
        }
        
        results = {}
        for condition, directory in conditions.items():
            print(f"\nEvaluating {condition} conditions...")
            
            total_time = 0
            total_detections = 0
            fps_list = []
            confidence_scores = []
            
            image_files = list(directory.glob('*.jpg')) + list(directory.glob('*.png'))
            
            for img_path in image_files:
                eval_result = self.evaluate_single_image(img_path)
                if eval_result is None:
                    continue
                    
                total_time += eval_result['inference_time']
                total_detections += len(eval_result['detections'])
                fps_list.append(eval_result['fps'])
                
                for detection in eval_result['detections']:
                    confidence_scores.append(detection['confidence'])
            
            if len(image_files) > 0:
                avg_fps = sum(fps_list) / len(fps_list)
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                
                results[condition] = {
                    'avg_fps': avg_fps,
                    'avg_confidence': avg_confidence,
                    'total_detections': total_detections,
                    'images_processed': len(image_files)
                }
        
        return results

def main():
    # Initialize evaluator
    evaluator = PerformanceEvaluator()
    
    # Define paths for normal and foggy conditions
    normal_dir = "dataset/normal"
    foggy_dir = "dataset/foggy"
    
    # Run evaluation
    results = evaluator.evaluate_dataset(normal_dir, foggy_dir)
    
    # Print results
    print("\n=== Performance Evaluation Results ===")
    for condition, metrics in results.items():
        print(f"\n{condition.upper()} Conditions:")
        print(f"Average FPS: {metrics['avg_fps']:.2f}")
        print(f"Average Confidence: {metrics['avg_confidence']:.2f}")
        print(f"Total Detections: {metrics['total_detections']}")
        print(f"Images Processed: {metrics['images_processed']}")
    
    # Save results to JSON
    with open('performance_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nResults have been saved to performance_results.json")

if __name__ == "__main__":
    main()
