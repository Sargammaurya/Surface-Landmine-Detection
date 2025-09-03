#!/usr/bin/env python3
"""
RTDETR Testing and Evaluation Script for Starfish and Butterfly Detection
Author: Generated for starfish/butterfly object detection
"""

import os
import argparse
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import RTDETR
import logging
from collections import defaultdict
import pandas as pd
from PIL import Image

# --- HOTFIX for Ultralytics ratio_pad float vs tuple ---
from ultralytics.utils import ops as _ops

_orig_scale_boxes = _ops.scale_boxes

def _safe_scale_boxes(img1_shape, boxes, img0_shape=None, ratio_pad=None, padding=True):
    # Accept scalar or partially-specified ratio_pad and normalize to the new structure
    if isinstance(ratio_pad, (float, int)):
        ratio_pad = ((float(ratio_pad), float(ratio_pad)), (0.0, 0.0))
    elif isinstance(ratio_pad, (list, tuple)):
        # cases like (gain,) or (gain, (padw, padh))
        if len(ratio_pad) == 1 and isinstance(ratio_pad[0], (float, int)):
            ratio_pad = ((float(ratio_pad[0]), float(ratio_pad[0])), (0.0, 0.0))
        elif len(ratio_pad) == 2 and isinstance(ratio_pad[0], (float, int)):
            pad = ratio_pad[1] if isinstance(ratio_pad[1], (list, tuple)) and len(ratio_pad[1]) == 2 else (0.0, 0.0)
            ratio_pad = ((float(ratio_pad[0]), float(ratio_pad[0])), (float(pad[0]), float(pad[1])))
    return _orig_scale_boxes(img1_shape, boxes, img0_shape, ratio_pad, padding)

_ops.scale_boxes = _safe_scale_boxes

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RTDETRTester:
    def __init__(self, weights_path, data_config='dataset.yaml'):
        """Initialize RTDETR tester"""
        self.weights_path = weights_path
        self.data_config = data_config
        self.model = None
        self.class_names = {0: 'starfish', 1: 'butterfly'}
        self.results_dir = Path('runs/detect/test')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(self):
        """Load trained model"""
        try:
            self.model = RTDETR(self.weights_path)
            logger.info(f"Model loaded successfully from: {self.weights_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
            
    def load_dataset_config(self):
        """Load dataset configuration"""
        with open(self.data_config, 'r') as file:
            config = yaml.safe_load(file)
        return config
        
    def run_validation(self):
        """Run validation on the validation set"""
        logger.info("Running validation...")
        config = self.load_dataset_config()
        
        # Run validation
        results = self.model.val(
            data=self.data_config,
            imgsz=640,
            batch=16,
            save_json=True,
            save_hybrid=True,
            conf=0.001,
            iou=0.6,
            max_det=300,
            project='runs/detect',
            name='test'
        )
        
        logger.info("Validation completed")
        return results
        
    def test_single_image(self, image_path, conf_threshold=0.25, save_result=True):
        """Test on a single image"""
        logger.info(f"Testing on image: {image_path}")
        
        # Run inference
        results = self.model(image_path, conf=conf_threshold, save=save_result)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    detections.append({
                        'class': self.class_names[cls],
                        'confidence': float(conf),
                        'bbox': box.tolist()
                    })
        
        return detections, results
        
    def test_image_folder(self, folder_path, conf_threshold=0.25):
        """Test on all images in a folder"""
        logger.info(f"Testing on images in folder: {folder_path}")
        
        folder = Path(folder_path)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        all_detections = {}
        total_detections = defaultdict(int)
        
        for img_path in folder.glob('*'):
            if img_path.suffix.lower() in image_extensions:
                try:
                    detections, _ = self.test_single_image(str(img_path), conf_threshold, save_result=False)
                    all_detections[str(img_path)] = detections
                    
                    # Count detections per class
                    for det in detections:
                        total_detections[det['class']] += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {str(e)}")
                    
        return all_detections, total_detections
        
    def visualize_detections(self, image_path, detections, save_path=None):
        """Visualize detections on an image"""
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Define colors for each class
        colors = {
            'starfish': (255, 0, 0),    # Red
            'butterfly': (0, 255, 0)    # Green
        }
        
        # Draw detections
        for det in detections:
            bbox = det['bbox']
            class_name = det['class']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(image, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         colors[class_name], 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image,
                         (int(bbox[0]), int(bbox[1] - label_size[1] - 10)),
                         (int(bbox[0] + label_size[0]), int(bbox[1])),
                         colors[class_name], -1)
            cv2.putText(image, label,
                       (int(bbox[0]), int(bbox[1] - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save or display
        if save_path:
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Detections: {len(detections)} objects')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Visualization saved to: {save_path}")
        
        return image
        
    def generate_test_report(self, validation_results, folder_results=None):
        """Generate comprehensive test report"""
        logger.info("Generating test report...")
        
        report_path = self.results_dir / 'test_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("RTDETR STARFISH & BUTTERFLY DETECTION - TEST REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Model information
            f.write("MODEL INFORMATION:\n")
            f.write(f"Weights: {self.weights_path}\n")
            f.write(f"Classes: {list(self.class_names.values())}\n\n")
            
            # Validation results
            if validation_results:
                f.write("VALIDATION RESULTS:\n")
                # Extract key metrics from validation results
                try:
                    metrics = validation_results.results_dict
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"{key}: {value:.4f}\n")
                except:
                    f.write("Validation metrics extraction failed\n")
                f.write("\n")
            
            # Folder test results
            if folder_results:
                f.write("FOLDER TEST RESULTS:\n")
                total_images = len(folder_results[0])
                f.write(f"Total images tested: {total_images}\n")
                f.write("Detections per class:\n")
                for class_name, count in folder_results[1].items():
                    f.write(f"  {class_name}: {count} detections\n")
                f.write("\n")
        
        logger.info(f"Test report saved to: {report_path}")
        
    def create_detection_summary_plots(self, folder_results):
        """Create summary plots for detection results"""
        if not folder_results or not folder_results[1]:
            logger.warning("No detection data available for plotting")
            return
            
        # Detection count per class
        classes = list(folder_results[1].keys())
        counts = list(folder_results[1].values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, counts, color=['red', 'green'])
        plt.title('Detection Count per Class')
        plt.xlabel('Class')
        plt.ylabel('Number of Detections')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = self.results_dir / 'detection_summary.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Detection summary plot saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Test RTDETR model')
    parser.add_argument('--weights', type=str, required=True, 
                       help='Path to trained model weights')
    parser.add_argument('--data', type=str, default='dataset.yaml',
                       help='Path to dataset configuration')
    parser.add_argument('--image', type=str, 
                       help='Path to single image for testing')
    parser.add_argument('--folder', type=str,
                       help='Path to folder containing test images')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--val', action='store_true',
                       help='Run validation on validation set')
    
    args = parser.parse_args()
    
    try:
        # Initialize tester
        tester = RTDETRTester(args.weights, args.data)
        tester.load_model()
        
        validation_results = None
        folder_results = None
        
        # Run validation if requested
        if args.val:
            validation_results = tester.run_validation()
        
        # Test single image
        if args.image:
            detections, results = tester.test_single_image(args.image, args.conf)
            logger.info(f"Found {len(detections)} detections in {args.image}")
            
            # Visualize results
            vis_path = tester.results_dir / f"detection_{Path(args.image).stem}.png"
            tester.visualize_detections(args.image, detections, vis_path)
        
        # Test folder of images
        if args.folder:
            folder_results = tester.test_image_folder(args.folder, args.conf)
            logger.info(f"Tested {len(folder_results[0])} images")
            logger.info(f"Total detections: {dict(folder_results[1])}")
            
            # Create summary plots
            tester.create_detection_summary_plots(folder_results)
        
        # Generate comprehensive report
        tester.generate_test_report(validation_results, folder_results)
        
        logger.info("Testing completed successfully!")
        logger.info(f"Results saved in: {tester.results_dir}")
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
