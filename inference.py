#!/usr/bin/env python3
"""
Inference script for RTDETR model - run inference on new images
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import RTDETR
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_inference(model_path, input_path, output_dir, conf_threshold=0.25, save_plots=True):
    """Run inference on images or video"""
    
    # Load model
    model = RTDETR(model_path)
    logger.info(f"Model loaded from: {model_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single image or video
        results = model(str(input_path), conf=conf_threshold, save=True, project=str(output_dir))
        logger.info(f"Inference completed on: {input_path}")
        
    elif input_path.is_dir():
        # Directory of images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(image_files)} images in {input_path}")
        
        for img_file in image_files:
            results = model(str(img_file), conf=conf_threshold, save=True, project=str(output_dir))
            logger.info(f"Processed: {img_file.name}")
    
    else:
        logger.error(f"Invalid input path: {input_path}")
        return
    
    logger.info(f"Results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run RTDETR inference')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image, video, or directory')
    parser.add_argument('--output', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    try:
        run_inference(args.weights, args.input, args.output, args.conf)
        logger.info("Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
