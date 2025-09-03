#!/usr/bin/env python3
"""
RTDETR Training Script for Starfish and Butterfly Detection
Author: Generated for starfish/butterfly object detection
"""

import os
import yaml
import torch
from ultralytics import RTDETR
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_directories():
    """Create necessary directories"""
    dirs = ['runs', 'runs/detect', 'runs/detect/train']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.info("Directories created successfully")

def train_rtdetr(config):
    """Train RTDETR model"""
    logger.info("Starting RTDETR training...")
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Initialize model
    if config['model'].endswith('.pt'):
        # Use pre-trained model
        model = RTDETR(config['model'])
        logger.info(f"Loaded pre-trained model: {config['model']}")
    else:
        # Initialize model architecture
        model = RTDETR(config['model'])
        logger.info(f"Initialized model architecture: {config['model']}")
    
    # Train the model
    results = model.train(
        data=config['data'],
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch'],
        workers=config['workers'],
        device=device,
        project=config['project'],
        name=config['name'],
        optimizer=config['optimizer'],
        lr0=config['lr0'],
        lrf=config['lrf'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        warmup_epochs=config['warmup_epochs'],
        warmup_momentum=config['warmup_momentum'],
        warmup_bias_lr=config['warmup_bias_lr'],
        hsv_h=config['hsv_h'],
        hsv_s=config['hsv_s'],
        hsv_v=config['hsv_v'],
        degrees=config['degrees'],
        translate=config['translate'],
        scale=config['scale'],
        shear=config['shear'],
        perspective=config['perspective'],
        flipud=config['flipud'],
        fliplr=config['fliplr'],
        mosaic=config['mosaic'],
        mixup=config['mixup'],
        copy_paste=config['copy_paste'],
        val=config['val'],
        save_period=config['save_period'],
        patience=config['patience'],
        save=config['save'],
        plots=config['plots']
    )
    
    logger.info("Training completed successfully!")
    return results

def main():
    """Main training function"""
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Setup directories
        setup_directories()
        
        # Train model
        results = train_rtdetr(config)
        
        # Log training summary
        logger.info("="*50)
        logger.info("TRAINING SUMMARY")
        logger.info("="*50)
        logger.info(f"Model saved to: {config['project']}/{config['name']}/weights/")
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
