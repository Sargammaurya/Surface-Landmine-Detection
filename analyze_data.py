#!/usr/bin/env python3
"""
Data preprocessing and analysis script for RTDETR training
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    def __init__(self, data_path='data'):
        self.data_path = Path(data_path)
        self.images_path = self.data_path / 'images'
        self.labels_path = self.data_path / 'labels'
        self.class_names = {0: 'starfish', 1: 'butterfly'}
        
    def analyze_dataset_structure(self):
        """Analyze the dataset structure and provide statistics"""
        logger.info("Analyzing dataset structure...")
        
        # Count images and labels in each folder
        image_folders = [f for f in self.images_path.iterdir() if f.is_dir()]
        
        stats = {
            'folders': [],
            'image_counts': [],
            'label_counts': [],
            'total_images': 0,
            'total_labels': 0
        }
        
        for folder in image_folders:
            folder_name = folder.name
            
            # Count images
            image_files = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
            image_count = len(image_files)
            
            # Count labels
            label_folder = self.labels_path / folder_name
            if label_folder.exists():
                label_files = list(label_folder.glob('*.txt'))
                label_count = len(label_files)
            else:
                label_count = 0
            
            stats['folders'].append(folder_name)
            stats['image_counts'].append(image_count)
            stats['label_counts'].append(label_count)
            stats['total_images'] += image_count
            stats['total_labels'] += label_count
            
            logger.info(f"{folder_name}: {image_count} images, {label_count} labels")
        
        return stats
    
    def analyze_annotations(self, sample_size=1000):
        """Analyze annotation statistics"""
        logger.info("Analyzing annotations...")
        
        class_counts = Counter()
        bbox_sizes = []
        bbox_ratios = []
        objects_per_image = []
        
        # Sample annotations from all folders
        all_label_files = []
        for folder in self.labels_path.iterdir():
            if folder.is_dir():
                label_files = list(folder.glob('*.txt'))
                all_label_files.extend(label_files)
        
        # Limit sample size
        if len(all_label_files) > sample_size:
            import random
            all_label_files = random.sample(all_label_files, sample_size)
        
        for label_file in all_label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                image_objects = 0
                for line in lines:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            class_counts[class_id] += 1
                            bbox_sizes.append(width * height)  # Normalized area
                            bbox_ratios.append(width / height if height > 0 else 0)
                            image_objects += 1
                
                objects_per_image.append(image_objects)
                
            except Exception as e:
                logger.warning(f"Error processing {label_file}: {e}")
        
        return {
            'class_counts': dict(class_counts),
            'bbox_sizes': bbox_sizes,
            'bbox_ratios': bbox_ratios,
            'objects_per_image': objects_per_image
        }
    
    def analyze_images(self, sample_size=500):
        """Analyze image properties"""
        logger.info("Analyzing image properties...")
        
        image_sizes = []
        aspect_ratios = []
        
        # Sample images from all folders
        all_image_files = []
        for folder in self.images_path.iterdir():
            if folder.is_dir():
                image_files = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
                all_image_files.extend(image_files)
        
        # Limit sample size
        if len(all_image_files) > sample_size:
            import random
            all_image_files = random.sample(all_image_files, sample_size)
        
        for image_file in all_image_files:
            try:
                img = cv2.imread(str(image_file))
                if img is not None:
                    h, w = img.shape[:2]
                    image_sizes.append((w, h))
                    aspect_ratios.append(w / h)
            except Exception as e:
                logger.warning(f"Error processing {image_file}: {e}")
        
        return {
            'image_sizes': image_sizes,
            'aspect_ratios': aspect_ratios
        }
    
    def create_analysis_plots(self, dataset_stats, annotation_stats, image_stats):
        """Create visualization plots for dataset analysis"""
        logger.info("Creating analysis plots...")
        
        # Create output directory
        output_dir = Path('dataset_analysis')
        output_dir.mkdir(exist_ok=True)
        
        # 1. Dataset structure plot
        plt.figure(figsize=(12, 6))
        x = range(len(dataset_stats['folders']))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], dataset_stats['image_counts'], 
                width, label='Images', alpha=0.8)
        plt.bar([i + width/2 for i in x], dataset_stats['label_counts'], 
                width, label='Labels', alpha=0.8)
        
        plt.xlabel('Folders')
        plt.ylabel('Count')
        plt.title('Dataset Structure - Images and Labels per Folder')
        plt.xticks(x, dataset_stats['folders'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'dataset_structure.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Class distribution plot
        if annotation_stats['class_counts']:
            plt.figure(figsize=(8, 6))
            classes = [self.class_names.get(k, f'Class {k}') for k in annotation_stats['class_counts'].keys()]
            counts = list(annotation_stats['class_counts'].values())
            
            colors = ['red', 'green'][:len(classes)]
            bars = plt.bar(classes, counts, color=colors, alpha=0.8)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        str(count), ha='center', va='bottom')
            
            plt.xlabel('Classes')
            plt.ylabel('Number of Annotations')
            plt.title('Class Distribution in Dataset')
            plt.tight_layout()
            plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Bounding box analysis
        if annotation_stats['bbox_sizes']:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Bbox sizes histogram
            ax1.hist(annotation_stats['bbox_sizes'], bins=50, alpha=0.7, color='blue')
            ax1.set_xlabel('Normalized Bbox Area')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Bounding Box Size Distribution')
            
            # Bbox ratios histogram
            ax2.hist(annotation_stats['bbox_ratios'], bins=50, alpha=0.7, color='green')
            ax2.set_xlabel('Width/Height Ratio')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Bounding Box Aspect Ratio Distribution')
            
            # Objects per image
            ax3.hist(annotation_stats['objects_per_image'], bins=20, alpha=0.7, color='orange')
            ax3.set_xlabel('Objects per Image')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Objects per Image Distribution')
            
            # Image aspect ratios
            if image_stats['aspect_ratios']:
                ax4.hist(image_stats['aspect_ratios'], bins=50, alpha=0.7, color='purple')
                ax4.set_xlabel('Image Width/Height Ratio')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Image Aspect Ratio Distribution')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'bbox_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Analysis plots saved to: {output_dir}")
    
    def generate_analysis_report(self, dataset_stats, annotation_stats, image_stats):
        """Generate a comprehensive analysis report"""
        logger.info("Generating analysis report...")
        
        output_dir = Path('dataset_analysis')
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / 'dataset_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("DATASET ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset structure
            f.write("DATASET STRUCTURE:\n")
            f.write(f"Total folders: {len(dataset_stats['folders'])}\n")
            f.write(f"Total images: {dataset_stats['total_images']}\n")
            f.write(f"Total labels: {dataset_stats['total_labels']}\n")
            f.write(f"Folders: {', '.join(dataset_stats['folders'])}\n\n")
            
            # Class distribution
            if annotation_stats['class_counts']:
                f.write("CLASS DISTRIBUTION:\n")
                total_annotations = sum(annotation_stats['class_counts'].values())
                for class_id, count in annotation_stats['class_counts'].items():
                    class_name = self.class_names.get(class_id, f'Class {class_id}')
                    percentage = (count / total_annotations) * 100
                    f.write(f"{class_name}: {count} annotations ({percentage:.1f}%)\n")
                f.write(f"Total annotations: {total_annotations}\n\n")
            
            # Annotation statistics
            if annotation_stats['bbox_sizes']:
                f.write("ANNOTATION STATISTICS:\n")
                f.write(f"Average bbox size: {np.mean(annotation_stats['bbox_sizes']):.4f}\n")
                f.write(f"Median bbox size: {np.median(annotation_stats['bbox_sizes']):.4f}\n")
                f.write(f"Average bbox ratio: {np.mean(annotation_stats['bbox_ratios']):.2f}\n")
                f.write(f"Average objects per image: {np.mean(annotation_stats['objects_per_image']):.2f}\n\n")
            
            # Image statistics
            if image_stats['image_sizes']:
                widths = [size[0] for size in image_stats['image_sizes']]
                heights = [size[1] for size in image_stats['image_sizes']]
                
                f.write("IMAGE STATISTICS:\n")
                f.write(f"Average image width: {np.mean(widths):.0f}px\n")
                f.write(f"Average image height: {np.mean(heights):.0f}px\n")
                f.write(f"Image size range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}\n")
                f.write(f"Average aspect ratio: {np.mean(image_stats['aspect_ratios']):.2f}\n")
        
        logger.info(f"Analysis report saved to: {report_path}")

def main():
    """Main analysis function"""
    try:
        analyzer = DatasetAnalyzer()
        
        # Analyze dataset
        dataset_stats = analyzer.analyze_dataset_structure()
        annotation_stats = analyzer.analyze_annotations(sample_size=2000)
        image_stats = analyzer.analyze_images(sample_size=1000)
        
        # Create visualizations
        analyzer.create_analysis_plots(dataset_stats, annotation_stats, image_stats)
        
        # Generate report
        analyzer.generate_analysis_report(dataset_stats, annotation_stats, image_stats)
        
        logger.info("Dataset analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
