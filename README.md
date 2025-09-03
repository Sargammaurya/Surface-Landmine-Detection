# RTDETR Starfish and Butterfly Detection

This project trains and tests an RTDETR (Real-Time Detection Transformer) model for detecting starfish and butterflies using Ultralytics.

## Dataset Structure
- Classes: 0=starfish, 1=butterfly
- Format: YOLO format with bounding boxes
- Data location: data

## Files
- `train_rtdetr.py`: Training script
- `test_rtdetr.py`: Testing and evaluation script
- `dataset.yaml`: Dataset configuration
- `config.yaml`: Training configuration
- `requirements.txt`: Dependencies

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_rtdetr.py
```

3. Test the model:
```bash
python test_rtdetr.py --weights runs/detect/train/weights/best.pt
```

## Results
Training results will be saved in `runs/detect/train/`
Test results will be saved in `runs/detect/test/`
