# Real-Time Drivable Space Segmentation

## Problem
Identify drivable areas from road images using pixel-wise segmentation.

## Approach
- Lightweight CNN (encoder-decoder style)
- Trained from scratch (no pretrained models)
- BCE + Dice Loss

## Pipeline
Input Image → Model → Drivable Area Mask

## Results
- Model shows decreasing loss
- Generates probability maps of drivable regions

## Limitations
- Noisy dataset masks
- Limited training time
- Lightweight model

## How to Run

pip install -r requirements.txt  
python train.py