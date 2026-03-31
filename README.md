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

## Sample Outputs
<img width="1200" height="400" alt="output_0" src="https://github.com/user-attachments/assets/0d43931f-106b-4256-bd86-28da1a570ed5" />
<img width="1200" height="400" alt="output_4" src="https://github.com/user-attachments/assets/e5897966-fc40-48a8-a3a6-71dc69a48644" />
