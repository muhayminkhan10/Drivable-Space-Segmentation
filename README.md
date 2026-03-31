# Real-Time Drivable Space Segmentation

## Project Overview
This project focuses on identifying drivable (free) space from road images using semantic segmentation.  
The system processes camera input and predicts drivable vs non-drivable regions.

---

## Model Architecture
- Lightweight encoder-decoder CNN (U-Net inspired)
- Performs pixel-wise segmentation
- Designed for real-time inference

---

## Dataset Used
- nuScenes dataset (urban driving scenes)
- Contains road images and segmentation masks
- Dataset used in a simplified format for this project

---

## Setup & Installation
pip install -r requirements.txt 

## Sample Outputs
<img width="1200" height="400" alt="output_0" src="https://github.com/user-attachments/assets/0d43931f-106b-4256-bd86-28da1a570ed5" />
<img width="1200" height="400" alt="output_4" src="https://github.com/user-attachments/assets/e5897966-fc40-48a8-a3a6-71dc69a48644" />
