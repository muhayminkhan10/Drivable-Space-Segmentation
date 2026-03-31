import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import UNet
from dataset import RoadDataset


# -------------------------------
# PATHS
# -------------------------------
image_dir = "images"
mask_dir = "masks"


# -------------------------------
# DATASET + DATALOADER
# -------------------------------
dataset = RoadDataset(image_dir, mask_dir)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print("Dataset loaded, total samples:", len(dataset))


# -------------------------------
# DEVICE
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# -------------------------------
# MODEL
# -------------------------------
model = UNet().to(device)


# -------------------------------
# LOSS FUNCTIONS
# -------------------------------
bce = nn.BCEWithLogitsLoss()

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-5

    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def loss_fn(pred, target):
    return bce(pred, target) + dice_loss(pred, target)


# -------------------------------
# OPTIMIZER
# -------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# -------------------------------
# TRAINING LOOP
# -------------------------------
epochs = 5

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    print(f"\nEpoch {epoch+1}/{epochs}")

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print("Average Loss:", avg_loss)


# -------------------------------
# SAVE MODEL
# -------------------------------
torch.save(model.state_dict(), "model.pth")
print("\nModel saved!")


# -------------------------------
# 🔥 PREDICTION VISUALIZATION
# -------------------------------
import os

os.makedirs("outputs", exist_ok=True)

model.eval()

for i in range(5):
    image, mask = dataset[i]

    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(device))
        pred = torch.sigmoid(pred)
        pred = pred.squeeze().cpu().numpy()
        

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image.permute(1, 2, 0))

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(mask.squeeze(), cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(pred, cmap="gray")

    plt.savefig(f"outputs/output_{i}.png")
    plt.close()