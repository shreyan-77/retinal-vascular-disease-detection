import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.cnn_model import SimpleCNN


# =========================
# Dataset class
# =========================
class DriveDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # image path
        img_path = os.path.join(self.img_dir, img_name)

        # correct mask mapping
        mask_name = img_name.split("_")[0] + "_manual1.gif"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # ----- Load image -----
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # green channel extraction
        green = img[:, :, 1]
        green = cv2.resize(green, (256, 256))

        # ----- Load mask -----
        mask = cv2.imread(mask_path, 0)

        # IMPORTANT: resize mask to match CNN output (64x64)
        mask = cv2.resize(mask, (64, 64))

        # normalize
        green = green / 255.0
        mask = mask / 255.0

        # convert to tensor
        green = torch.tensor(green).unsqueeze(0).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return green, mask


# =========================
# Paths (YOUR PATHS)
# =========================
img_dir = r"C:\SHREYAN\Retinalproject\DRIVE\training\images"
mask_dir = r"C:\SHREYAN\Retinalproject\DRIVE\training\1st_manual"


# =========================
# Dataset + loader
# =========================
dataset = DriveDataset(img_dir, mask_dir)
print("Dataset size:", len(dataset))

loader = DataLoader(dataset, batch_size=2, shuffle=True)


# =========================
# Model setup
# =========================
model = SimpleCNN()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# =========================
# Training loop
# =========================
epochs = 3

for epoch in range(epochs):
    for imgs, masks in loader:

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")


# =========================
# Save model
# =========================
torch.save(model.state_dict(), "model.pth")
print("✅ Model saved successfully")