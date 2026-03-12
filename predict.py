import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.cnn_model import SimpleCNN

# =========================
# ⭐ CONFIGURATION & LOAD
# =========================
img_path = r"C:\SHREYAN\Retinalproject\DRIVE\test\images\01_test.tif"

model = SimpleCNN()
# Using map_location='cpu' ensures it runs smoothly on any machine
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# =========================
# ⭐ PREPROCESSING
# =========================
img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Image not found at path: {img_path}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

green = img_rgb[:, :, 1]
green_resized = cv2.resize(green, (256, 256))
green_normalized = green_resized / 255.0

tensor = torch.tensor(green_normalized).unsqueeze(0).unsqueeze(0).float()

# =========================
# ⭐ PREDICTION & FIX
# =========================
with torch.no_grad():
    pred = model(tensor)

pred_np = pred.squeeze().numpy()

# -----------------------------------------------------
# APPLIED FIX: Min-Max Normalization (Contrast Stretching)
# Forces the model's confidence to scale from 0.0 to 1.0
# -----------------------------------------------------
if pred_np.max() > pred_np.min(): 
    pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min())

# =========================
# ⭐ METRICS CALCULATION
# =========================
vessel_score = float(pred_np.mean())
binary_mask = pred_np > 0.5
vessel_density = float(binary_mask.sum() / binary_mask.size)
vessel_pixels = int(binary_mask.sum())

print("\n" + "="*30)
print("📊 METRICS REPORT")
print("="*30)
print(f"Mean Vessel Probability : {vessel_score:.3f}")
print(f"Vessel Density          : {vessel_density:.2%}")
print(f"Vessel Pixel Count      : {vessel_pixels:,}")
print("="*30 + "\n")

# =========================
# ⭐ VISUALIZATION
# =========================
h, w, _ = img_rgb.shape
pred_up = cv2.resize(pred_np, (w, h))

# Professional Cyan Overlay (matching Streamlit app)
mask = np.zeros_like(img_rgb)
mask[:, :, 1] = (pred_up * 255).astype(np.uint8) # Green
mask[:, :, 2] = (pred_up * 255).astype(np.uint8) # Blue (Creates Cyan)
overlay = cv2.addWeighted(img_rgb, 0.6, mask, 0.7, 0)

# Matplotlib Plot
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Input Scan", fontweight='bold')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f"AI Segmentation (Density: {vessel_density:.1%})", fontweight='bold')
plt.imshow(pred_up, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Pathological Overlay", fontweight='bold')
plt.imshow(overlay)
plt.axis('off')

plt.tight_layout()
plt.show()