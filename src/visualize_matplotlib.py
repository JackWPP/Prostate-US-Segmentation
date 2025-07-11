
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- Configuration ---
PROCESSED_DATA_PATH = "processed_data"
IMG_DIR = os.path.join(PROCESSED_DATA_PATH, "train", "images")
MASK_DIR = os.path.join(PROCESSED_DATA_PATH, "train", "masks")

# --- Load File Paths ---
image_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.npy")))
mask_paths = sorted(glob.glob(os.path.join(MASK_DIR, "*.npy")))

# --- Robustness Check ---
if not image_paths or not mask_paths:
    print("Error: No preprocessed .npy files found.")
    print(f"Please run 'python src/preprocess.py' first to generate the data.")
    exit()

# --- Visualization Function ---
def create_overlay(img, mask):
    """Creates a color overlay of the mask on the image."""
    img_display = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
    
    red_mask = np.zeros_like(img_bgr, dtype=np.uint8)
    red_mask[:, :, 2] = mask * 255  # Red channel for the mask
    
    return cv2.addWeighted(img_bgr, 0.7, red_mask, 0.3, 0)

# --- Set up the plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25) # Make space for the slider

# --- Initial Image ---
initial_index = 0
img_0 = np.load(image_paths[initial_index]).squeeze()
mask_0 = np.load(mask_paths[initial_index]).squeeze()
overlay_0 = create_overlay(img_0, mask_0)

im1 = ax1.imshow(img_0, cmap='gray')
ax1.set_title("Preprocessed Image")
ax1.axis('off')

im2 = ax2.imshow(cv2.cvtColor(overlay_0, cv2.COLOR_BGR2RGB))
ax2.set_title("Image with Mask Overlay")
ax2.axis('off')

fig.suptitle(f"File: {os.path.basename(image_paths[initial_index])}")

# --- Create the Slider ---
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(
    ax=ax_slider,
    label='Image Index',
    valmin=0,
    valmax=len(image_paths) - 1,
    valinit=initial_index,
    valstep=1
)

# --- Update function for the slider ---
def update(val):
    index = int(slider.val)
    
    img = np.load(image_paths[index]).squeeze()
    mask = np.load(mask_paths[index]).squeeze()
    overlay = create_overlay(img, mask)
    
    # Update image data
    im1.set_data(img)
    im2.set_data(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    
    # Update title
    fig.suptitle(f"File: {os.path.basename(image_paths[index])}")
    
    # Redraw the canvas
    fig.canvas.draw_idle()

# --- Register the update function ---
slider.on_changed(update)

# --- Show the plot ---
print("Launching Matplotlib interactive window...")
print("Close the window to exit the script.")
plt.show()
