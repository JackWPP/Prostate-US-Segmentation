

import torch
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Project Setup ---
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import the best performing model
from src.models_zoo.attention_model.model import MicroSegNetAttention

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
TEST_IMAGE_DIR = os.path.join(PROJECT_ROOT, 'processed_data', 'test', 'images')
TEST_MASK_DIR = os.path.join(PROJECT_ROOT, 'processed_data', 'test', 'masks')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
OUTPUT_FILENAME = "segmentation_comparison.png"

# Colors for contours (BGR format for OpenCV)
COLOR_GROUND_TRUTH = (0, 255, 0)  # Green
COLOR_PREDICTION = (0, 0, 255)   # Red

# --- Main Visualization Logic ---
def main():
    print("--- Starting Final Result Visualization ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Model
    print("Loading MicroSegNetAttention model...")
    try:
        model = MicroSegNetAttention(num_classes=1)
        path = os.path.join(MODELS_DIR, "attention", "best_attention_model.pth")
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Fatal: Could not load the model. Error: {e}")
        return

    # 2. Select and Load Data
    test_images = sorted([os.path.basename(p) for p in glob.glob(os.path.join(TEST_IMAGE_DIR, "*.npy"))])
    if not test_images:
        print("Fatal: No test images found.")
        return
    
    # We'll use the first image as our example
    image_name = test_images[0]
    print(f"Processing image: {image_name}")
    img_path = os.path.join(TEST_IMAGE_DIR, image_name)
    mask_path = os.path.join(TEST_MASK_DIR, image_name)

    image_np = np.load(img_path).astype(np.float32)
    mask_np = np.load(mask_path).astype(np.uint8)
    if mask_np.ndim == 3:
        mask_np = np.squeeze(mask_np, axis=-1)

    # 3. Make Prediction
    print("Generating model prediction...")
    with torch.no_grad():
        # Preprocessing must match the model's training
        input_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        output = model(input_tensor)
        pred = (output > 0.5).cpu().numpy()
        pred_mask = np.squeeze(pred).astype(np.uint8)

    # 4. Create Overlay Image
    print("Creating overlay image...")
    # Convert grayscale image to 3-channel BGR to draw colored contours
    overlay_image = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Find and draw contours for Ground Truth
    contours_gt, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_image, contours_gt, -1, COLOR_GROUND_TRUTH, 2) # Thickness=2

    # Find and draw contours for Prediction
    contours_pred, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_image, contours_pred, -1, COLOR_PREDICTION, 2) # Thickness=2

    # 5. Plot and Save with Matplotlib
    print("Saving final image...")
    plt.style.use('default')
    fig, ax = plt.subplots(1, figsize=(8, 8))
    
    # OpenCV uses BGR, Matplotlib uses RGB. Convert for display.
    ax.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    
    # Create a custom legend
    legend_elements = [Line2D([0], [0], color='g', lw=2, label='Ground Truth'),
                       Line2D([0], [0], color='r', lw=2, label='Prediction (MicroSegNetAttention)')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.title('Segmentation Result Comparison', fontsize=14)
    
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"\n--- Visualization Complete ---")
    print(f"Result saved to: {save_path}")

if __name__ == "__main__":
    main()

