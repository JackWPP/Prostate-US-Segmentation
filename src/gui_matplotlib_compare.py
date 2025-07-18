import torch
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Project Setup ---
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import models
from src.models_zoo.base_model.model import MicroSegNet
from src.models_zoo.attention_model.model import MicroSegNetAttention
from src.models_zoo.unet.model import UNet

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
TEST_IMAGE_DIR = os.path.join(PROJECT_ROOT, 'processed_data', 'test', 'images')
TEST_MASK_DIR = os.path.join(PROJECT_ROOT, 'processed_data', 'test', 'masks')

# --- Model Loading ---
def load_models():
    """Load all trained models into a dictionary."""
    models = {}
    # Define models to load with their specific paths and classes
    models_to_load = [
        ("MicroSegNet", MicroSegNet(num_classes=1), os.path.join(MODELS_DIR, "best_microsegnet_model.pth")),
        ("MicroSegNetAttention", MicroSegNetAttention(num_classes=1), os.path.join(MODELS_DIR, "attention", "best_attention_model.pth")),
        ("UNet", UNet(n_channels=1, n_classes=1), os.path.join(MODELS_DIR, "unet", "best_unet_model.pth"))
    ]

    for name, model_instance, path in models_to_load:
        try:
            model_instance.load_state_dict(torch.load(path, map_location=DEVICE))
            model_instance.to(DEVICE)
            model_instance.eval()
            models[name] = model_instance
            print(f"Successfully loaded {name}.")
        except Exception as e:
            print(f"Could not load {name}: {e}")
    return models

# --- GUI Class ---
class ComparisonGUI:
    def __init__(self, models):
        self.models = models
        self.test_images = sorted([os.path.basename(p) for p in glob.glob(os.path.join(TEST_IMAGE_DIR, "*.npy"))])
        self.current_index = 0

        # Define preprocessing pipelines
        self.transform_unet = A.Compose([ToTensorV2()])

        # Create the plot
        self.fig, self.axes = plt.subplots(1, len(models) + 2, figsize=(20, 5))
        self.fig.suptitle('Model Comparison Tool', fontsize=16)
        
        self.update_plot()

    def predict(self, image_np):
        """Run predictions with all loaded models."""
        predictions = {}
        with torch.no_grad():
            for name, model in self.models.items():
                if "MicroSegNet" in name:
                    input_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                elif "UNet" in name:
                    augmented = self.transform_unet(image=image_np)
                    input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
                
                output = model(input_tensor)
                
                if name == 'UNet':
                    pred = (torch.sigmoid(output) > 0.5).cpu().numpy()
                else:
                    pred = (output > 0.5).cpu().numpy()
                
                predictions[name] = np.squeeze(pred).astype(np.uint8)
        return predictions

    def update_plot(self):
        """Clear axes and draw new images and predictions."""
        for ax in self.axes:
            ax.clear()

        image_name = self.test_images[self.current_index]
        img_path = os.path.join(TEST_IMAGE_DIR, image_name)
        mask_path = os.path.join(TEST_MASK_DIR, image_name)

        image_np = np.load(img_path).astype(np.float32)
        mask_np = np.load(mask_path).astype(np.uint8)
        if mask_np.ndim == 3:
            mask_np = np.squeeze(mask_np, axis=-1)

        predictions = self.predict(image_np)

        # Display images
        self.axes[0].imshow(image_np, cmap='gray')
        self.axes[0].set_title('Original Image')
        self.axes[0].axis('off')

        self.axes[1].imshow(mask_np, cmap='gray')
        self.axes[1].set_title('Ground Truth')
        self.axes[1].axis('off')

        # Check if models dictionary is not empty before iterating
        if self.models:
            for i, name in enumerate(self.models.keys()):
                # Ensure index is within bounds
                if i + 2 < len(self.axes):
                    self.axes[i + 2].imshow(predictions[name], cmap='gray')
                    self.axes[i + 2].set_title(name)
                    self.axes[i + 2].axis('off')
        
        self.fig.suptitle(f'Model Comparison: {image_name} ({self.current_index + 1}/{len(self.test_images)})', fontsize=16)
        plt.draw()

    def next_image(self, event):
        self.current_index = (self.current_index + 1) % len(self.test_images)
        self.update_plot()

    def prev_image(self, event):
        self.current_index = (self.current_index - 1 + len(self.test_images)) % len(self.test_images)
        self.update_plot()

def main():
    print("Loading models...")
    models = load_models()
    if not models:
        print("No models could be loaded. Aborting GUI launch.")
        return

    gui = ComparisonGUI(models)

    # Add buttons
    ax_prev = plt.axes([0.7, 0.01, 0.1, 0.075])
    ax_next = plt.axes([0.81, 0.01, 0.1, 0.075])
    btn_prev = Button(ax_prev, 'Previous')
    btn_next = Button(ax_next, 'Next')

    btn_prev.on_clicked(gui.prev_image)
    btn_next.on_clicked(gui.next_image)

    print("Launching Matplotlib GUI... Close the window to exit.")
    plt.show()

if __name__ == "__main__":
    main()