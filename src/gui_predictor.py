
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import os
import torch
import cv2
import sys
import importlib

# Add project root to the Python path to allow imports from 'src'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Configuration ---
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'processed_data', 'test')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
IMG_SIZE = (256, 256)
# Colors for overlay (BGR format for OpenCV)
GT_MASK_COLOR = [0, 255, 0]  # Green
PRED_MASK_COLOR_A = [0, 0, 255] # Red
PRED_MASK_COLOR_B = [255, 0, 0] # Blue

# --- Model Registry ---
# This dictionary maps a key (derived from the model path) to the model's class
# and the module where it's defined. This makes the GUI scalable to new models.
MODEL_REGISTRY = {
    "base": ("src.models_zoo.base_model.model", "MicroSegNet"),
    "attention": ("src.models_zoo.attention_model.model", "MicroSegNetAttention"),
    "unet": ("src.models_zoo.unet.model", "UNet"),
    "transunet": ("src.models_zoo.transunet.model", "TransUNet"),
    "mamba": ("src.mamba_unet", "MambaUNet"),
}

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Universal Model Comparator")
        self.geometry("1280x550")

        self.loaded_models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        top_frame = ttk.Frame(self, padding=10)
        top_frame.pack(fill=tk.X)
        bottom_frame = ttk.Frame(self, padding=10)
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        
        self.setup_controls(top_frame)
        self.setup_image_displays(bottom_frame)
        self.populate_dropdowns()
        self.after(100, self.initial_load)

    def setup_controls(self, parent):
        # ... (GUI setup code remains the same)
        img_frame = ttk.LabelFrame(parent, text="Test Image", padding=5)
        img_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.image_var = tk.StringVar()
        self.image_dropdown = ttk.Combobox(img_frame, textvariable=self.image_var, state="readonly", width=40)
        self.image_dropdown.pack(fill=tk.X)
        self.image_dropdown.bind("<<ComboboxSelected>>", self.on_selection_change)

        model_a_frame = ttk.LabelFrame(parent, text="Model A", padding=5)
        model_a_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.model_a_var = tk.StringVar()
        self.model_a_dropdown = ttk.Combobox(model_a_frame, textvariable=self.model_a_var, state="readonly", width=40)
        self.model_a_dropdown.pack(fill=tk.X)
        self.model_a_dropdown.bind("<<ComboboxSelected>>", self.on_selection_change)

        model_b_frame = ttk.LabelFrame(parent, text="Model B", padding=5)
        model_b_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.model_b_var = tk.StringVar()
        self.model_b_dropdown = ttk.Combobox(model_b_frame, textvariable=self.model_b_var, state="readonly", width=40)
        self.model_b_dropdown.pack(fill=tk.X)
        self.model_b_dropdown.bind("<<ComboboxSelected>>", self.on_selection_change)

    def setup_image_displays(self, parent):
        # ... (GUI setup code remains the same)
        self.canvas_orig = self.create_image_canvas(parent, "Original Image")
        self.canvas_gt = self.create_image_canvas(parent, "Ground Truth (Green)")
        self.canvas_pred_a = self.create_image_canvas(parent, "Prediction A (Red)")
        self.canvas_pred_b = self.create_image_canvas(parent, "Prediction B (Blue)")

    def create_image_canvas(self, parent, title):
        # ... (GUI setup code remains the same)
        frame = ttk.Frame(parent)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(frame, text=title, anchor=tk.CENTER).pack(pady=5)
        canvas = tk.Canvas(frame, width=IMG_SIZE[0], height=IMG_SIZE[1], bg="white", relief="solid", borderwidth=1)
        canvas.pack()
        return canvas

    def get_model_paths(self):
        # ... (This function remains the same)
        paths = []
        for root, _, files in os.walk(MODELS_DIR):
            for file in files:
                if file.endswith(".pth"):
                    rel_path = os.path.relpath(os.path.join(root, file), MODELS_DIR)
                    paths.append(rel_path)
        return sorted(paths)

    def populate_dropdowns(self):
        # ... (This function remains the same)
        try:
            image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.npy')])
            self.image_dropdown['values'] = image_files
            if image_files:
                self.image_var.set(image_files[0])
        except Exception as e:
            print(f"Error reading images: {e}")

        model_paths = self.get_model_paths()
        if model_paths:
            self.model_a_dropdown['values'] = model_paths
            self.model_b_dropdown['values'] = model_paths
            self.model_a_var.set(model_paths[0])
            self.model_b_var.set(model_paths[0])
        else:
            print("No .pth models found in the models directory.")

    def initial_load(self):
        self.on_selection_change(None)

    def on_selection_change(self, event):
        # ... (This function remains the same)
        image_file = self.image_var.get()
        model_a_path = self.model_a_var.get()
        model_b_path = self.model_b_var.get()

        if not all([image_file, model_a_path, model_b_path]):
            return

        try:
            img_np = np.load(os.path.join(IMAGE_DIR, image_file))
            gt_mask_np = np.load(os.path.join(MASK_DIR, image_file))

            pred_mask_a = self.predict(img_np, model_a_path)
            pred_mask_b = self.predict(img_np, model_b_path)

            original_img_display = self.prepare_image_for_display(img_np)
            gt_overlay_display = self.create_overlay(img_np, gt_mask_np, GT_MASK_COLOR)
            pred_a_overlay_display = self.create_overlay(img_np, pred_mask_a, PRED_MASK_COLOR_A)
            pred_b_overlay_display = self.create_overlay(img_np, pred_mask_b, PRED_MASK_COLOR_B)

            self.photo_orig = self.to_photoimage(original_img_display)
            self.photo_gt = self.to_photoimage(gt_overlay_display)
            self.photo_pred_a = self.to_photoimage(pred_a_overlay_display)
            self.photo_pred_b = self.to_photoimage(pred_b_overlay_display)

            self.canvas_orig.create_image(0, 0, anchor=tk.NW, image=self.photo_orig)
            self.canvas_gt.create_image(0, 0, anchor=tk.NW, image=self.photo_gt)
            self.canvas_pred_a.create_image(0, 0, anchor=tk.NW, image=self.photo_pred_a)
            self.canvas_pred_b.create_image(0, 0, anchor=tk.NW, image=self.photo_pred_b)

        except Exception as e:
            print(f"Failed to update images: {e}")

    def predict(self, img_np, model_rel_path):
        # ... (This function remains the same)
        model = self.get_model(model_rel_path)
        if model is None:
            return np.zeros(IMG_SIZE, dtype=np.uint8)

        if img_np.ndim == 3:
            img_np = img_np[:, :, 0]
        
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            pred_tensor = model(img_tensor)
            
        pred_np = pred_tensor.cpu().squeeze().numpy()
        return (pred_np > 0.5).astype(np.uint8)

    def get_model(self, model_rel_path):
        """
        Dynamically loads a model using the MODEL_REGISTRY.
        This is the key change for scalability.
        """
        if model_rel_path in self.loaded_models:
            return self.loaded_models[model_rel_path]
        
        try:
            print(f"Loading model: {model_rel_path}...")
            
            # Determine model type from path
            # Sort keys by length (descending) to match more specific names first
            model_key = "base" # Default
            sorted_keys = sorted(MODEL_REGISTRY.keys(), key=len, reverse=True)
            for key in sorted_keys:
                if key in model_rel_path.lower():
                    model_key = key
                    break
            
            print(f"Identified model type: '{model_key}'")
            
            module_path, class_name = MODEL_REGISTRY[model_key]
            
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            full_path = os.path.join(MODELS_DIR, model_rel_path)
            
            # Handle special case for TransUNet needing img_size
            if model_key == 'transunet':
                model = model_class(img_size=IMG_SIZE[0], num_classes=1, pretrained=False)
            else:
                model = model_class(n_classes=1) if model_key == 'unet' else model_class(num_classes=1)

            model.load_state_dict(torch.load(full_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.loaded_models[model_rel_path] = model
            print("...done.")
            return model
        except Exception as e:
            print(f"Error loading model {model_rel_path}: {e}")
            return None

    def prepare_image_for_display(self, img):
        # ... (This function remains the same)
        if img.ndim == 3:
            img = img[:, :, 0]
        img_display = (img * 255).astype(np.uint8)
        return cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)

    def create_overlay(self, img, mask, color):
        # ... (This function remains the same)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        
        overlay = self.prepare_image_for_display(img)
        color_mask = np.zeros_like(overlay, dtype=np.uint8)
        binary_mask = mask > 0
        color_mask[binary_mask] = color
        
        return cv2.addWeighted(overlay, 1, color_mask, 0.5, 0)

    def to_photoimage(self, array):
        # ... (This function remains the same)
        return ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB)))

if __name__ == "__main__":
    app = App()
    app.mainloop()
