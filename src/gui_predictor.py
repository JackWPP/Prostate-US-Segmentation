
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import os
import torch
import cv2
from .models_zoo.base_model.model import MicroSegNet

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_microsegnet_model.pth')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed_data', 'test')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
IMG_SIZE = (256, 256)
# Colors for overlay (BGR format for OpenCV)
GT_MASK_COLOR = [0, 255, 0]  # Green
PRED_MASK_COLOR = [0, 0, 255] # Red

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Prostate Segmentation Predictor")
        self.geometry("950x450")

        # --- Load Model ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        print(f"Model loaded on {self.device}")

        # --- Panes ---
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # --- Left Pane (File List) ---
        list_frame = ttk.Frame(main_pane, padding=10)
        main_pane.add(list_frame, weight=1)
        
        ttk.Label(list_frame, text="Test Images").pack(anchor=tk.W)
        
        list_container = ttk.Frame(list_frame, borderwidth=1, relief="solid")
        list_container.pack(fill=tk.BOTH, expand=True)
        
        self.listbox = tk.Listbox(list_container)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)
        
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        # --- Right Pane (Image Display) ---
        self.image_frame = ttk.Frame(main_pane, padding=10)
        main_pane.add(self.image_frame, weight=4)

        self.canvas1 = self.create_image_canvas("Original Image")
        self.canvas2 = self.create_image_canvas("Ground Truth Overlay")
        self.canvas3 = self.create_image_canvas("Prediction Overlay")

        # --- Populate List ---
        self.populate_listbox()

    def create_image_canvas(self, title):
        frame = ttk.Frame(self.image_frame)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(frame, text=title, anchor=tk.CENTER).pack(pady=5)
        canvas = tk.Canvas(frame, width=IMG_SIZE[0], height=IMG_SIZE[1], bg="white", relief="solid", borderwidth=1)
        canvas.pack()
        return canvas

    def load_model(self):
        try:
            model = MicroSegNet(num_classes=1)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            self.destroy()

    def populate_listbox(self):
        try:
            files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.npy')])
            if not files:
                self.listbox.insert(tk.END, "No images found.")
                return
            for f in files:
                self.listbox.insert(tk.END, f)
            # Select the first item by default
            self.listbox.selection_set(0)
            self.on_select(None)
        except FileNotFoundError:
            self.listbox.insert(tk.END, f"Error: Path not found.")
        except Exception as e:
            self.listbox.insert(tk.END, f"An error occurred: {e}")

    def on_select(self, event):
        selection_indices = self.listbox.curselection()
        if not selection_indices:
            return
        
        filename = self.listbox.get(selection_indices[0])
        self.update_images(filename)

    def update_images(self, filename):
        try:
            # --- Load Data ---
            img_np = np.load(os.path.join(IMAGE_DIR, filename))
            gt_mask_np = np.load(os.path.join(MASK_DIR, filename))

            # --- Predict ---
            pred_mask_np = self.predict(img_np)

            # --- Create Overlays ---
            original_img_display = self.prepare_image_for_display(img_np)
            gt_overlay_display = self.create_overlay(img_np, gt_mask_np, GT_MASK_COLOR)
            pred_overlay_display = self.create_overlay(img_np, pred_mask_np, PRED_MASK_COLOR)

            # --- Update GUI ---
            self.photo1 = ImageTk.PhotoImage(image=Image.fromarray(original_img_display))
            self.photo2 = ImageTk.PhotoImage(image=Image.fromarray(gt_overlay_display))
            self.photo3 = ImageTk.PhotoImage(image=Image.fromarray(pred_overlay_display))

            self.canvas1.create_image(0, 0, anchor=tk.NW, image=self.photo1)
            self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.photo2)
            self.canvas3.create_image(0, 0, anchor=tk.NW, image=self.photo3)
        except Exception as e:
            print(f"Failed to update images for {filename}: {e}")

    def predict(self, img_np):
        if img_np.ndim == 3:
            img_np = img_np[:, :, 0]
        
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            pred_tensor = self.model(img_tensor)
            
        pred_np = pred_tensor.cpu().squeeze().numpy()
        pred_mask = (pred_np > 0.5).astype(np.uint8)
        return pred_mask

    def prepare_image_for_display(self, img):
        # Normalize and convert to 3-channel BGR for display
        if img.ndim == 3:
            img = img[:, :, 0]
        img_display = (img * 255).astype(np.uint8)
        return cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)

    def create_overlay(self, img, mask, color):
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        
        # Prepare original image
        overlay = self.prepare_image_for_display(img)
        
        # Create a color mask
        color_mask = np.zeros_like(overlay, dtype=np.uint8)
        color_mask[mask == 1] = color
        
        # Blend the images
        # Add weighted combines the two images
        return cv2.addWeighted(overlay, 1, color_mask, 0.5, 0)

if __name__ == "__main__":
    app = App()
    app.mainloop()
