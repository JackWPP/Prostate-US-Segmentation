

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Import the new attention model
from models_zoo.attention_model.model import MicroSegNetAttention

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSED_DATA_PATH = "processed_data"
TRAIN_IMG_PATH = os.path.join(PROCESSED_DATA_PATH, "train", "images")
TRAIN_MASK_PATH = os.path.join(PROCESSED_DATA_PATH, "train", "masks")
TEST_IMG_PATH = os.path.join(PROCESSED_DATA_PATH, "test", "images")
TEST_MASK_PATH = os.path.join(PROCESSED_DATA_PATH, "test", "masks")
# Save to a separate directory to avoid overwriting the base model
MODEL_SAVE_PATH = "models/attention" 

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 25
IMG_SIZE = 256

# --- Dataset (Re-used from original train script) ---
class ProstateDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.npy")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.npy")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.load(self.image_paths[idx])
        mask = np.load(self.mask_paths[idx])
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        return img, mask

# --- Loss Function (Re-used) ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dice_score = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1 - dice_score

# --- Training & Validation Functions (Re-used) ---
def train_one_epoch(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader, desc="Training")
    running_loss = 0.0
    for data, targets in loop:
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return running_loss / len(loader)

def validate_model(loader, model, loss_fn):
    model.eval()
    val_loss = 0.0
    dice_score = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            val_loss += loss_fn(preds, y).item()
            preds_flat = (preds > 0.5).float().view(-1)
            y_flat = y.view(-1)
            intersection = (preds_flat * y_flat).sum()
            score = (2. * intersection) / (preds_flat.sum() + y_flat.sum() + 1e-6)
            dice_score += score.item()
    avg_val_loss = val_loss / len(loader)
    avg_dice_score = dice_score / len(loader)
    print(f"Validation -> Avg. Loss: {avg_val_loss:.4f}, Avg. Dice Score: {avg_dice_score:.4f}")
    return avg_dice_score

def main():
    print(f"--- Training MicroSegNet with Attention ---")
    print(f"Using device: {DEVICE}")

    # Create datasets and dataloaders
    train_dataset = ProstateDataset(image_dir=TRAIN_IMG_PATH, mask_dir=TRAIN_MASK_PATH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_dataset = ProstateDataset(image_dir=TEST_IMG_PATH, mask_dir=TEST_MASK_PATH)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize the ATTENTION model
    model = MicroSegNetAttention(num_classes=1).to(DEVICE)
    loss_fn = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create model save directory if it doesn't exist
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    best_dice_score = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn)
        print(f"Epoch {epoch+1} Average Training Loss: {train_loss:.4f}")
        dice_score = validate_model(val_loader, model, loss_fn)

        if dice_score > best_dice_score:
            best_dice_score = dice_score
            # Save with a new name in the new directory
            model_path = os.path.join(MODEL_SAVE_PATH, "best_microsegnet_attention_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"-> New best attention model saved to {model_path} with Dice Score: {dice_score:.4f}")

    print("\nTraining complete.")
    print(f"Best Dice Score achieved for attention model: {best_dice_score:.4f}")

if __name__ == "__main__":
    main()

