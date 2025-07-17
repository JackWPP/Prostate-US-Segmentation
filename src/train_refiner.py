
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Import the new composite model
from .unet_refiner import UNetWithMambaRefiner

# --- Configuration ---
DATA_DIR = 'processed_data/train'
MODEL_SAVE_PATH = 'models/unet_refiner_model.pth'
NUM_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# --- Dataset Class (same as other training scripts) ---
class ProstateDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(os.path.join(data_dir, 'images')) if f.endswith('.npy')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, 'images', img_name)
        mask_path = os.path.join(self.data_dir, 'masks', img_name)

        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        mask = mask.permute(2, 0, 1)
        return image, mask

# --- Transformations ---
train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

# --- Dice Loss ---
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

# --- Training Function ---
def train_model():
    print("Starting training for the UNetWithMambaRefiner model...")

    dataset = ProstateDataset(data_dir=DATA_DIR, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model without the final sigmoid for training
    model = UNetWithMambaRefiner(n_channels=1, n_classes=1, use_sigmoid=False).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True)
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images) # These are logits
            
            loss_bce = criterion_bce(outputs, masks)
            loss_dice = criterion_dice(torch.sigmoid(outputs), masks)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
