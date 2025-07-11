

import os
import glob
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm
import albumentations as A

# --- Configuration ---
# Input data paths
BASE_DATA_PATH = "dataset/Micro_Ultrasound_Prostate_Segmentation_Dataset"
TRAIN_PATH = os.path.join(BASE_DATA_PATH, "train")
TEST_PATH = os.path.join(BASE_DATA_PATH, "test")

# Output data paths
PROCESSED_DATA_PATH = "processed_data"
PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DATA_PATH, "train")
PROCESSED_TEST_PATH = os.path.join(PROCESSED_DATA_PATH, "test")
PROCESSED_TRAIN_IMAGES_PATH = os.path.join(PROCESSED_TRAIN_PATH, "images")
PROCESSED_TRAIN_MASKS_PATH = os.path.join(PROCESSED_TRAIN_PATH, "masks")
PROCESSED_TEST_IMAGES_PATH = os.path.join(PROCESSED_TEST_PATH, "images")
PROCESSED_TEST_MASKS_PATH = os.path.join(PROCESSED_TEST_PATH, "masks")

# Preprocessing parameters
IMG_SIZE = (256, 256)

# Augmentation pipeline for the training set
# We apply geometric transformations that are common for medical imaging tasks.
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.RandomBrightnessContrast(p=0.2),
])

def create_dir(path):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def process_and_save_nifti(image_paths, mask_paths, output_image_path, output_mask_path, augment=False):
    """
    Loads NIFTI files, processes them slice by slice, and saves them as .npy files.
    Each slice of the 3D NIFTI scan is treated as an individual 2D image.
    """
    patient_id = 1
    for img_path, msk_path in tqdm(zip(sorted(image_paths), sorted(mask_paths)), total=len(image_paths), desc=f"Processing {'Train' if augment else 'Test'} Set"):
        try:
            # Load NIFTI files
            img_nifti = nib.load(img_path)
            msk_nifti = nib.load(msk_path)

            img_data = img_nifti.get_fdata()
            msk_data = msk_nifti.get_fdata()

            # Process each slice
            for i in range(img_data.shape[2]):
                slice_img = img_data[:, :, i]
                slice_msk = msk_data[:, :, i]

                # Skip empty masks to focus on slices with the prostate
                if np.sum(slice_msk) == 0:
                    continue

                # Resize
                slice_img = cv2.resize(slice_img, IMG_SIZE, interpolation=cv2.INTER_AREA)
                slice_msk = cv2.resize(slice_msk, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

                # Normalize image to [0, 1]
                slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-6)
                slice_img = slice_img.astype(np.float32)

                # Ensure mask is binary [0, 1]
                slice_msk = (slice_msk > 0).astype(np.uint8)

                # Apply augmentation if specified
                if augment:
                    augmented = augmentation_pipeline(image=slice_img, mask=slice_msk)
                    slice_img = augmented['image']
                    slice_msk = augmented['mask']

                # Add channel dimension for compatibility with deep learning models
                slice_img = np.expand_dims(slice_img, axis=-1)
                slice_msk = np.expand_dims(slice_msk, axis=-1)

                # Save processed files
                slice_filename = f"patient_{patient_id:03d}_slice_{i:03d}.npy"
                np.save(os.path.join(output_image_path, slice_filename), slice_img)
                np.save(os.path.join(output_mask_path, slice_filename), slice_msk)

            patient_id += 1

        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {e}")


def main():
    """
    Main function to run the data preprocessing and augmentation pipeline.
    """
    print("Starting data preprocessing and augmentation...")

    # --- Create output directories ---
    print("Creating output directories...")
    create_dir(PROCESSED_TRAIN_IMAGES_PATH)
    create_dir(PROCESSED_TRAIN_MASKS_PATH)
    create_dir(PROCESSED_TEST_IMAGES_PATH)
    create_dir(PROCESSED_TEST_MASKS_PATH)

    # --- Get file paths ---
    train_image_paths = glob.glob(os.path.join(TRAIN_PATH, "micro_ultrasound_scans", "*.nii.gz"))
    train_mask_paths = glob.glob(os.path.join(TRAIN_PATH, "expert_annotations", "*.nii.gz"))
    test_image_paths = glob.glob(os.path.join(TEST_PATH, "micro_ultrasound_scans", "*.nii.gz"))
    test_mask_paths = glob.glob(os.path.join(TEST_PATH, "expert_annotations", "*.nii.gz"))

    # --- Process Training Data (with Augmentation) ---
    process_and_save_nifti(train_image_paths, train_mask_paths, PROCESSED_TRAIN_IMAGES_PATH, PROCESSED_TRAIN_MASKS_PATH, augment=True)

    # --- Process Test Data (without Augmentation) ---
    process_and_save_nifti(test_image_paths, test_mask_paths, PROCESSED_TEST_IMAGES_PATH, PROCESSED_TEST_MASKS_PATH, augment=False)

    print("\nData preprocessing and augmentation complete.")
    print(f"Processed data saved in: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
