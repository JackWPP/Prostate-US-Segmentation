# Development Guide

This document provides a comprehensive guide for setting up the development environment and running the scripts for the Prostate US Segmentation project.

## 1. Project Overview

The goal of this project is to develop a deep learning model for segmenting the prostate in micro-ultrasound images. The current implementation uses the MicroSegNet architecture with a PyTorch backend.

## 2. Current Progress

The following milestones have been completed:

1.  **Data Preprocessing (`src/preprocess.py`):**
    *   Loads original NIFTI (`.nii.gz`) files.
    *   Processes 3D scans into 2D slices.
    *   Resizes images and masks to a uniform size (256x256).
    *   Normalizes image pixel values to the [0, 1] range.
    *   Applies data augmentation (flips, rotations, etc.) to the training set to improve model robustness.
    *   Saves the processed data as `.npy` files in the `processed_data/` directory.

2.  **Model Implementation (`src/model.py`):**
    *   Implemented the **MicroSegNet** architecture in PyTorch, based on the official paper and repository.
    *   The model follows a U-Net-like encoder-decoder structure, using DenseNet blocks in the encoder to enhance feature propagation.
    *   Corrected channel and tensor dimension mismatches to ensure smooth training.

3.  **Training Script (`src/train.py`):**
    *   A full training pipeline has been set up.
    *   Includes a custom `ProstateDataset` class for loading the preprocessed `.npy` data.
    *   Uses Dice Loss, a standard loss function for segmentation tasks.
    *   The training loop evaluates the model on the test set after each epoch and saves the model with the best Dice Score.

4.  **Verification Script (`src/verify_setup.py`):**
    *   A test script that runs a single training iteration to confirm that the environment, data loading, model, and training logic are all configured correctly.

5.  **GUI Predictor (`src/gui_predictor.py`):**
    *   A graphical user interface built with Tkinter for interactive prediction.
    *   Allows users to select any test image from a list and instantly view the original image, the ground-truth mask overlay, and the model's predicted mask overlay for easy comparison.

## 3. Setup for a New GPU Server

Follow these steps to set up the project on a new machine, such as a GPU-enabled server.

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd Prostate-US-Segmentation
```

### Step 2: Set Up a Python Environment

It is highly recommended to use a virtual environment to avoid conflicts with system-wide packages.

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venc\Scripts\activate
```

### Step 3: Install Dependencies

All required Python packages are listed in `requirements.txt`. The PyTorch version specified is for CUDA, so ensure you have a compatible NVIDIA GPU and drivers installed.

```bash
# Install all required packages
pip install -r requirements.txt
```

## 4. How to Run the Project

Make sure you have activated the virtual environment (`source venv/bin/activate`) before running the scripts.

### Step 1: Run Data Preprocessing

This only needs to be done once. This script will process the raw data from `dataset/` and create the `processed_data/` directory.

```bash
python src/preprocess.py
```

### Step 2: Verify the Setup (Optional but Recommended)

Before starting a long training session, run the verification script to ensure everything is working correctly.

```bash
python src/verify_setup.py
```
If it prints `[SUCCESS] All tests passed`, you are ready to train.

### Step 3: Start Model Training

This script will train the MicroSegNet model. Progress will be displayed in the console. The best-performing model will be saved automatically to the `models/` directory.

```bash
python src/train.py
```

## 5. Next Steps

As outlined in `docs/work.md`, the next phase of the project will focus on:

*   **Integrating Attention Mechanisms:** To help the model focus on more relevant features.
*   **Implementing Multi-Scale Feature Fusion:** To better capture details at different resolutions.
*   **Adding Deep Supervision:** To improve gradient flow and training for deeper networks.

