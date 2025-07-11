# Development Guide

This document provides a comprehensive guide for setting up the development environment and running the scripts for the Prostate US Segmentation project.

## 1. Project Overview

The goal of this project is to develop a deep learning model for segmenting the prostate in micro-ultrasound images. The current implementation uses the MicroSegNet architecture with a PyTorch backend.

## 2. Code Structure

The project follows a modular structure. Key components are located in the `src/` directory. A notable feature is the `src/models_zoo/` directory, designed to manage multiple model architectures independently.

- `src/models_zoo/base_model`: Contains the original `MicroSegNet` implementation.
- `src/models_zoo/attention_model`: Contains the `MicroSegNetAttention` model, which integrates the CBAM attention mechanism.
- `src/train.py`: Script for training the base model.
- `src/train_attention.py`: Script for training the attention model.
- `src/gui_predictor.py`: A Tkinter-based GUI for visual evaluation.

## 3. Current Progress

The following milestones have been completed:

1.  **Data Preprocessing (`src/preprocess.py`):** Processes raw data into `.npy` slices.
2.  **Model Implementation (`src/model.py`):** Implemented the base `MicroSegNet` architecture.
3.  **Training Script (`src/train.py`):** Set up a full training pipeline for the base model.
4.  **Verification Script (`src/verify_setup.py`):** A test script to ensure the environment is correctly configured.
5.  **GUI Predictor (`src/gui_predictor.py`):** A graphical user interface for interactive prediction and visualization.
6.  **Attention Mechanism Integration:**
    *   Refactored the project structure to support multiple models via a `models_zoo` directory.
    *   Implemented a standalone CBAM (Convolutional Block Attention Module).
    *   Created a new model, `MicroSegNetAttention`, which integrates CBAM into the decoder's skip connections.
    *   Provided a separate training script, `train_attention.py`, for the new model.

## 4. Setup for a New GPU Server

Follow these steps to set up the project on a new machine.

### Step 1: Clone the Repository
...
### Step 2: Set Up a Python Environment
...
### Step 3: Install Dependencies
...

## 5. How to Run the Project

Make sure you have activated the virtual environment before running the scripts.

### Step 1: Run Data Preprocessing
This only needs to be done once.
```bash
python src/preprocess.py
```

### Step 2: Verify the Setup (Optional)
This script checks the base model and data loading.
```bash
python src/verify_setup.py
```

### Step 3: Start Model Training
You can train either the base model or the new attention-enhanced model.

- **To train the base MicroSegNet model:**
  ```bash
  python src/train.py
  ```
  The best model will be saved in `models/best_microsegnet_model.pth`.

- **To train the MicroSegNetAttention model:**
  ```bash
  python src/train_attention.py
  ```
  The best model will be saved in `models/attention/best_microsegnet_attention_model.pth`.

## 6. Next Steps

As outlined in `docs/work.md`, the next phase of the project will focus on:

*   **Integrating Attention Mechanisms:** To help the model focus on more relevant features.
*   **Implementing Multi-Scale Feature Fusion:** To better capture details at different resolutions.
*   **Adding Deep Supervision:** To improve gradient flow and training for deeper networks.

