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

### Phase 1: Basic Framework

-   **Data Preprocessing (`src/preprocess.py`):** A complete script for processing raw NIFTI data into 2D `.npy` slices.
-   **Model Implementation:** The core `MicroSegNet` architecture was implemented in `src/models_zoo/base_model`.
-   **Training & Verification:** A full training pipeline (`src/train.py`) and a verification script (`src/verify_setup.py`) were established for the base model.

### Phase 2: Model Optimization & Refactoring

-   **Code Refactoring:** Created the `src/models_zoo/` directory to support multiple model architectures.
-   **Attention Mechanism:** Successfully implemented the **CBAM** attention module and integrated it into a new `MicroSegNetAttention` model.
-   **Dedicated Training Scripts:** Created a separate training script (`src/train_attention.py`) for the attention model.
-   **Advanced GUI Comparator:** Developed a sophisticated GUI tool (`src/gui_predictor.py`) for ablation studies. It dynamically loads all available models from the `models/` directory and allows for side-by-side comparison of any two models, greatly facilitating qualitative analysis.

### Phase 3: Exploring Advanced Architectures

-   **Integration of U-Net and TransUNet:** Successfully added `U-Net` and `TransUNet` to the project, each with its own dedicated training script.
-   **Implementation and Debugging of Mamba-UNet:**
    -   **Initial Implementation:** Implemented a `MambaUNet` based on the U-Mamba architecture, where Mamba blocks were used throughout the encoder.
    -   **Iterative Debugging:** The initial model failed to learn effectively. The debugging process involved several key steps:
        1.  **Activation Function Fix:** Corrected a conflict between the model's final `Sigmoid` layer and the `BCEWithLogitsLoss` function.
        2.  **Normalization Layer Fix:** Addressed issues with mixed `BatchNorm` and `LayerNorm` layers, which caused unstable training.
        3.  **Vanishing Gradient Fix:** Identified that an overly deep decoder was impeding gradient flow. The decoder was simplified to a more standard, robust U-Net design.
    -   **Final Hybrid Architecture:** The final, successful model is a **hybrid CNN-Mamba architecture**. It uses standard convolutional blocks in the early encoder stages to extract robust low-level features, and Mamba blocks in the deeper stages to model long-range dependencies. This design proved to be stable and effective in initial tests.

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

