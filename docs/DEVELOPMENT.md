# Development Guide

This document provides a comprehensive guide for setting up the development environment and running the scripts for the Prostate US Segmentation project.

## 1. Project Overview

The goal of this project is to develop a deep learning model for segmenting the prostate in micro-ultrasound images. The current implementation uses the MicroSegNet architecture with a PyTorch backend.

## 2. Code Structure

The project follows a modular structure. Key components are located in the `src/` directory. A notable feature is the `src/models_zoo/` directory, designed to manage multiple model architectures independently.

- `src/models_zoo/base_model`: Contains the original `MicroSegNet` implementation.
- `src/models_zoo/attention_model`: Contains the `MicroSegNetAttention` model.
- `src/hm_segnet.py`: Contains the experimental `HMSegNet` model, which integrates Mamba blocks into the `MicroSegNet` architecture.
- `src/gui_predictor.py`: A Tkinter-based GUI for visual evaluation.

## 3. Current Progress

### Phase 1 & 2: Foundational Work
- Established the base `MicroSegNet` and `MicroSegNetAttention` models and their training pipelines.

### Phase 3: Integration of Advanced Architectures
- Successfully integrated standard `U-Net` and `TransUNet` models into the project, providing a robust set of baseline and advanced architectures.

### Phase 4: HM-SegNet Implementation
- **Correct Interpretation:** After multiple unsuccessful attempts to integrate Mamba with a standard U-Net, a thorough review of `docs/mamba.md` revealed the correct approach: integrating Mamba with the DenseNet-like structure of `MicroSegNet`.
- **Successful Implementation:** A new model, `HMSegNet`, was created in `src/hm_segnet.py`. This was achieved by inheriting from the base `MicroSegNet` class and replacing the convolutional `_DenseLayer` with a new `_MambaDenseLayer` in the encoder's `_DenseBlock`s. This approach correctly follows the technical documentation and preserves the model's critical skip-connection logic.
- **Training Pipeline:** A dedicated training script, `src/train_hm_segnet.py`, was created for the new model.

## 4. How to Run the Project

### Step 1: Run Data Preprocessing
This only needs to be done once.
```bash
python -m src.preprocess
```

### Step 2: Start Model Training
You can train any of the available models. `TransUNet` is the recommended production-ready model, while `HMSegNet` is the recommended experimental model.

- **To train the base MicroSegNet model:**
  ```bash
  python -m src.train
  ```
- **To train the TransUNet model:**
  ```bash
  python -m src.train_transunet
  ```
- **To train the HMSegNet model:**
  ```bash
  python -m src.train_hm_segnet
  ```

## 5. Next Steps

The primary focus is now on training and evaluating the existing, powerful models in the repository.

*   **Train and Evaluate:** Perform full training runs (50-100 epochs) on `TransUNet` and `HMSegNet` to compare their performance.
*   **Quantitative Analysis:** Write scripts to calculate and compare key metrics (Dice, IoU, etc.) for all models.
*   **Qualitative Analysis:** Use the GUI tool to visually compare the results and identify strengths and weaknesses of each model.
