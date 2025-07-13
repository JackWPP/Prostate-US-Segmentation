# Prostate-US-Segmentation

## 项目概述

本项目旨在研究和开发基于深度学习的前列腺超声图像自动分割算法，以解决临床实践中前列腺轮廓提取的难题，为前列腺疾病的诊断和治疗提供更精确的影像学辅助工具。

本项目将探索并实现先进的深度学习模型（以 **MicroSegNet** 为核心），以期克服超声图像存在的噪声大、对比度低、边界模糊等挑战，提高分割的准确性和鲁棒性。

---

## 快速上手指南

本节提供项目的快速安装和使用说明。更详细的步骤和解释，请参阅 [**开发文档 (docs/DEVELOPMENT.md)**](./docs/DEVELOPMENT.md)。

### 1. 环境设置

首先，克隆本仓库并进入项目目录。

```bash
git clone <your-repository-url>
cd Prostate-US-Segmentation
```

强烈建议使用Python虚拟环境。

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# Linux/macOS:
source venv/bin/activate
# Windows:
venc\Scripts\activate
```

### 2. 安装依赖

所有必需的库都列在 `requirements.txt` 中。

```bash
pip install -r requirements.txt
```

### 3. 运行项目

请按以下顺序执行脚本：

**第一步：数据预处理** (只需运行一次)
此脚本会处理 `dataset/` 中的原始数据，并将其保存在 `processed_data/` 目录中。

```bash
python src/preprocess.py
```

**第二步：验证设置** (可选，但推荐)
在开始完整训练前，运行此脚本可快速检查环境、模型和数据加载是否都已正确配置。

```bash
python src/verify_setup.py
```

**第三步：开始训练**
本项目提供了两个模型供训练：

- **训练基础模型 (MicroSegNet)**:
  ```bash
  python src/train.py
  ```

- **训练注意力模型 (MicroSegNet + CBAM)**:
  ```bash
  python src/train_attention.py
  ```

**第四步：模型对比与消融研究 (GUI)**
此脚本将启动一个为消融研究定制的图形化对比工具。

```bash
python src/gui_predictor.py
```
在GUI中，您可以：
- 从顶部下拉菜单中选择任意测试图像。
- 动态选择 `models/` 目录下的任意两个模型进行对比（模型A vs 模型B）。
- 在界面中实时查看 **原始图像**、**真实掩码**、**模型A预测** 和 **模型B预测** 的四图对比，直观评估不同模型之间的差异。

---
## 项目进度

- **[✔️] 阶段一：基础框架搭建**
  - [x] **数据预处理**: 实现了完整的脚本 (`src/preprocess.py`)。
  - [x] **模型实现**: 在 PyTorch 中成功实现了 MicroSegNet 核心架构。
  - [x] **训练与验证**: 搭建了完整的训练流程 (`src/train.py`) 和验证脚本。

- **[✔️] 阶段二：模型优化与重构**
  - [x] **代码结构重构**: 创建了 `models_zoo` 目录，以支持多模型管理，将不同模型架构解耦。
  - [x] **集成注意力机制**: 成功实现了 **CBAM** 注意力模块，并将其集成到新的 `MicroSegNetAttention` 模型中 (`src/models_zoo/attention_model`)。
  - [x] **独立训练流程**: 为注意力模型创建了专属的训练脚本 (`src/train_attention.py`)。
  - [x] **图形化预测工具**: 开发了一个基于 Tkinter 的 GUI (`src/gui_predictor.py`)。

---

## 路线图

- **[▶️] 阶段三：实验与评估 (进行中)**
  - [x] **训练与评估注意力模型**: 执行 `train_attention.py` 并评估其性能。
  - [x] **更新GUI对比工具**: 增强 `gui_predictor.py`，使其能够同时加载和对比基础模型与注意力模型的分割结果。
  - [ ] **进行消融研究**: 系统性地评估不同模块（如CBAM）对模型性能的贡献。
  - [ ] **与其他模型对比**: 将优化后的模型与 U-Net、TransUNet 等其他经典或前沿模型进行性能比较。

- **[ ] 阶段四：进一步优化**
  - [ ] **探索多尺度特征融合**: 研究并实现更高效的特征金字塔或类似结构。
  - [ ] **应用深度监督**: 为模型的中间层添加辅助损失函数，以改善梯度流。

---

## 技术栈

*   **编程语言**: Python
*   **深度学习框架**: **PyTorch**
*   **核心模型**: MicroSegNet, U-Net
*   **主要库**: OpenCV, Albumentations, Nibabel, NumPy

## 数据集

本项目使用的开源数据集可从 Zenodo 获取：[Micro-Ultrasound Prostate Segmentation Dataset](https://zenodo.org/records/10475293)。

## 后续任务

项目的未来工作将聚焦于集成**注意力机制**、**多尺度特征融合**和**深度监督**等先进技术，以进一步提升模型的分割精度。

## 许可证

本项目采用 [MIT](./LICENSE) 许可证。