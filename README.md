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
python -m src.preprocess
```

**第二步：开始训练**
您可以选择训练项目中的任意模型：

- **训练基础模型 (MicroSegNet)**:
  ```bash
  python -m src.train
  ```

- **训���注意力模型 (MicroSegNet + CBAM)**:
  ```bash
  python -m src.train_attention
  ```

- **训练 U-Net 模型**:
  ```bash
  python -m src.train_unet
  ```

- **训练 TransUNet 模型**:
  ```bash
  python -m src.train_transunet
  ```

**第三步：模型对比与消融研究 (GUI)**
此脚本将启动一个为消融研究定制的、可扩展的图形化对比工具。

```bash
python -m src.gui_predictor
```
在GUI中，您可以：
- 从顶部下拉菜单中选择任意测试图像。
- 动态选择 `models/` 目录下的任意两个模型进行对比（模型A vs 模型B）。
- GUI会自动识别并加载 `MicroSegNet`, `Attention`, `U-Net`, `TransUNet` 等所有已支持的模型。
- 在界面中实时查看 **原始图像**、**真实掩码**、**模型A预测** 和 **模型B预测** 的四图对比，直观评估不同模型之间的差异。

---
## 项目进度

- **[✔️] 阶段一：基础框架搭建**
  - [x] **数据预处理**: 实现了完整的脚本 (`src/preprocess.py`)。
  - [x] **模型实现**: 在 PyTorch 中成功实现了 MicroSegNet 核心架构。
  - [x] **训练与验证**: 搭建了完整的训练流程 (`src/train.py`) 和验证脚本。

- **[✔️] 阶段二：模型优化与重构**
  - [x] **代码结构重构**: 创建了 `models_zoo` 目录，以支持多模型管理，将不同模型架构解耦。
  - [x] **集成注意力机制**: 成功实现了 **CBAM** 注意力模块，并将其集成到新的 `MicroSegNetAttention` 模型中。
  - [x] **独立训练流程**: 为注意力模型创建了专属的训练脚本 (`src/train_attention.py`)。

- **[✔️] 阶段三：实验与评估**
  - [x] **集成新模型**: 成功将 **U-Net** 和 **TransUNet** 集成到项目中，并提供了独立的训练脚本。
  - [x] **开发通用对比工具**: 将GUI工具升级为一个可扩展的、支持所有已实现模型的通用对比平台，极大地便利了消融实验和定性分析。

---

## 路线图

- **[▶️] 阶段四：训练、评估与分析 (进行中)**
  - [ ] **全面训练**: 执行所有模型的训练脚本，获取最优权重。
  - [ ] **定量分析**: 编写脚本计算并对比不同模型在测试集上的 Dice Score, IoU, Precision, Recall 等关键指标。
  - [ ] **定性分析**: 使用最终的GUI工具，进行可视化对比，分析不同模型在具体病例上的优劣。
  - [ ] **准备研究报告**: 整理所有实验结果，撰写详细的技术报告或论文。

- **[ ] 阶段五：进一步优化**
  - [ ] **探索多尺度特征融合**: 研究并实现更高效的特征金字塔或类似结构。
  - [ ] **应用深度监督**: 为模型的中间层添加辅助损失函数，以改善梯度流。

---

## 技术栈

*   **编程语言**: Python
*   **深度学习框架**: **PyTorch**
*   **核心模型**: MicroSegNet, U-Net, TransUNet, Attention U-Net
*   **主要库**: OpenCV, Albumentations, Nibabel, NumPy, Timm

## 数据集

本项目使用的开源数据集可从 Zenodo 获取：[Micro-Ultrasound Prostate Segmentation Dataset](https://zenodo.org/records/10475293)。

## 许可证

本项目采用 [MIT](./LICENSE) 许可证。