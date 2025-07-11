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
此脚本将启动 MicroSegNet 模型的训练流程。表现最佳的模型将被保存在 `models/` 目录下。

```bash
python src/train.py
```

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