# Prostate-US-Segmentation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 项目概述

本项目旨在研究和开发基于深度学习的前列腺超声图像自动分割算法，以解决临床实践中前列腺轮廓提取的难题，为前列腺疾病的诊断和治疗提供更精确的影像学辅助工具。

项目最终成功实现并评估了多种先进的深度学习模型，包括经典的 **MicroSegNet**、**UNet**，以及作为本项目核心创新的 **MicroSegNetAttention**。

---

## ✨ 项目亮点

- **多模型实现:** 成功实现并对比了多种经典及创新的分割架构。
- **核心创新:** 创新性地将 **CBAM 注意力机制** 与 `MicroSegNet` 结合，显著提升了模型的性能均衡性。
- **全面评估体系:** 开发了包含**定量指标计算**、**定性效果对比**和**像素级混淆矩阵**在内的完整评估流程。
- **丰富的可视化工具:** 构建了**三种**不同的可视化工具，以满足从快速浏览到生成出版级图表���多种定性分析需求。
- **详尽的文档:** 为项目的核心创新、评估计划和最终成果撰写了清晰、专业的技术文档。

---

## 🚀 快速上手

### 1. 环境设置
```bash
# 克隆仓库
git clone <your-repository-url>
cd Prostate-US-Segmentation

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据预处理
> ⚠️ 此步骤只需运行一次。
```bash
python -m src.preprocess
```

### 3. 模型训练
> `MicroSegNetAttention` 在我们的测试中表现最佳，是推荐的优先训练模型。
```bash
# 训练注意力模型 (🏆 推荐)
python -m src.train_attention

# 训练基础模型
python -m src.train

# 训练 U-Net 模型
python -m src.train_unet
```

---

## 📊 模型性能

项目最终阶段对所有核心模型进行了全面的定量评估，结果如下。

| model                |      dice |       iou |   hausdorff_95 |
|:---------------------|----------:|----------:|---------------:|
| **MicroSegNetAttention** | **0.9317** | **0.8844** | **13.6580** |
| MicroSegNet          |  0.9291 |  0.8799 |      14.1911   |
| UNet                 |  0.9235 |  0.8711 |      14.8810   |
| TransUNet            |  0.0006 |  0.0003 |     179.5045   |

### 结论分析
*   **最佳模型:** `MicroSegNetAttention` 在所有关键指标上均取得了最佳或接近最佳的成绩，是本项目**最平衡、最强大的模型**。
*   **TransUNet 问题:** `TransUNet` 的评估结果表明，当前保存的权重文件是一个训练失败或未经训练的模型。

### 最佳模型可视化效果

下图展示了 `MicroSegNetAttention` 模型在一个典型测试样本上的分割效果。

![Segmentation Result](output/segmentation_comparison.png)

---

## 🔬 核心创新：MicroSegNet with Attention

本项目的核心创新在于将 **CBAM (Convolutional Block Attention Module)** 注意力机制成功集成到了 `MicroSegNet` 的密集连接层中。这种方法让模型能够自适应地“聚焦”于图像中的关键特征区域，从而在几乎不增加计算成本的前提下，有效提升了分割的准确性和边界质量。

> 更详细的技术实现和分析，请参阅我们的技术报告：
> **[《在MicroSegNet中集成CBAM注意力机制》](./docs/ATTENTION_MECHANISM.md)**

---

## 🛠️ 可视化与分析工具

本项目开发了多种工具，以支持全面的定性分析。

### 1. 交互式多模型对比工具 (Matplotlib GUI)
一个本地 GUI 窗口，可并排比较所有模型的分割结果，并通过按钮浏览整个测试集。
```bash
python -m src.gui_matplotlib_compare
```

### 2. 生成出版级对比图
为指定的测试图像生成一个 2x2 的、包含多个模型对比结果的网格图，适合用于报告或论文。
```bash
# 示例:
python -m src.generate_comparison_grid patient_001_slice_002.npy
```

### 3. 生成混淆矩阵
为指定的模型计算并绘制一个美观的、包含像素数和百分比的混淆矩阵热力图。
```bash
# 示例:
python -m src.generate_confusion_matrix MicroSegNetAttention
```

---

## 💻 技术栈

*   **编程语言**: Python
*   **深度学习框架**: **PyTorch**
*   **核心模型**: MicroSegNet, U-Net, MicroSegNetAttention
*   **主要库**: OpenCV, Albumentations, Nibabel, NumPy, Timm, MedPy, Seaborn

## 📄 许可证

本项目采用 [MIT](./LICENSE) 许可证。