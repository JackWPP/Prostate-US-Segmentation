# Prostate-US-Segmentation

## 项目概述

本项目旨在研究和开发基于深度学习的前列腺超声图像自动分割算法，以解决临床实践中前列腺轮廓提取的难题，为前列腺疾病的诊断和治疗提供更精确的影像学辅助工具。

本项目最终成功实现并评估了多种先进的深度学习模型，包括经典的 **MicroSegNet**、**UNet**，以及作为本项目核心创新的 **MicroSegNetAttention**。

---

## 快速上手指南

本节提供项目的快速安装和使用说明。更详细的步骤和解释，请参阅 [**开发文档 (docs/DEVELOPMENT.md)**](./docs/DEVELOPMENT.md)。

### 1. 环境设置

首先，克隆本仓库并进入项目目录，并安装所需的依赖库。

```bash
git clone <your-repository-url>
cd Prostate-US-Segmentation
pip install -r requirements.txt
```

### 2. 运行项目

请按以下顺序执行脚本：

**第一步：数据预处理** (只需运行一次)
```bash
python -m src.preprocess
```

**第二步：开始训练**
您可以选择训练项目中的任意模型。`MicroSegNetAttention` 在测试中表现最佳，是我们的推荐模型。

- **训练注意力模型 (推荐)**:
  ```bash
  python -m src.train_attention
  ```
- **训练基础模型 (MicroSegNet)**:
  ```bash
  python -m src.train
  ```
- **训练 U-Net 模型**:
  ```bash
  python -m src.train_unet
  ```
- **训练 TransUNet 模型 (当前权重无效)**:
  ```bash
  python -m src.train_transunet
  ```

**第三步：模型对比与评估 (GUI)**
此脚本将启动一个图形化对比工具，您可以直观地评估和对比不同模型的分割效果。
```bash
python -m src.gui_predictor
```

---

## 模型性能

项目最终阶段对所有核心模型进行了全面的定量评估，结果如下。

| model                |      dice |       iou |   precision |      recall |   hausdorff_95 |
|:---------------------|----------:|----------:|------------:|------------:|---------------:|
| **MicroSegNetAttention** | **0.9317** | **0.8844** | **0.9236** | **0.9486** | **13.6580** |
| MicroSegNet          |  0.9291 |  0.8799 |    0.9213 |    0.9458 |      14.1911   |
| UNet                 |  0.9235 |  0.8711 |    0.9187 |    0.9361 |      14.8810   |
| TransUNet            |  0.0006 |  0.0003 |    0.0004 |    0.0018 |     179.5045   |

### 结论分析
*   **最佳模型:** `MicroSegNetAttention` 在所有关键指标上均取得了最佳或接近最佳的成绩，特别是在 Dice/IoU 和 Hausdorff 距离上表现出色，是本项目**最平衡、最强大的模型**。
*   **TransUNet 问题:** 尽管评估脚本已修复，但 `TransUNet` 的评估结果表明，当前保存的权重文件是一个训练失败或未经训练的模型。

---

## 核心创新：MicroSegNet with Attention

本项目的核心创新在于将 **CBAM (Convolutional Block Attention Module)** 注意力机制成功集成到了 `MicroSegNet` 的密集连接层中，创建了 `MicroSegNetAttention` 模型。

这种方法让模型能够自适应地“聚焦”于图像中的关键特征区域，从而在几乎不增加计算成本的前提下，有效提升了分割的准确性和边界质量。

更详细的技术实现和分析，请参阅我们的技术报告：
[**《在MicroSegNet中集成CBAM注意力机制》**](./docs/ATTENTION_MECHANISM.md)

---

## 技术栈

*   **编程语言**: Python
*   **深度学习框架**: **PyTorch**
*   **核心模型**: MicroSegNet, U-Net, MicroSegNetAttention
*   **主要库**: OpenCV, Albumentations, Nibabel, NumPy, Timm, MedPy

## 数据集

本项目使用的开源数据集可从 Zenodo 获取：[Micro-Ultrasound Prostate Segmentation Dataset](https://zenodo.org/records/10475293)。

## 许可证

本项目采用 [MIT](./LICENSE) 许可证。
