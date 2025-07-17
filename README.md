# Prostate-US-Segmentation

## 项目概述

本项目旨在研究和开发基于深度学习的前列腺超声图像自动分割算法，以解决临床实践中前列腺轮廓提取的难题，为前列腺疾病的诊断和治疗提供更精确的影像学辅助工具。

本项目将探索并实现先进的深度学习模型，包括经典的 **MicroSegNet** 和 **TransUNet**。

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

**第一步：数据预处理** (只需运行��次)
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
- **训练注意力模型 (MicroSegNet + CBAM)**:
  ```bash
  python -m src.train_attention
  ```
- **训练 U-Net 模型**:
  ```bash
  python -m src.train_unet
  ```
- **训练 TransUNet 模型 (推荐)**:
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
- GUI会自动识别并加载所有已支持的模型。
- 在界面中实时查看 **原始图像**、**真实掩码**、**模型A预测** 和 **模型B预测** 的四图对比，直观评估不同模型之间的差异。

---

## 项目进度

- **[✔️] 阶段一：基础框架搭建**
- **[✔️] 阶段二：���型优化与重构**
- **[✔️] 阶段三：集成先进架构 (U-Net, TransUNet)**
- **[✔️] 阶段四：Mamba架构探索 (已终止)**
  - [X] **深入探索**: 尝试将Mamba模块与U-Net架构融合，以探索其在分割任务中的潜力。
  - [X] **得出结论**: 经过系统性实验，我们发现Mamba的核心机制与U-Net的跳跃连接存在根本冲突，无法有效学习分割特征。
  - [X] **终止路线**: **Mamba-UNet技术路线已被正式终止**。详细的技术总结请参阅 [**Mamba探索总结报告 (docs/MAMBA_EXPLORATION_SUMMARY.md)**](./docs/MAMBA_EXPLORATION_SUMMARY.md)。

---

## 路线图

- **[▶️] 阶段五：深度训练与评估 (当前焦点)**
  - [ ] **全面训练**: 对所有模型，特别是 **TransUNet**，进行充分的训练（例如50-100个epoch）以获得最优权重。
  - [ ] **定量分析**: 编写脚本计算并对比不同模型在测试集上的 Dice Score, IoU, Precision, Recall 等关键指标。
  - [ ] **定性分析**: 使用GUI工具，进行可视化对比，分析不同模型在具体病例上的优劣。
  - [ ] **准备研究报告**: 整理所有实验结果，撰写详细的技术报告或论文。

---

## 技术栈

* **编程语言**: Python
* **深度学习框架**: **PyTorch**
* **核心模型**: MicroSegNet, U-Net, TransUNet, Attention U-Net
* **主要库**: OpenCV, Albumentations, Nibabel, NumPy, Timm

## 数据集

本项目使用的开源数据集可从 Zenodo 获取：[Micro-Ultrasound Prostate Segmentation Dataset](https://zenodo.org/records/10475293)。

## 许可证

本项目采用 [MIT](./LICENSE) 许可证。