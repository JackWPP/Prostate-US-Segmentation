# 开发指南

本文档为“前列腺超声图像分割”项目提供了详细的开发环境设置与脚本运行指南。

## 1. 项目概述

本项目旨在为微超声图像中的前列腺分割开发一个深度学习模型。目前的实现采用了基于 **PyTorch** 的 **MicroSegNet** 架构。

## 2. 当前进度

以下里程碑已经完成：

1.  **数据预处理 (`src/preprocess.py`):**
    *   加载原始的 NIFTI (`.nii.gz`) 文件。
    *   将 3D 扫描数据处理成 2D 切片。
    *   将图像和掩码的尺寸统一调整为 256x256。
    *   将图像像素值归一化到 [0, 1] 范围。
    *   对训练集应用数据增强（如翻转、旋转等）以提升模型鲁棒性。
    *   将处理好的数据以 `.npy` 格式保存在 `processed_data/` 目录下。

2.  **模型实现 (`src/model.py`):**
    *   基于官方论文和代码库，在 PyTorch 中实现了 **MicroSegNet** 架构。
    *   模型遵循类似 U-Net 的编码器-解码器结构，并在编码器中使用了 DenseNet 模块来增强特征传播。
    *   修正了原始实现中的通道和张量维度不匹配问题，确保训练能够顺利进行。

3.  **训练脚本 (`src/train.py`):**
    *   搭建了完整的模型训练流程。
    *   包含一个自定义的 `ProstateDataset` 类，用于加载预处理后的 `.npy` 数据。
    *   使用 Dice Loss 作为损失函数，这是图像分割任务中的常用选择。
    *   训练循环会在每个周期（epoch）结束后，在测试集上评估模型，并自动保存 Dice 分数最高的最佳模型。

4.  **验证脚本 (`src/verify_setup.py`):**
    *   一个用于快速测试的脚本，它会运行一次完整的训练迭代（前向传播、计算损失、反向传播），以确认环境、数据加载、模型和训练逻辑均已正确配置。

5.  **图形化预测工具 (`src/gui_predictor.py`):**
    *   一个使用 Tkinter 构建的图形用户界面，用于进行交互式预测。
    *   用户可以从列表中选择任意测试图片，并即时查看原始图像、真实掩码叠加图以及模型预测的掩码叠加图，方便进行效果对比。

## 3. 新GPU服务器环境设置

请按照以下步骤在新设备（如GPU服务器）上配置项目。

### 步骤 1: 克隆代码仓库

```bash
git clone <your-repository-url>
cd Prostate-US-Segmentation
```

### 步骤 2: 设置Python环境

强烈建议使用虚拟环境，以避免与系统级的包产生冲突。

```bash
# 创建一个名为 'venv' 的虚拟环境
python3 -m venv venv

# 激活环境
# 在 Linux/macOS 上:
source venv/bin/activate
# 在 Windows 上:
venc\\Scripts\\activate
```

### 步骤 3: 安装依赖

所有必需的Python包都已在 `requirements.txt` 中列出。指定的PyTorch版本适用于CUDA，请确保您已安装兼容的NVIDIA GPU和驱动程序。

```bash
# 安装所有必需的包
pip install -r requirements.txt
```

## 4. 如何运行项目

在运行脚本前，请确保已激活虚拟环境 (`source venv/bin/activate`)。

### 步骤 1: 运行数据预处理

此步骤只需执行一次。该脚本会处理 `dataset/` 目录下的原始数据，并创建 `processed_data/` 目录。

```bash
python src/preprocess.py
```

### 步骤 2: 验证设置 (可选，但强烈推荐)

在开始长时间的训练之前，运行此验证脚本，以确保一切工作正常。

```bash
python src/verify_setup.py
```
如果脚本最后输出 `[SUCCESS] All tests passed`，则说明您可以准备开始训练了。

### 步骤 3: 开始模型训练

此脚本将训练MicroSegNet模型。训练进度会显示在控制台中。表现最佳的模型将被自动保存在 `models/` 目录下。

```bash
python src/train.py
```

## 5. 后续任务

正如 `docs/work.md` 中所述，项目的下一阶段将专注于：

*   **集成注意力机制**：帮助模型关注更相关的图像特征。
*   **实现多尺度特征融合**：更好地捕捉不同分辨率下的细节信息。
*   **添加深度监督**：改善深层网络的梯度流，辅助训练过程。
