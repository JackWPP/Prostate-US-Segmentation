# 开发指南

本文档为“前列腺超声图像分割”项目提供了详细的开发环境设置与脚本运行指南。

## 1. 项目概述

本项目旨在为微超声图像中的前列腺分割开发一个深度学习模型。目前的实现采用了基于 **PyTorch** 的 **MicroSegNet** 架构。

## 2. 代码结构

项目遵循模块化结构，核心组件位于 `src/` 目录下。一个显著的特点是 `src/models_zoo/` 目录，其设计旨在独立管理多种模型架构。

- `src/models_zoo/base_model`: 包含原始的 `MicroSegNet` 实现。
- `src/models_zoo/attention_model`: 包含集成了 CBAM 注意力机制的 `MicroSegNetAttention` 模型。
- `src/train.py`: 用于训练基础模型的脚本。
- `src/train_attention.py`: 用于训练注意力模型的脚本。
- `src/gui_predictor.py`: 一个基于 Tkinter 的图形化评估工具。

## 3. 当前进度

以下里程碑已经完成：

1.  **数据预处理 (`src/preprocess.py`):** 将原始数据处理为 `.npy` 切片。
2.  **模型实现 (`src/model.py`):** 实现了基础的 `MicroSegNet` 架构。
3.  **训练脚本 (`src/train.py`):** 为基础模型搭建了完整的训练流程。
4.  **验证脚本 (`src/verify_setup.py`):** 用于确保环境配置正确的测试脚本。
5.  **图形化预测工具 (`src/gui_predictor.py`):** 一个用于交互式预测和可视化的图形界面。
6.  **集成注意力机制:**
    *   通过 `models_zoo` 目录重构了项目结构以支持多模型管理。
    *   实现了独立的 CBAM (卷积块注意力模块)。
    *   创建了将 CBAM 集成到解码器跳跃连接中的新模型 `MicroSegNetAttention`。
    *   为新模型提供了独立的训练脚本 `train_attention.py`。

## 4. 新GPU服务器环境设置

请按照以下步骤在新设备上配置项目。

### 步骤 1: 克隆代码仓库
...
### 步骤 2: 设置Python环境
...
### 步骤 3: 安装依赖
...

## 5. 如何运行项目

在运行脚本前，请确保已激活虚拟环境。

### 步骤 1: 运行数据预处理
此步骤只需执行一次。
```bash
python src/preprocess.py
```

### 步骤 2: 验证设置 (可选)
此脚本会检查基础模型和数据加载流程。
```bash
python src/verify_setup.py
```

### 步骤 3: 开始模型训练
您可以选择训练基础模型或新的注意力增强模型。

- **训练基础 MicroSegNet 模型:**
  ```bash
  python src/train.py
  ```
  最佳模型将保存在 `models/best_microsegnet_model.pth`。

- **训练 MicroSegNetAttention 模型:**
  ```bash
  python src/train_attention.py
  ```
  最佳模型将保存在 `models/attention/best_microsegnet_attention_model.pth`。

## 6. 后续任务

正如 `docs/work.md` 中所述，项目的下一阶段将专注于：

*   **集成注意力机制**：帮助模型关注更相关的图像特征。
*   **实现多尺度特征融合**：更好地捕捉不同分辨率下的细节信息。
*   **添加深度监督**：改善深层网络的梯度流，辅助训练过程。
