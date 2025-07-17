# 开发指南

本文档为“前列腺超声图像分割”项目提供了详细的开发环境设置与脚本运行指南。

## 1. 项目概述

本项目旨在为微超声图像中的前列腺分割开发一个深度学习模型。目前的实现采用了基于 **PyTorch** 的 **MicroSegNet** 架构。

## 2. 代码结构

项目遵循模块化结构，核心组件位于 `src/` 目录下。一个显著的特点是 `src/models_zoo/` 目录，其设计旨在独立管理多种模型架构。

- `src/models_zoo/base_model`: 包含原始的 `MicroSegNet` 实现。
- `src/models_zoo/attention_model`: 包含集成了CBAM注意力机制的 `MicroSegNetAttention` 模型。
- `src/hm_segnet.py`: 包含实验性的 `HMSegNet` 模型，该模型将Mamba模块集成到了 `MicroSegNet` 架构中。
- `src/gui_predictor.py`: 一个基于 Tkinter 的图形化评估工具。

## 3. 当前进度

### 阶段一与阶段二：基础工作
- 成功建立了 `MicroSegNet` 和 `MicroSegNetAttention` 基础模型及其训练流程。

### 阶段三：集成先进架构
- 成功将标准的 `U-Net` 和 `TransUNet` 模型集成到项目中，提供了一套强大的基��和先进架构。

### 阶段四：HM-SegNet 实现
- **正确解读**: 在多次尝试将Mamba与标准U-Net结合失败后，通过深入复审 `docs/mamba.md`，我们最终明确了正确的方案：将Mamba与 `MicroSegNet` 的类DenseNet结构进行融合。
- **成功实现**: 在 `src/hm_segnet.py` 中创建了新模型 `HMSegNet`。我们通过继承 `MicroSegNet` 基类，并将其编码器中 `_DenseBlock` 里的卷积层 `_DenseLayer` 替换为新的 `_MambaDenseLayer`，最终成功实现了该混合模型。此方法精确地遵循了技术文档，并保留了模型关键的跳跃连接逻辑。
- **建立训练流程**: 为新模型创建了专属的训练脚本 `src/train_hm_segnet.py`。

## 4. 如何运行项目

### 步骤 1: 运行数据预处理
此步骤只需执行一次。
```bash
python -m src.preprocess
```

### 步骤 2: 开始模型训练
您可以训练任何一个可用的模型。`TransUNet` 是推荐的、可用于生产的模型，而 `HMSegNet` 是推荐的实验性模型。

- **训练基础 MicroSegNet 模型:**
  ```bash
  python -m src.train
  ```
- **训练 TransUNet 模型:**
  ```bash
  python -m src.train_transunet
  ```
- **训练 HMSegNet 模型:**
  ```bash
  python -m src.train_hm_segnet
  ```

## 5. 后续任务

现在，项目的主要焦点是训练和评估仓库中现��且强大的模型。

*   **训练与评估:** 对 `TransUNet` 和 `HMSegNet` 进行一次完整的、长时间的训练（50-100个周期），以比较它们的性能。
*   **定量分析:** 编写脚本来计算并比较所有模型的关键指标（Dice、IoU等）。
*   **定性分析:** 使用GUI工具来直观对比结果，识别每个模型的优缺点。
