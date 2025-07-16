好的，这是您研究报告中引用的参考文献摘要。我已将它们整理成一个列表，并尽可能提供了详细信息和来源链接。

### 参考文献摘要

#### 核心架构与模型

1. **Ma, J., Li, F., & Wang, B. (2024).  *U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation* .**
   * **摘要** : 这篇论文提出了U-Mamba，一个用于生物医学图像分割的通用网络。它设计了一个混合CNN-SSM（状态空间模型）模块，结合了CNN的局部特征提取能力和Mamba在处理长程依赖关系上的优势。该模型在多个2D和3D分割任务中表现优于现有的CNN和Transformer模型 ^1^。
   * **来源** : [arXiv:2401.04722](https://arxiv.org/abs/2401.04722) ^2^,
     [GitHub](https://github.com/bowang-lab/U-Mamba) ^4^。
2. **Liu, Z., et al. (2024).  *VMamba: Visual State Space Model* .**
   * **摘要** : 该研究将Mamba架构成功应用于通用视觉任务，提出了Vision Mamba (VMamba)及其核心模块二维选择性扫描(SS2D)。SS2D通过多方向扫描，解决了将一维序列模型应用于二维图像的挑战，在保持线性复杂度的同时实现了全局感受野 ^5^。
   * **来源** : [arXiv:2401.10166](https://arxiv.org/abs/2401.10166) ^5^,
     [GitHub](https://github.com/MzeroMiko/VMamba) ^5^。
3. **Liu, J., et al. (2024).  *VM-UNet: Vision Mamba UNet for Medical Image Segmentation* .**
   * **摘要** : 这项工作首次探索了完全基于Mamba（纯SSM）构建U-Net形态的分割模型。它使用视觉状态空间块(VSS Block)作为编码器和解码器的基础构建单元，验证了纯SSM架构在医学图像分割任务中的潜力 ^7^。
   * **来源** : [arXiv:2402.02491](https://arxiv.org/abs/2402.02491) ^7^。
4. **Liu, J., et al. (2024).  *Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining* .**
   * **摘要** : 该论文探讨了预训练对Mamba分割模型的积极影响。通过使用在ImageNet上预训练的VMamba权重来初始化U-Net的编码器，Swin-UMamba在多个医学分割数据集上取得了显著的性能提升，凸显了预训练对于数据高效学习的重要性 ^9^。
   * **来源** : [GitHub](https://github.com/openmedlab/Swin-UMamba) ^9^。
5. **Isensee, F., et al. (2021).  *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation* .**
   * **摘要** : nnU-Net是一个能为任何新的医学分割任务自动配置完整流程（包括预处理、网络训练、推断和后处理）的框架。由于其强大的鲁棒性和性能，它已成为该领域的黄金标准，U-Mamba等多个先进模型均基于此框架构建 ^2^。

#### 关键技术与方法

6. **Gu, A., & Dao, T. (2023).  *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* .**
   * **摘要** : Mamba的开创性论文。它引入了选择性扫描机制（S6），使状态空间模型（SSM）的参数变为数据依赖型，从而能够像注意力机制一样过滤信息，同时保持线性计算复杂度，解决了传统SSM和Transformer的局限性 ^11^。
7. **Salehi, S. S. M., et al. (2017).  *Tversky loss function for image segmentation using 3D fully convolutional deep networks* .**
   * **摘要** : 该研究提出了Tversky损失函数，它是Dice损失的推广。通过引入两个参数 `$\alpha$`和 `$\beta$`，它可以灵活地调整对假阳性（FP）和假阴性（FN）的惩罚权重，这对于需要权衡精确率和召回率的医学分割任务（如前列腺分割）尤其有效 ^13^。
8. **Shiri, I., et al. (2022).  *The Effect of Loss Function on the Performance of a Deep Learning Model for Prostate Gland Segmentation in T2-Weighted MRI* .**
   * **摘要** : 这篇综述性研究比较了多种损失函数在前列腺分割任务中的表现。研究发现，复合损失函数（如BCE+Dice）和高级损失函数（如Focal Tversky Loss）通常优于单一损失函数，因为它们能更好地处理类别不平衡和边界模糊等问题 ^14^。
9. **Gildenblat, J. (2021).  *pytorch-grad-cam* .**
   * **摘要** : 一个广泛使用的PyTorch库，实现了多种可解释性AI算法（如Grad-CAM, Grad-CAM++等）。它支持分类、检测和分割等多种任务，能够生成热力图来可视化模型在做决策时关注的图像区域，是模型诊断和调试的重要工具 ^15^。
   * **来源** : [GitHub](https://github.com/jacobgil/pytorch-grad-cam) ^16^。

#### 数据集

10. **Micro-Ultrasound Prostate Segmentation Dataset (2024).**
    * **摘要** : 本研究使用的公开数据集，包含75位患者的微超声扫描图像和前列腺分割标注。数据以NIFTI格式提供，并分为训练集（55例）和测试集（20例），同时提供了来自不同经验水平标注者的多个标签，其中专家标注被视为“金标准” ^17^。
    * **来源** : [Zenodo](https://zenodo.org/records/10475293) ^17^。
