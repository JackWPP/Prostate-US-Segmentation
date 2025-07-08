# Prostate-US-Segmentation

## 项目概述

本项目旨在研究和开发基于深度学习的前列腺超声图像自动分割算法，以解决临床实践中前列腺轮廓提取的难题，为前列腺疾病的诊断和治疗提供更精确的影像学辅助工具。前列腺疾病（如前列腺增生、前列腺炎和前列腺癌）是男性常见的健康问题，其中前列腺癌是全球男性癌症发病率第二高的恶性肿瘤。经直肠超声（TRUS）是前列腺检查的首选影像学方法，其无创、实时、经济、便捷的优势使其在临床中应用广泛。准确分割TRUS图像中的前列腺轮廓对于前列腺体积测量、癌症风险评估、活检和治疗计划设计至关重要。本项目将探索并实现先进的深度学习模型，以期克服超声图像存在的噪声大、对比度低、边界模糊等挑战，提高分割的准确性和鲁棒性。

## 项目任务

本项目将围绕以下几个核心任务展开：

1.  [cite_start]**数据集理解与特性分析**：深入理解开源前列腺超声图像数据集（[https://zenodo.org/records/10475293](https://zenodo.org/records/10475293) [cite: 8]）及其图像属性。
2.  **数据预处理与增强**：
    * 实现超声图像的预处理流程，包括噪声抑制、对比度增强、归一化等。
    * 探索超声图像特有的数据增强策略，如旋转、翻转、缩放、弹性变换等。
3.  **前列腺分割深度学习模型设计与实现**：
    * [cite_start]实现经典的医学图像分割网络（如 U-Net [cite: 6][cite_start]、SegNet [cite: 6]）作为基线模型。
    * [cite_start]重点实现 [MicroSegNet](https://github.com/mirthAI/MicroSegNet) 网络模型，解决边界模糊、对比度低等问题 [cite: 7]。
    * [cite_start]研究并实现注意力机制、多尺度特征融合、深度监督等技术，以提高分割精度 [cite: 7]。
4.  **模型评估与性能分析**：
    * [cite_start]实现多种分割评价指标计算，包括 Dice 系数、Jaccard 指数、Hausdorff 距离、平均表面距离等 [cite: 7]。
    * [cite_start]设计对比实验，比较不同分割算法的性能 [cite: 7]。
5.  [cite_start]**结果图表分析与可视化**：对模型训练过程中的准确率和损失函数图 [cite: 12][cite_start]、混淆矩阵 [cite: 13][cite_start]、分类器性能比较 [cite: 15][cite_start]以及超声分割效果图 [cite: 17]进行分析和可视化。
6.  **撰写生产实习报告与PPT**。

## 技术栈

* **编程语言**：Python
* **深度学习框架**：待定 (通常为 TensorFlow / PyTorch)
* **主要模型**：U-Net, SegNet, MicroSegNet
* **辅助工具**：NumPy, OpenCV, Matplotlib, scikit-image 等

## 数据集

本项目使用的开源数据集可从以下链接获取：
[cite_start][https://zenodo.org/records/10475293](https://zenodo.org/records/10475293) [cite: 8]

## 参考资料

* **项目搜索**：
    * [cite_start][http://www.github.com/search](http://www.github.com/search) [cite: 8]
* **英文论文搜索**：
    * [cite_start]Elsevier ScienceDirect: [https://tsg.buct.edu.cn/main.htm](https://tsg.buct.edu.cn/main.htm) [cite: 8]
    * [cite_start]SpringerLink: [https://tsg.buct.edu.cn/main.htm](https://tsg.buct.edu.cn/main.htm) [cite: 8]
    * [cite_start]Google Scholar: [https://scholar.google.com](https://scholar.google.com) [cite: 9]
* **中文论文搜索**：
    * [cite_start]CNKI中国知网: [https://tsg.buct.edu.cn/main.htm](https://tsg.buct.edu.cn/main.htm) [cite: 9]
* **AI 编程工具**：
    * [cite_start]Claude Sonnet: [https://www.anthropic.com/claude/sonnet](https://www.anthropic.com/claude/sonnet) [cite: 9]
    * [cite_start]Gemini: [https://aistudio.google.com](https://aistudio.google.com) [cite: 9]

## 结果图表示例 (待补充)

本节将展示项目完成后的主要结果图表，包括：

* **深度学习分割流程图**：描述数据处理和模型训练的整体流程。
* **准确率和损失函数图**：展示模型训练过程中的收敛情况和性能变化。
* **混淆矩阵**：评估模型在不同类别上的分类性能。
* **分类器性能比较**：对比不同算法在准确率、AUC、灵敏度、特异性等指标上的表现。
* **超声分割效果**：直观展示模型对超声图像中前列腺的分割效果。

## 贡献

欢迎对本项目提出建议和贡献。如果您有任何问题或想法，请随时提交 Issue 或 Pull Request。

## 许可证

本项目采用 [您的许可证，例如 MIT 或 Apache 2.0]。

---

[cite_start]**创建日期**: 2025年6月30日 [cite: 5]
[cite_start]**更新日期**: 2025年7月6日 [cite: 5]
