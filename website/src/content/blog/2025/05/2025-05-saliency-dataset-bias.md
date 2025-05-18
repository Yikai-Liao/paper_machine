---
title: "Modeling Saliency Dataset Bias"
pubDatetime: 2025-05-15T10:55:47+00:00
slug: "2025-05-saliency-dataset-bias"
type: "arxiv"
id: "2505.10169"
score: 0.4158367844516192
author: "grok-3-latest"
authors: ["Matthias Kümmerer", "Harneet Khanuja", "Matthias Bethge"]
tags: ["Saliency Prediction", "Dataset Bias", "Generalization Gap", "Multiscale Features", "Visual Attention"]
institution: ["University of Tübingen", "Tübingen AI Center"]
description: "本文提出一种结合多尺度特征提取和数据集偏见参数适配的显著性预测模型，成功关闭跨数据集泛化差距并在多个基准上刷新最优性能，同时提供对显著性分布差异的深刻洞察。"
---

> **Summary:** 本文提出一种结合多尺度特征提取和数据集偏见参数适配的显著性预测模型，成功关闭跨数据集泛化差距并在多个基准上刷新最优性能，同时提供对显著性分布差异的深刻洞察。 

> **Keywords:** Saliency Prediction, Dataset Bias, Generalization Gap, Multiscale Features, Visual Attention

**Authors:** Matthias Kümmerer, Harneet Khanuja, Matthias Bethge

**Institution(s):** University of Tübingen, Tübingen AI Center


## Problem Background

图像显著性预测（Saliency Prediction）旨在预测人类在图像中的注视位置，尽管在现有基准（如 MIT300）上性能接近饱和，但跨数据集应用时性能下降显著（约 40%），表明存在严重的泛化差距（Generalization Gap）；
作者指出这一问题源于数据集特有的偏见（Dataset Bias），如中心偏见（Center Bias）、多尺度分布差异等，单纯增加训练数据多样性无法解决，需要针对性建模这些偏见以提升模型在新数据集上的表现。

## Method

* **核心思想：** 提出一种新型显著性预测模型，通过多尺度特征提取和少量可解释的数据集偏见参数（Dataset Bias Parameters）分离通用特征与数据集特有差异，从而提升跨数据集泛化能力。
* **多尺度特征提取：** 将输入图像缩放到多种分辨率，包括绝对尺度（以像素/度为单位，反映视觉角度大小）和相对尺度（以总像素为单位，反映图像比例），使用预训练的 CLIP 和 DINOv2 编码器分别提取全局和局部特征，随后将多尺度特征平均合并。
* **解码器设计：** 采用轻量级的读取网络（Readout Network），通过五层 1x1 卷积将特征图转换为显著性优先级图（Priority Map），参数少至可在小数据集上训练。
* **数据集偏见参数：** 引入少于 20 个可解释参数，控制多尺度权重（Multiscale Weights，调节绝对与相对尺度的贡献）、优先级缩放（Priority Scaling，反映显著性强度的相对差异）、模糊大小（Blur Size，控制注视分布的扩散程度）、中心偏见（Center Bias，建模注视中心倾向）及其权重（Center Bias Weight，平衡图像内容与先验偏见的贡献）。
* **泛化与适配策略：** 在泛化设置中，使用训练数据集的平均偏见参数应用到新数据集；在适配设置中，仅微调这些偏见参数以适应新数据集，显著提升性能且数据高效。
* **优势：** 方法在保持模型核心架构不变的情况下，通过少量参数调整即可适配新环境，避免了大规模重新训练的开销，同时参数的可解释性提供了对数据集差异的洞察。

## Experiment

* **泛化差距验证：** 实验在五个显著性数据集（MIT1003、CAT2000、COCO-Freeview、DAEMONS、FIGRIM）上进行，确认跨数据集性能下降显著（Inter-Dataset Gap 超 40%），即使在多个数据集联合训练，58% 的泛化差距（Generalization Gap）仍未解决。
* **适配效果：** 通过适配数据集偏见参数，关闭了约 76% 的泛化差距，且仅需 50 张图像即可接近完全适配性能，显示出极高的数据效率。
* **性能提升：** 在 MIT/Tuebingen Saliency Benchmark（MIT300、CAT2000、COCO-Freeview）上，模型在泛化、适配和完全训练三种设置下均刷新最优性能（State-of-the-Art），AUC 指标提升至少 1%，如 MIT300 上 AUC 从 0.883（DeepGaze IIE）提升至 0.894。
* **消融分析：** 多尺度架构、CLIP+DINOv2 编码器和数据集偏见参数均对性能有贡献，其中中心偏见和多尺度权重对泛化差距的关闭贡献最大，但不同数据集对具体偏见的依赖性不同。
* **实验设置合理性：** 实验涵盖多种训练场景（单个数据集、联合训练、留一法测试）和评估指标（Information Gain、AUC 等），并扩展到其他数据集（如 Kienzle、Toronto、SALICON）验证泛化能力，设置全面且数据支持结论显著。

## Further Thoughts

数据集偏见的显式建模思路值得关注，通过少量可解释参数捕捉跨数据集差异，不仅适用于显著性预测，还可能推广至其他视觉任务（如目标检测）以解决跨域泛化问题；
多尺度特征提取中绝对与相对尺度的分离对处理视觉大小感知的任务（如场景理解）有启发；
数据高效适配的特性提示未来可结合元学习（Meta-Learning）或少样本学习（Few-Shot Learning），以系统预测新数据集偏见参数，甚至实现无样本适配。