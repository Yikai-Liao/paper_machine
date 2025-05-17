---
title: "Modeling Saliency Dataset Bias"
pubDatetime: 2025-05-15T10:55:47+00:00
slug: "2025-05-saliency-dataset-bias"
type: "arxiv"
id: "2505.10169"
score: 0.4158367844516192
author: "grok-3-latest"
authors: ["Matthias Kümmerer", "Harneet Khanuja", "Matthias Bethge"]
tags: ["Saliency Prediction", "Dataset Bias", "Generalization Gap", "Multiscale Architecture", "Feature Extraction"]
institution: ["Tübingen AI Center", "University of Tübingen"]
description: "本文提出一种显著性预测模型，通过少量可解释的数据集偏见参数显著提升跨数据集泛化能力，并在多个基准上创下新纪录。"
---

> **Summary:** 本文提出一种显著性预测模型，通过少量可解释的数据集偏见参数显著提升跨数据集泛化能力，并在多个基准上创下新纪录。 

> **Keywords:** Saliency Prediction, Dataset Bias, Generalization Gap, Multiscale Architecture, Feature Extraction

**Authors:** Matthias Kümmerer, Harneet Khanuja, Matthias Bethge

**Institution(s):** Tübingen AI Center, University of Tübingen


## Problem Background

图像显著性预测模型在现有基准（如 MIT300）上性能接近金标准，但跨数据集泛化时性能下降高达 40%，即使增加数据集多样性仍有约 60% 的差距源于数据集特有偏见；本文旨在解决这一泛化差距问题，提高模型对未见数据集的鲁棒性。

## Method

* **核心思想**：提出一种基于多尺度骨干架构的显著性预测模型，通过少于 20 个可解释的数据集特有参数，建模数据集偏见以提升泛化能力。
* **架构设计**：输入图像被缩放到多个分辨率（包括相对尺度和绝对视觉角度尺度），使用 CLIP 和 DINOv2 预训练编码器提取深层特征，跨尺度平均后通过一个轻量级解码器（五层 1x1 卷积的读出网络）生成空间优先级图。
* **数据集偏见参数**：包括多尺度权重（控制相对和绝对大小的影响）、优先级缩放（调整对象显著性差异）、模糊大小（控制注视点扩散）、中心偏见及其权重（建模注视中心倾向并允许调整），这些参数在泛化时取训练数据集平均值，在适应时针对新数据集微调。
* **训练与应用**：模型大部分参数（约 26,460 个）跨数据集联合训练，仅偏见参数为数据集特有；这种参数高效和数据高效的设计允许在少量样本（如 50 张图像）上快速适应新数据集。

## Experiment

* **有效性**：实验在五个显著性数据集（MIT1003、CAT2000、COCO-Freeview、DAEMONS、FIGRIM）上进行，确认跨数据集性能下降超 40%，通过适应偏见参数关闭了约 76% 的泛化差距。
* **性能提升**：在 MIT/Tuebingen 显著性基准（MIT300、CAT2000、COCO-Freeview）上，AUC 指标至少提升 1.1%-1.5%，在泛化、适应和全训练设置下均创下新纪录。
* **数据效率**：只需 5-10 张图像即可优于泛化性能，50 张图像接近完全适应性能，展现了方法的高效性。
* **实验设置**：覆盖多种训练场景（单个数据集、联合训练、留一法测试），与多个基线模型（DeepGaze IIE、UNISAL 等）对比，使用 IG 和 AUC 指标，设置全面合理。
* **局限性**：模型在低级模式和抽象图形预测上仍有不足，表明空间显著性问题未完全解决。

## Further Thoughts

通过少量可解释参数显式建模数据集偏见的思路非常启发性，不仅提升了泛化能力，还为理解数据集差异提供了洞察；这种方法可推广到其他领域，如自然语言处理中的偏见建模，或多模态任务中的跨模态一致性问题；此外，未来探索图像依赖的多尺度结构可能进一步提升模型自适应性，提示我们可以引入动态调整机制根据内容选择尺度权重。