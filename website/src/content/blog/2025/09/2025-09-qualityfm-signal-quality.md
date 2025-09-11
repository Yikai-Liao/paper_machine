---
title: "QualityFM: a Multimodal Physiological Signal Foundation Model with Self-Distillation for Signal Quality Challenges in Critically Ill Patients"
pubDatetime: 2025-09-08T10:20:56+00:00
slug: "2025-09-qualityfm-signal-quality"
type: "arxiv"
id: "2509.06516"
score: 0.7300757192501139
author: "grok-3-latest"
authors: ["Zongheng Guo", "Tao Chen", "Manuela Ferrario"]
tags: ["Foundation Model", "Signal Quality", "Self-Distillation", "Transformer", "Physiological Signals"]
institution: ["Politecnico di Milano", "Zhejiang University"]
description: "本文提出 QualityFM，一个通过自蒸馏和大规模预训练构建的多模态生理信号基础模型，显著提升了临床环境中信号质量挑战下的下游任务性能。"
---

> **Summary:** 本文提出 QualityFM，一个通过自蒸馏和大规模预训练构建的多模态生理信号基础模型，显著提升了临床环境中信号质量挑战下的下游任务性能。 

> **Keywords:** Foundation Model, Signal Quality, Self-Distillation, Transformer, Physiological Signals

**Authors:** Zongheng Guo, Tao Chen, Manuela Ferrario

**Institution(s):** Politecnico di Milano, Zhejiang University


## Problem Background

在重症监护室（ICU）和手术室（OR）等临床环境中，光电容积描记图（PPG）和心电图（ECG）信号常因患者移动、电极接触不良或仪器噪声而质量不佳，导致误报和诊断不准确。
现有方法受限于泛化能力不足、依赖大量标注数据以及跨任务迁移能力差，难以应对复杂的信号质量挑战。

## Method

*   **核心思想:** 构建一个多模态生理信号基础模型（QualityFM），通过自监督学习和预训练获得对信号质量的通用理解，支持多种下游任务。
*   **自蒸馏框架:** 采用双轨架构处理质量不同的信号对，利用高质量信号的编码器（教师模型）指导低质量信号的编码器（学生模型），通过交叉熵损失对齐两者的输出分布；教师模型参数通过指数移动平均（EMA）平滑更新，增强泛化能力并过滤噪声。
*   **窗口化稀疏注意力机制:** 基于 Transformer 架构，针对长序列信号设计窗口化注意力，降低计算复杂度（从 O(n²) 降至 O(n×w)，w 为窗口大小），同时在浅层捕捉局部形态特征，在深层学习长距离准周期性模式。
*   **复合损失函数:** 结合直接蒸馏损失（对齐编码器输出）和间接重建损失（通过解码器重建信号的功率谱和相位谱），确保保留信号的频域特性，增强特征提取的鲁棒性。
*   **预训练与迁移:** 在超过2100万段波形数据上进行预训练，构建通用信号表示，随后通过迁移学习适应不同临床任务。

## Experiment

*   **有效性:** QualityFM 在三个下游任务（假性心室性心动过速报警识别、房颤识别、动脉血压估计）上显著优于现有最先进方法；例如，在 VTaC 数据集上，QualityFM-Huge 模型准确率达 0.8551，F1 分数为 0.7607，远超其他方法。
*   **模型规模影响:** 随着参数从 9.6M 增加到 319M，性能持续提升，验证了基础模型的可扩展性。
*   **实验设置合理性:** 预训练数据量巨大（179,757 小时），下游任务涵盖分类和回归，评估指标全面（准确率、F1 分数、MAE 等）；消融实验验证了预训练、窗口化注意力（最佳窗口大小为 8）和复合损失函数的有效性。
*   **局限性:** 模型参数规模相对较小，计算资源需求较高，完整微调可能限制实际应用。

## Further Thoughts

自蒸馏策略可推广至其他领域，通过高质量数据指导低质量数据的特征提取；频域信息的关注提示我们在时间序列处理中应重视频域特征，尤其在噪声环境下；模型规模的性能提升启发我们探索更大规模的生理信号基础模型，甚至结合更多模态数据以提升鲁棒性。