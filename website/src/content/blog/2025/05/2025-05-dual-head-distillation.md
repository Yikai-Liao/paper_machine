---
title: "Simple Semi-supervised Knowledge Distillation from Vision-Language Models via $\mathbf{\texttt{D}}$ual-$\mathbf{\texttt{H}}$ead $\mathbf{\texttt{O}}$ptimization"
pubDatetime: 2025-05-12T15:39:51+00:00
slug: "2025-05-dual-head-distillation"
type: "arxiv"
id: "2505.07675"
score: 0.4442097720167462
author: "grok-3-latest"
authors: ["Seongjae Kang", "Dong Bok Lee", "Hyungjoon Jang", "Sung Ju Hwang"]
tags: ["LLM", "Proxy Model", "Distillation", "Sampling", "Reasoning", "Test Time Scaling", "Pre-Training", "Post-Training", "RLHF"]
institution: ["VUNO Inc.", "KAIST", "DeepAuto.ai"]
description: "本文提出双头优化（DHO）框架，通过分离监督和蒸馏信号的优化路径缓解梯度冲突，提升半监督场景下从视觉-语言模型到紧凑模型的知识蒸馏效果，并在 ImageNet 上取得最先进性能。"
---

> **Summary:** 本文提出双头优化（DHO）框架，通过分离监督和蒸馏信号的优化路径缓解梯度冲突，提升半监督场景下从视觉-语言模型到紧凑模型的知识蒸馏效果，并在 ImageNet 上取得最先进性能。 

> **Keywords:** LLM, Proxy Model, Distillation, Sampling, Reasoning, Test Time Scaling, Pre-Training, Post-Training, RLHF

**Authors:** Seongjae Kang, Dong Bok Lee, Hyungjoon Jang, Sung Ju Hwang

**Institution(s):** VUNO Inc., KAIST, DeepAuto.ai


## Problem Background

视觉-语言模型（VLMs）因其大规模参数和计算需求，在资源受限环境中部署困难，因此需要通过知识蒸馏（KD）将其能力转移到紧凑模型上。
然而，现有 KD 方法多采用多阶段训练，计算开销高，且在半监督场景下，标注数据有限，监督信号与蒸馏信号之间存在梯度冲突，导致特征学习效果不佳。
本文旨在设计一种简单高效的 KD 方法，缓解梯度冲突，提升学生模型性能。

## Method

*   **核心思想:** 提出双头优化（Dual-Head Optimization, DHO）框架，通过两个独立的预测头分别优化监督信号（来自标注数据）和蒸馏信号（来自教师模型预测），避免梯度冲突，提升特征学习效果。
*   **具体实现:**
    *   学生模型包含一个共享特征提取器和两个预测头：h_CE 优化交叉熵损失以匹配标注数据，h_KD 优化 Kullback-Leibler 散度以匹配教师模型预测。
    *   训练时，两个头独立学习，特征提取器通过加权损失（λL_CE + (1-λ)L_KD）更新，λ 为平衡参数。
    *   推理时，通过线性组合两个头输出（p = α·σ(h_CE(z)) + (1-α)·σ(h_KD(z)/β)）生成最终预测，α 和 β 为可调参数，用于平衡两种信号的影响。
    *   额外优化：采用语言感知初始化，利用教师模型文本编码器初始化预测头权重；提出 KD 头对齐策略，鼓励 KD 头模仿教师模型的余弦相似度预测行为。
*   **关键优势:** 通过结构分离避免优化冲突，推理时灵活调整信号权重，无需重新训练即可适应不同任务需求。

## Experiment

*   **有效性:** DHO 在 11 个数据集上均优于单头 KD 基线，在 ImageNet 上以 1% 和 10% 标注数据分别提升准确率 3% 和 0.1%，达到最先进性能（SoTA）。
*   **提升显著性:** 通过 t-SNE 可视化和线性评估实验，DHO 的特征表示能力显著优于基线，验证了梯度冲突缓解对特征学习的重要性。
*   **实验设置合理性:** 实验覆盖多种模型架构（ResNet-18/50, MobileNetV2, ViT-B/16 等）和数据场景（1-16 shot, 1%-10% 标注数据），对比了自监督学习、CLIP 相关方法等多种基线，设置全面且具代表性。
*   **计算开销:** DHO 额外成本低，推理时参数和 FLOPs 增加微乎其微（例如 ResNet-18 参数增加 4.4%，FLOPs 几乎不变），适合实际部署。

## Further Thoughts

双头设计通过分离优化目标缓解梯度冲突的思路，可推广至多任务学习或多模态学习中分离不同目标优化路径；推理时线性插值调整信号权重的后训练调整机制，为模型适应性提供了新思路；语言感知初始化提示在多模态蒸馏中充分利用模态间预训练知识可能显著提升效果。