---
title: "LASPA: Language Agnostic Speaker Disentanglement with Prefix-Tuned Cross-Attention"
pubDatetime: 2025-06-02T10:59:31+00:00
slug: "2025-06-language-agnostic-speaker"
type: "arxiv"
id: "2506.02083"
score: 0.7151347031459008
author: "grok-3-latest"
authors: ["Aditya Srinivas Menon", "Raj Prakash Gohil", "Kumud Tripathi", "Pankaj Wasnik"]
tags: ["Speaker Recognition", "Multilingual", "Disentanglement", "Cross-Attention", "Prefix Tuning"]
institution: ["Sony Research India"]
description: "本文提出 LASPA 框架，通过前缀调优的交叉注意力机制有效解耦说话人身份和语言特征，显著提升了多语言说话人识别的性能和泛化能力。"
---

> **Summary:** 本文提出 LASPA 框架，通过前缀调优的交叉注意力机制有效解耦说话人身份和语言特征，显著提升了多语言说话人识别的性能和泛化能力。 

> **Keywords:** Speaker Recognition, Multilingual, Disentanglement, Cross-Attention, Prefix Tuning

**Authors:** Aditya Srinivas Menon, Raj Prakash Gohil, Kumud Tripathi, Pankaj Wasnik

**Institution(s):** Sony Research India


## Problem Background

在多语言场景下，说话人识别系统面临语言特征与说话人身份特征纠缠的问题，导致跨语言识别准确率下降。
传统模型难以区分由语言引起的语音变异和说话人特有的声学特征，尤其当说话人在不同语言间切换时，系统可能将同一说话人误判为不同个体。

## Method

*   **核心思想:** 提出 LASPA 框架，通过前缀调优的交叉注意力机制（Prefix-Tuned Cross-Attention）解耦说话人身份特征和语言相关特征，实现语言无关的说话人嵌入。
*   **具体实现:**
    *   **双编码器结构:** 使用 Speaker Encoder 和 Language Encoder 分别从输入 mel-spectrogram 中提取说话人嵌入和语言嵌入，确保初始特征分离。
    *   **前缀调优器（Prefix-Tuners）:** 引入两个交叉注意力模块（Speaker-Language 和 Language-Speaker），通过前缀向量增强特征交互，而不直接修改模型权重，保持高效性。
    *   **解码器重建:** 将融合后的嵌入输入解码器，重建输入 mel-spectrogram，确保特征提取的有效性和一致性。
    *   **多任务损失函数:** 训练时结合多种损失，包括 AAM Softmax（说话人分类损失）、NLL（语言分类损失）、MAPC（鼓励特征解耦）和 MSE（重建损失），以平衡解耦效果和识别性能。
*   **创新点:** 相比传统基于梯度反转层（GRL）的方法，LASPA 避免了训练不稳定和超参数敏感问题，同时前缀调优仅占总参数的 1.16%，计算开销低。

## Experiment

*   **有效性:** LASPA 在多个数据集上显著提升了说话人识别性能，例如在多语言数据集 VoxCeleb1-B 上，LASPA ReDimNet 的 EER 降至 1.62%（基线为 1.66%），在未见语言数据集 NISP-B 上 EER 降至 10.22%（基线为 11.90%）。
*   **优越性:** 相比基线模型（如 ResNet, ECAPA）和 GRL 方法，LASPA 一致性地降低了 EER 和 minDCF，尤其在跨语言场景中表现出更强的鲁棒性和泛化能力。
*   **实验设置:** 实验覆盖了单语言（VoxCeleb1 系列）、多语言（VoxSRC 2021）和未见语言（NISP-B）场景，使用多种骨干网络（ResNet-S/L, ECAPA, ReDimNet）进行验证，设置全面合理。
*   **局限性:** 论文未详细讨论计算开销和推理时间，前缀调优和交叉注意力可能对实时应用造成一定影响。

## Further Thoughts

前缀调优作为一种轻量级适配方法，可以扩展到其他多模态任务中，如语音情感识别或图像风格解耦，通过调整特征交互实现更通用的信息分离；此外，动态调整前缀参数以适应不同输入特性（如语言或说话人风格）可能进一步提升模型鲁棒性。