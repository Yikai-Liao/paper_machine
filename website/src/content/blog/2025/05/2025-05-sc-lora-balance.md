---
title: "SC-LoRA: Balancing Efficient Fine-tuning and Knowledge Preservation via Subspace-Constrained LoRA"
pubDatetime: 2025-05-29T17:55:21+00:00
slug: "2025-05-sc-lora-balance"
type: "arxiv"
id: "2505.23724"
score: 0.7976505192676854
author: "grok-3-latest"
authors: ["Minrui Luo", "Fuhang Kuang", "Yu Wang", "Zirui Liu", "Tianxing He"]
tags: ["LLM", "Parameter Efficient Tuning", "LoRA", "Knowledge Preservation", "Fine-Tuning"]
institution: ["Institute for Interdisciplinary Information Sciences, Tsinghua University", "Shanghai Qi Zhi Institute", "Institute of Information Engineering, Chinese Academy of Sciences"]
description: "SC-LoRA 通过子空间约束的 LoRA 初始化方法，在高效微调与知识保留之间取得平衡，显著提升了大语言模型在下游任务上的性能并减少了预训练知识的遗忘。"
---

> **Summary:** SC-LoRA 通过子空间约束的 LoRA 初始化方法，在高效微调与知识保留之间取得平衡，显著提升了大语言模型在下游任务上的性能并减少了预训练知识的遗忘。 

> **Keywords:** LLM, Parameter Efficient Tuning, LoRA, Knowledge Preservation, Fine-Tuning

**Authors:** Minrui Luo, Fuhang Kuang, Yu Wang, Zirui Liu, Tianxing He

**Institution(s):** Institute for Interdisciplinary Information Sciences, Tsinghua University, Shanghai Qi Zhi Institute, Institute of Information Engineering, Chinese Academy of Sciences


## Problem Background

大语言模型（LLM）的参数高效微调（PEFT）方法，如低秩适应（LoRA），在减少训练参数的同时面临两大问题：微调收敛速度慢和知识遗忘（Catastrophic Forgetting），即可能破坏预训练模型中的世界知识或安全对齐特性。
以往方法只能解决其中一个问题，而无法同时兼顾高效微调与知识保留的平衡。

## Method

*   **核心思想:** 提出子空间约束的 LoRA（SC-LoRA），通过数据驱动的初始化方法，将 LoRA 适配器的输出约束在一个低秩子空间内，使其聚焦于微调数据的上下文信息，同时尽量避免干扰预训练知识。
*   **具体实现:** 
    *   **子空间选择:** 针对微调任务数据（目标任务）和保留知识数据（预训练知识），分别计算模型线性层输出的协方差矩阵 Cov+ 和 Cov-，通过特征值分解找到一个低秩子空间，该子空间最大化微调数据的投影信息，同时最小化保留知识的投影信息。
    *   **初始化策略:** 使用子空间的主方向（前 r 个特征向量）初始化 LoRA 适配器的权重矩阵 B 和 A，其中 B 直接设为特征向量矩阵，A 设为特征向量矩阵与原始权重矩阵的乘积，确保适配器输出始终在目标子空间内。
    *   **平衡参数 β:** 引入超参数 β（范围 0 到 1）调节微调性能与知识保留的权衡，β 越大，对保留知识的保护越强。
*   **优势:** 不改变模型架构，仅通过初始化约束即可实现目标，且提供了理论分析支持子空间选择的合理性。

## Experiment

*   **有效性:** SC-LoRA 在多个任务上表现出色，例如在良性数据（Samsum）微调中，ROUGE-1 分数达到 52.54（β=0.5），优于全微调（51.41）和 Vanilla LoRA（50.32），同时有害性评分（1.161）接近原始模型（1.100）；在毒化数据（MetaMathQA）中，数学任务准确率达 45.26（β=0.9），优于全微调（41.47），安全指标也接近原始模型。
*   **知识保留:** 在世界知识保留实验中，SC-LoRA (β=0.8) 的平均知识分数（22.73）优于大多数基线，同时数学任务性能（30.04）也保持领先。
*   **实验设置合理性:** 实验覆盖了总结、数学等多种下游任务，评估了安全和世界知识两种保留目标，并通过调整 β 值验证了权衡效果，数据点分布广泛且对比充分。
*   **局限性:** 作为初始化方法，SC-LoRA 未对微调过程更新进行强约束，长期微调可能仍导致知识遗忘；样本稀疏时存在数值不稳定性。

## Further Thoughts

SC-LoRA 的子空间约束思想启发了对任务隔离的进一步探索：是否可以通过动态子空间选择（如在线更新）或自适应 β 参数调节（基于微调反馈）进一步减少知识遗忘？此外，该方法是否可扩展至多模态模型，平衡不同模态知识与任务性能的权衡？