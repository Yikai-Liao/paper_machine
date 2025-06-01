---
title: "SC-LoRA: Balancing Efficient Fine-tuning and Knowledge Preservation via Subspace-Constrained LoRA"
pubDatetime: 2025-05-29T17:55:21+00:00
slug: "2025-05-sc-lora-finetuning"
type: "arxiv"
id: "2505.23724"
score: 0.7976505192676854
author: "grok-3-latest"
authors: ["Minrui Luo", "Fuhang Kuang", "Yu Wang", "Zirui Liu", "Tianxing He"]
tags: ["LLM", "Parameter Efficient Tuning", "Fine-Tuning", "Knowledge Preservation", "Subspace Constraint"]
institution: ["Institute for Interdisciplinary Information Sciences, Tsinghua University", "Shanghai Qi Zhi Institute", "Institute of Information Engineering, Chinese Academy of Sciences"]
description: "本文提出 SC-LoRA，通过子空间约束的 LoRA 初始化方法平衡高效微调和知识保留的双重目标，并在实验中显著优于现有方法。"
---

> **Summary:** 本文提出 SC-LoRA，通过子空间约束的 LoRA 初始化方法平衡高效微调和知识保留的双重目标，并在实验中显著优于现有方法。 

> **Keywords:** LLM, Parameter Efficient Tuning, Fine-Tuning, Knowledge Preservation, Subspace Constraint

**Authors:** Minrui Luo, Fuhang Kuang, Yu Wang, Zirui Liu, Tianxing He

**Institution(s):** Institute for Interdisciplinary Information Sciences, Tsinghua University, Shanghai Qi Zhi Institute, Institute of Information Engineering, Chinese Academy of Sciences


## Problem Background

大型语言模型（LLMs）在参数高效微调（PEFT）中面临两大挑战：传统低秩适应（LoRA）方法收敛速度慢，且微调过程可能导致灾难性遗忘（Catastrophic Forgetting），即丢失预训练知识（如世界知识或安全对齐）；现有方法往往只能解决其中一个问题，而无法同时兼顾高效微调和知识保留。

## Method

* **核心思想**：提出子空间约束的 LoRA 初始化框架（SC-LoRA），通过数据驱动的方式找到一个低秩子空间，使 LoRA 适配器的输出聚焦于微调数据的上下文信息，同时尽量避免干扰预训练知识的上下文信息。
* **具体实现**：
  * 使用微调任务数据（正任务）和需要保留的知识数据（负任务）分别计算线性层输出的协方差矩阵 Cov+ 和 Cov-。
  * 引入超参数 β 平衡两者的权重，构造目标矩阵 ΔCov = (1-β)Cov+ - βCov-，并对其进行特征值分解，提取前 r 个特征向量作为子空间的基向量。
  * 使用这些基向量初始化 LoRA 适配器矩阵 A 和 B，具体为 B_init = Q_r（基向量矩阵），A_init = Q_r^T * W_0（原始权重矩阵），从而约束适配器输出在目标子空间内。
  * 调整残差权重 W_res = W_0 - B_init * A_init，确保初始时模型输出与预训练一致。
* **理论支持**：通过子空间投影的数学分析，证明该初始化方法能最大化微调数据的上下文信息，同时最小化对保留知识的干扰。
* **优势**：相比传统 LoRA 的随机初始化或基于 SVD 的方法，SC-LoRA 更注重数据驱动和任务平衡，且不增加推理时计算开销。

## Experiment

* **有效性**：SC-LoRA 在微调性能上显著优于传统 LoRA 及其他基线方法（如 PiSSA, CorDA IPA），甚至在某些场景下超越全参数微调；例如，在 MetaMathQA 毒化数据集上，SC-LoRA (β=0.5) 的 GSM8k 准确率达 45.56%，比全参数微调高约 4 个百分点。
* **知识保留**：在安全性和世界知识保留方面表现优异，例如在良性数据微调中，SC-LoRA (β=0.9) 的危害评分接近原始模型（1.097 vs 1.100），而其他方法显著恶化。
* **实验设置合理性**：实验覆盖多种任务（对话摘要、数学推理）和数据集（Samsum, MetaMathQA, GSM8k 等），考虑了良性和毒化数据场景，设置全面；超参数 β 的调节展示了方法在效率和保留之间的权衡能力。
* **不足**：实验仅用单一随机种子，结果可能有波动；长期微调或复杂任务下的知识保留效果需进一步验证。

## Further Thoughts

SC-LoRA 的子空间约束思想具有广泛适用性，例如可扩展到多任务学习中为每个任务分配不同子空间以减少干扰，或在多模态模型中分离视觉和语言特征更新空间以避免知识丢失；此外，β 参数的线性调节效果提示是否可以通过自适应方法动态调整 β 值，根据微调过程中的性能和遗忘程度实时优化权衡。