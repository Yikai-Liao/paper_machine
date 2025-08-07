---
title: "Amber Pruner: Leveraging N:M Activation Sparsity for Efficient Prefill in Large Language Models"
pubDatetime: 2025-08-04T07:22:36+00:00
slug: "2025-08-amber-pruner-sparsity"
type: "arxiv"
id: "2508.02128"
score: 0.706972541587996
author: "grok-3-latest"
authors: ["Tai An", "Ruwu Cai", "Yanzhe Zhang", "Yang Liu", "Hao Chen", "Pengcheng Xie", "Sheng Chang", "Yiwu Yao", "Gongyi Wang"]
tags: ["LLM", "Activation Sparsity", "N:M Sparsity", "Inference Efficiency", "Quantization"]
institution: ["Huawei Technologies Co., Ltd"]
description: "本文提出 Amber Pruner，一种无需训练的 N:M 激活稀疏化方法，显著提升大型语言模型预填充阶段的推理效率，同时保持模型性能。"
---

> **Summary:** 本文提出 Amber Pruner，一种无需训练的 N:M 激活稀疏化方法，显著提升大型语言模型预填充阶段的推理效率，同时保持模型性能。 

> **Keywords:** LLM, Activation Sparsity, N:M Sparsity, Inference Efficiency, Quantization

**Authors:** Tai An, Ruwu Cai, Yanzhe Zhang, Yang Liu, Hao Chen, Pengcheng Xie, Sheng Chang, Yiwu Yao, Gongyi Wang

**Institution(s):** Huawei Technologies Co., Ltd


## Problem Background

大型语言模型（LLM）在推理过程中，特别是在预填充（prefill）阶段，面临巨大的计算开销。
传统的权重稀疏化方法（如 SparseGPT）虽然能压缩模型，但往往导致显著的精度下降，尤其在复杂任务上。
激活稀疏化（Activation Sparsity）被认为是一种有潜力的替代方案，但现有方法多依赖训练过程，泛化性不足。
本文旨在开发一种无需训练的 N:M 激活稀疏化方法，针对预填充阶段的线性投影层，以在保持模型性能的同时加速推理。

## Method

*   **核心思想:** Amber Pruner 是一种无需训练的 N:M 激活稀疏化算法，专门针对 LLM 预填充阶段的线性投影层，通过结构化稀疏化减少计算开销，同时尽量保留模型性能。
*   **具体实现:**
    *   **Top-k 选择策略:** 在每 M 个连续激活元素中，基于幅度选择最重要的 N 个元素，构建 N:M 稀疏模式（如 2:4, 4:8, 8:16），从而压缩输入激活张量。
    *   **Robust-Norm Scoring 机制:** 不仅仅依赖激活值幅度，而是结合权重矩阵的统计信息（L2 范数）计算每个激活元素的重要性分数，公式为 S_ij = |X_ij| * f(W_:,j)，其中 f(W_:,j) 是标准化后的权重范数，避免简单剪枝导致关键输出丢失；同时通过异常值移除（剔除 0.5%-99.5% 以外的权重）和数值标准化增强分数稳定性。
    *   **Layer Skipping Strategy:** 通过敏感性分析（基于相对扰动误差 e_q），识别对稀疏化敏感的层（如 o_proj 和 up_proj），跳过这些层的稀疏化处理，以减少精度损失；例如，down_proj 通常敏感性最低，可全面剪枝，而 q_proj 和 gate_proj 则根据层级选择性剪枝。
*   **扩展框架 - Outstanding-sparse:** 将 Amber Pruner 与后训练量化（W8A8 quantization）结合，通过调整 SmoothQuant 的 scaling factor（反向扩展激活范围），增强激活分布的稀疏性，进一步提升推理效率。
*   **关键特点:** 不需要重新训练或微调模型，稀疏化仅在推理时应用于预填充阶段，与硬件友好的 N:M 模式兼容，同时通过模块化策略平衡效率与精度。

## Experiment

*   **有效性:** Amber Pruner 在多个模型（如 LLaMA3.1-8B, Qwen2-7B, Qwen3-30B-A3B）上测试，8:16 稀疏模式下加速了超过 55% 的线性投影计算，零样本任务平均精度损失小于 1%，生成任务（如 GSM8K）性能几乎不受影响。
*   **对比优势:** 相比简单的 Naïve top-k 稀疏方法，Amber Pruner 显著减少精度下降，例如在 LLaMA3.1-8B 上，8:16 模式下 Naïve top-k 导致 5.4% 精度下降，而 Amber Pruner（全功能）仅为 0.7%；Robust-Norm Scoring 和 Layer Skipping 的结合进一步优化了性能。
*   **实验设置合理性:** 实验覆盖了 Dense 和 MoE 两种模型架构，测试了多种稀疏比例（2:4, 4:8, 8:16），任务类型包括零样本（MMLU, BoolQ 等）、少样本（GSM8K）和长上下文理解（LongBench），数据集选择具有代表性；Outstanding-sparse 框架验证了与量化的兼容性。
*   **局限性与潜力:** 由于当前通用硬件对细粒度稀疏支持有限（如 SpMM 加速），实际加速效果未完全体现，但实验结果为未来软硬件协同优化提供了数据支持和方向。

## Further Thoughts

激活稀疏化在 LLM 推理优化中的潜力值得进一步探索，尤其是在预填充阶段激活值分布特性（更多接近零元素和异常值）的基础上，可以开发更动态的稀疏化策略，如基于上下文的自适应稀疏比例；
Layer Skipping Strategy 的敏感性分析方法提供了一种通用的模块化评估框架，未来可应用于其他压缩技术（如量化或低秩分解）；
Outstanding-sparse 框架启发我们可以在激活分布调整上做更多尝试，例如结合实时推理需求动态优化剪枝和量化的协同效应。