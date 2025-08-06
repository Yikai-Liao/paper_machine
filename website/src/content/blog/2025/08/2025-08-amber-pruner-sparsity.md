---
title: "Amber Pruner: Leveraging N:M Activation Sparsity for Efficient Prefill in Large Language Models"
pubDatetime: 2025-08-04T07:22:36+00:00
slug: "2025-08-amber-pruner-sparsity"
type: "arxiv"
id: "2508.02128"
score: 0.706972541587996
author: "grok-3-latest"
authors: ["Tai An", "Ruwu Cai", "Yanzhe Zhang", "Yang Liu", "Hao Chen", "Pengcheng Xie", "Sheng Chang", "Yiwu Yao", "Gongyi Wang"]
tags: ["LLM", "Activation Sparsity", "Inference Efficiency", "Quantization", "Structured Sparsity"]
institution: ["Huawei Technologies Co., Ltd"]
description: "本文提出 Amber Pruner，一种无需训练的 N:M 激活稀疏化方法，通过在预填充阶段对线性投影层进行结构化稀疏化，显著提升大型语言模型推理效率，同时保持精度。"
---

> **Summary:** 本文提出 Amber Pruner，一种无需训练的 N:M 激活稀疏化方法，通过在预填充阶段对线性投影层进行结构化稀疏化，显著提升大型语言模型推理效率，同时保持精度。 

> **Keywords:** LLM, Activation Sparsity, Inference Efficiency, Quantization, Structured Sparsity

**Authors:** Tai An, Ruwu Cai, Yanzhe Zhang, Yang Liu, Hao Chen, Pengcheng Xie, Sheng Chang, Yiwu Yao, Gongyi Wang

**Institution(s):** Huawei Technologies Co., Ltd


## Problem Background

大型语言模型（LLM）在推理过程中面临显著的计算瓶颈，尤其是在预填充（prefill）阶段，而传统的权重稀疏化方法往往导致精度大幅下降；激活稀疏化虽有潜力，但现有方法多依赖训练且泛化性不足，难以在大规模多批次场景中发挥作用。
因此，作者提出了一种无需训练的 N:M 激活稀疏化方法，旨在通过对线性投影层的输入激活进行结构化稀疏化，加速预填充阶段的计算，同时尽量减少精度损失。

## Method

*   **核心思想:** Amber Pruner 是一种无需训练的 N:M 激活稀疏化方法，专注于预填充阶段的线性投影层，通过对输入激活进行结构化稀疏化减少计算量，同时保持模型性能。
*   **具体实现:**
    *   **Top-k 选择策略:** 基于激活值的幅度，选择最重要的 N 个元素（在每 M 个连续元素中）进行保留，形成 N:M 稀疏模式，以减少计算负担。
    *   **Robust-Norm Scoring 机制:** 为提升选择精度，结合权重矩阵的通道级 L2 范数对激活值进行加权评分，通过异常值移除（剔除 0.5%-99.5% 以外的值）、标准化和最小范数归一化，确保评分稳定性和数值安全性，避免关键输出丢失。
    *   **Layer Skipping 策略:** 通过敏感性分析（基于相对扰动误差），识别对稀疏化敏感的层（如 o_proj 和 up_proj），跳过这些层的稀疏化操作，仅对低敏感性层（如 down_proj）进行处理，以平衡效率和精度。
*   **扩展框架 Outstanding-sparse:** 将 Amber Pruner 与后训练量化（W8A8 quantization）结合，通过调整 SmoothQuant 的缩放因子（反向扩展激活范围），增强激活分布的稀疏性，进一步提升推理效率。
*   **关键特点:** 方法不依赖模型重新训练或微调，权重评分系数可离线预计算，存储开销极小（不到模型大小的 0.05%），且通过算子融合实现高效推理。

## Experiment

*   **有效性:** Amber Pruner 在 8:16 稀疏度下表现最佳，零样本任务（如 MMLU、CMMLU）平均精度损失小于 1%，生成任务（如 GSM8K）性能几乎不受影响，成功加速了超过 55% 的线性投影计算。
*   **优越性:** 相比简单的 top-k 稀疏化方法，Amber Pruner（结合 Robust-Norm Scoring 和 Layer Skipping）显著减少了精度下降，尤其在高稀疏度（如 2:4）下表现更优；Outstanding-sparse 框架进一步验证了与量化的兼容性，在 MoE 模型（如 Qwen3-30B-A3B）上也保持了鲁棒性。
*   **实验设置:** 实验覆盖了多种模型（LLaMA3.1-8B、Qwen2-7B、Qwen3-30B-A3B）和稀疏度（2:4、4:8、8:16），任务包括零样本、少样本和长上下文理解（数据集如 MMLU、GSM8K、LongBench），设置较为全面合理。
*   **局限性:** 由于当前通用硬件对细粒度稀疏化（如稀疏-稠密矩阵乘法 SpMM）的支持有限，实际加速效果未完全体现，仅在理论上验证了计算量减少。

## Further Thoughts

Amber Pruner 揭示了激活稀疏化与硬件优化的巨大潜力，启发我们未来可以通过算法与硬件协同设计（如定制化 SpMM 加速器）进一步释放效率；此外，Layer Skipping 策略基于敏感性分析的模块化优化思路，提示我们可以针对模型不同组件定制差异化压缩策略，而非一刀切；Outstanding-sparse 中激活分布与量化交互的作用也值得探索，或许可以通过动态调整中间表示（如激活值范围）来优化多种压缩技术的协同效果。