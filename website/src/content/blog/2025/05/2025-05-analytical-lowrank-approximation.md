---
title: "A3 : an Analytical Low-Rank Approximation Framework for Attention"
pubDatetime: 2025-05-19T10:29:32+00:00
slug: "2025-05-analytical-lowrank-approximation"
type: "arxiv"
id: "2505.12942"
score: 0.7580513780260353
author: "grok-3-latest"
authors: ["Jeffrey T. H. Wong", "Cheng Zhang", "Xinye Cao", "Pedro Gimenes", "George A. Constantinides", "Wayne Luk", "Yiren Zhao"]
tags: ["LLM", "Low-Rank Approximation", "Transformer Compression", "Attention Mechanism", "Model Efficiency"]
institution: ["Imperial College London"]
description: "本文提出 `A[3]` 框架，通过对 Transformer 的 `QK`、`OV` 和 `MLP` 组件进行分析性低秩近似，显著降低模型大小、KV 缓存和计算量，同时避免运行时开销，并在性能上大幅优于现有方法。"
---

> **Summary:** 本文提出 `A[3]` 框架，通过对 Transformer 的 `QK`、`OV` 和 `MLP` 组件进行分析性低秩近似，显著降低模型大小、KV 缓存和计算量，同时避免运行时开销，并在性能上大幅优于现有方法。 

> **Keywords:** LLM, Low-Rank Approximation, Transformer Compression, Attention Mechanism, Model Efficiency

**Authors:** Jeffrey T. H. Wong, Cheng Zhang, Xinye Cao, Pedro Gimenes, George A. Constantinides, Wayne Luk, Yiren Zhao

**Institution(s):** Imperial College London


## Problem Background

大型语言模型（LLMs）因其巨大的参数量和计算复杂度导致高昂的部署成本，尤其是在资源受限的环境中。
现有的低秩近似方法通常针对一般线性层，忽视了 Transformer 架构的特性（如注意力机制的多头结构），且通过矩阵分解引入了额外的运行时开销（如 GEMM 操作），在性能上往往不如剪枝或量化等其他压缩技术。

## Method

*   **核心思想:** 提出 `A[3]` 框架，将 Transformer 层分解为三个功能组件（`QK`、`OV`、`MLP`），通过分析性低秩近似方法针对每个组件最小化其功能误差，减少内部隐藏维度，从而降低模型大小、KV 缓存和计算量（FLOPs），同时避免运行时开销。
*   **具体实现:**
    *   **QK 组件（Query-Key）:** 目标是最小化预softmax注意力分数误差。基于输入的自相关矩阵，使用截断奇异值分解（SVD）对融合的查询-键权重矩阵进行低秩近似，减少头维度（`d_qk`）。
    *   **OV 组件（Output-Value）:** 目标是最小化每头注意力输出误差。同样基于输入自相关矩阵，通过 SVD 对值-输出权重矩阵进行低秩近似，减少头维度（`d_vo`）。
    *   **MLP 组件（Multi-Layer Perceptron）:** 由于非线性激活函数的限制，采用 CUR 分解方法，通过评估中间激活和权重的外积范数，选择最重要的中间维度（`d_inter`），从而减少参数量。
    *   **适配性扩展:** 为支持现代 Transformer 变体，`A[3]` 针对分组查询注意力（GQA）设计了联合 SVD 策略，确保共享键/值头的压缩一致性；针对旋转位置编码（RoPE），通过 CUR 近似选择重要频率对，保持位置信息。
*   **关键优势:** 直接减少隐藏维度而非将大矩阵分解为多个小矩阵，避免了额外 GEMM 操作和内存操作，确保无运行时开销，同时对 Transformer 架构特性进行了深度优化。

## Experiment

*   **有效性:** `A[3]` 在多个模型和任务上显著优于现有低秩近似方法。例如，在 LLaMA 3.1-70B 模型（10% 压缩比例）上，WikiText-2 困惑度从 SVD-LLM 的 7.87 降至 4.69（相对改进 58.6%）；下游任务平均准确率也一致优于基线（如 LLaMA 2-7B 上从 0.4166 提升至 0.4960）。
*   **硬件效率:** 由于避免了额外 GEMM 操作，`A[3]` 在推理吞吐量（TPS）上表现优异，例如在 LLaMA 2-13B 上，`A[3]` 在所有压缩比例下均实现加速，而 SVD-LLM 仅在高压缩比例下有效。
*   **实验设置合理性:** 实验覆盖了多种 Transformer 架构（MHA、GQA、RoPE），不同压缩比例（10%-60%），以及预训练任务（WikiText-2、C4）和下游任务（GSM8K、MMLU），与多种基线方法（如 SVD-LLM、FWSVD）对比，验证了方法的广泛适用性和鲁棒性。
*   **消融研究:** 针对各组件（`QK`、`OV`、`MLP`）的单独评估表明，`A[3]` 在每个组件上均优于对应基线，尤其在 RoPE 和 GQA 适配上表现突出。

## Further Thoughts

1. `A[3]` 将复杂模型分解为功能组件并分别优化的思路启发我们可以在其他深度学习架构中应用类似策略，针对不同模块设计定制化压缩或优化方法，提升整体效率。
2. 通过减少隐藏维度而非矩阵分解来避免运行时开销的设计理念，提示我们在模型压缩中应优先考虑硬件友好性，这对边缘设备和实时应用场景具有重要意义。
3. CUR 分解在处理非线性组件时的应用虽然非最优，但提供了一种处理复杂结构的思路，未来可以探索结合其他近似方法（如张量分解）以进一步提升压缩性能。