---
title: "Accelerating Prefilling for Long-Context LLMs via Sparse Pattern Sharing"
pubDatetime: 2025-05-26T06:48:53+00:00
slug: "2025-05-shareprefill-sparse-attention"
type: "arxiv"
id: "2505.19578"
score: 0.7377083783460251
author: "grok-3-latest"
authors: ["Dan Peng", "Zhihui Fu", "Zewen Ye", "Zhuoran Song", "Jun Wang"]
tags: ["LLM", "Sparse Attention", "Long Context", "Pattern Sharing", "Efficiency"]
institution: ["OPPO Research Institute", "Zhejiang University", "Shanghai Jiaotong University"]
description: "本文提出 SharePrefill，通过动态生成并共享注意力头间的稀疏模式，显著加速长上下文大型语言模型的预填充阶段，同时保持最高精度。"
---

> **Summary:** 本文提出 SharePrefill，通过动态生成并共享注意力头间的稀疏模式，显著加速长上下文大型语言模型的预填充阶段，同时保持最高精度。 

> **Keywords:** LLM, Sparse Attention, Long Context, Pattern Sharing, Efficiency

**Authors:** Dan Peng, Zhihui Fu, Zewen Ye, Zhuoran Song, Jun Wang

**Institution(s):** OPPO Research Institute, Zhejiang University, Shanghai Jiaotong University


## Problem Background

大型语言模型（LLMs）在长上下文推理中的预填充阶段由于传统注意力机制的二次方计算复杂度，面临巨大的时间开销问题。
现有稀疏注意力方法依赖预定义的静态模式或不准确的估计，无法充分捕捉注意力机制的动态特性，导致效率和精度受限。
因此，需要一种能够在不牺牲模型性能的前提下加速预填充阶段的新方法。

## Method

*   **核心思想:** 提出 SharePrefill，一种基于注意力头之间模式相似性的稀疏注意力机制，通过动态生成并共享精确的稀疏模式来减少计算量，同时保持模型精度。
*   **关键观察:** 注意力头之间存在高度的模式相似性（inter-head similarity），且这种相似性在不同输入下保持一致（cross-input consistency），因此可以仅对一小部分头进行完整注意力计算，并将模式共享给其他相似头。
*   **具体实现:** 
    *   **离线阶段:** 使用自编码器对注意力分数图进行降维，并通过层次聚类将注意力头分组，基于相似性构建静态头字典，仅存储头索引而非模式本身。
    *   **在线阶段:** 在推理时动态构建关键模式（pivotal patterns），通过全局模式字典共享给相似头；对于不适合共享或高度稀疏的头，退回到保守的垂直斜杠模式（vertical-slash pattern）。
    *   **模式构建与验证:** 使用块级平均注意力分数（block-wise average attention scores）和累积分数阈值动态构建模式，并通过 Jensen-Shannon 距离验证相似性，确保共享的安全性。
*   **技术细节:** 稀疏注意力计算基于 Triton 内核，采用 FlashAttention 2 的块级策略，仅计算模式中标记为 1 的块，跳过标记为 0 的块，从而降低计算开销。
*   **优势:** 避免了预定义模式的局限性和池化估计的不准确性，作为无训练方法降低了资源成本，同时通过共享机制显著减少重复计算。

## Experiment

*   **精度表现:** 在 InfiniteBench 基准测试中，SharePrefill 在 Llama-3-8B-Instruct-262k 模型上取得了最高的平均精度（39.05%），优于 MInference（39.14%）和 FlexPrefill（36.44%）；在 Qwen2.5-7B-Instruct 模型上也表现出色（31.79%），显著优于 FlexPrefill（25.88%）。
*   **语言建模能力:** 在 PG-19 数据集的困惑度测试中，SharePrefill 的表现接近 FlashAttention 2 和 MInference（差距在 1.0 以内），并显著优于 FlexPrefill（在 Qwen2.5-7B-Instruct 上降低约 1.0-4.0）。
*   **效率提升:** 在 128K 上下文长度下，SharePrefill 的延迟显著低于基准方法（例如在 Llama-3-8B-Instruct-262k 上为 16.92 秒，优于 MInference 的 20+ 秒）。
*   **消融研究:** 去掉模式共享机制后精度下降至 38.70%，证明共享策略对精度维护至关重要；去掉高度稀疏头排除策略后精度略升至 39.35% 但延迟增加至 20.02 秒，表明排除策略对效率的贡献。
*   **实验设置合理性:** 实验覆盖了多种任务（问答、代码调试、数学查找等）和上下文长度（1K 至 128K），对比基准包括 FlashAttention 2、MInference 和 FlexPrefill，数据全面且结果可信。

## Further Thoughts

注意力头之间的相似性和跨输入一致性这一观察令人启发，是否可以通过分析头在模型中的功能角色（例如局部 vs 全局关注）进一步优化聚类和共享策略？此外，模式共享机制是否可以扩展到解码阶段或多模态模型中，尤其是在资源受限的边缘设备上部署长上下文模型时，可能带来更大的效率提升。