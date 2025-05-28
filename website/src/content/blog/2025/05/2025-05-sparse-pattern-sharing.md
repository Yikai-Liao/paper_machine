---
title: "Accelerating Prefilling for Long-Context LLMs via Sparse Pattern Sharing"
pubDatetime: 2025-05-26T06:48:53+00:00
slug: "2025-05-sparse-pattern-sharing"
type: "arxiv"
id: "2505.19578"
score: 0.7377083783460251
author: "grok-3-latest"
authors: ["Dan Peng", "Zhihui Fu", "Zewen Ye", "Zhuoran Song", "Jun Wang"]
tags: ["LLM", "Sparse Attention", "Long Context", "Pattern Sharing", "Inference Acceleration"]
institution: ["OPPO Research Institute", "Zhejiang University", "Shanghai Jiaotong University"]
description: "本文提出 SharePrefill 方法，通过动态生成并共享注意力头间的精确稀疏模式，显著加速长上下文大型语言模型的预填充阶段，同时保持高准确性。"
---

> **Summary:** 本文提出 SharePrefill 方法，通过动态生成并共享注意力头间的精确稀疏模式，显著加速长上下文大型语言模型的预填充阶段，同时保持高准确性。 

> **Keywords:** LLM, Sparse Attention, Long Context, Pattern Sharing, Inference Acceleration

**Authors:** Dan Peng, Zhihui Fu, Zewen Ye, Zhuoran Song, Jun Wang

**Institution(s):** OPPO Research Institute, Zhejiang University, Shanghai Jiaotong University


## Problem Background

大型语言模型（LLMs）在长上下文推理中的预填充（prefilling）阶段面临效率瓶颈。由于传统注意力机制的计算复杂度随序列长度呈平方增长，长上下文（如百万 token）推理在多文档问答、代码理解和多轮对话等应用中耗时严重。稀疏注意力（Sparse Attention）通过仅计算重要注意力分数来降低计算量，但现有方法依赖预定义的静态模式或不准确的动态估计，难以适应注意力模式的输入相关动态性，导致效率和准确性受限。本文旨在提出一种更精确、高效的稀疏注意力机制，加速预填充阶段，同时尽可能维持模型性能。

## Method

* **核心思想**：提出 SharePrefill 方法，利用注意力头（attention heads）间稀疏模式的高度相似性，通过动态生成并共享精确的稀疏模式，减少重复计算，加速长上下文 LLMs 的预填充阶段，同时保持高准确性。
* **关键观察**：基于两点发现：1）注意力头间的稀疏模式具有显著相似性（inter-head similarity）；2）这种相似性在不同输入下保持一致（cross-input similarity consistency）。
* **离线聚类（Offline Clustering）**：在离线阶段，使用自编码器（autoencoder）对注意力分数图（attention score maps）进行压缩表示，再通过层次聚类（hierarchical clustering）将相似的注意力头分组，形成簇，为后续模式共享奠定基础。
* **在线推理（Online Inference）**：在推理时，仅对每个簇中的少量‘关键头’（pivotal heads）计算完整注意力（full attention），生成精确的稀疏模式，并通过全局模式字典（pivotal pattern dictionary）将这些模式动态共享给簇内其他相似头，避免每个头单独计算完整注意力。
* **动态模式构建与共享**：通过计算块级平均注意力分数（block-wise average attention scores），结合累积分数阈值（cumulative score threshold）构建稀疏模式，并使用 Jensen-Shannon 距离（JS distance）验证头间相似性，确保共享模式的准确性。对于相似性不足或高度稀疏的头，回退到保守的垂直斜杠模式（vertical-slash pattern）。
* **稀疏注意力计算**：基于共享的稀疏模式，使用 Triton 实现的稀疏注意力内核，仅计算模式中标记为重要的块（blocks），跳过不重要区域，显著降低计算量。
* **创新点**：避免了静态模式的泛化问题和基于池化估计的不准确性，通过动态模式共享捕捉真实的注意力动态，同时减少计算开销。

## Experiment

* **准确性表现**：在 InfiniteBench 基准测试中，SharePrefill 在 Llama-3-8B-Instruct-262k 模型上的平均准确率为 39.05%，接近 MInference（39.14%），优于 FlexPrefill（36.44%）；在 Qwen2.5-7B-Instruct 模型上准确率为 31.79%，显著优于 MInference（29.23%）和 FlexPrefill（25.88%）。在 PG-19 语言建模任务中，困惑度（perplexity）接近全注意力方法 FlashAttention 2 和 MInference，优于 FlexPrefill（差距约 1.0-4.0）。
* **效率提升**：在延迟测试中，SharePrefill 在 128K 上下文长度下表现突出，例如在 Llama-3-8B 上延迟为 16.92 秒，低于 FlashAttention 2、MInference 和 FlexPrefill；在不同上下文长度下，延迟优势持续保持，展现出显著的加速效果。
* **实验设置合理性**：实验覆盖了两种主流长上下文模型（Llama-3-8B 和 Qwen2.5-7B），使用 InfiniteBench（平均 214K token）和 PG-19 数据集，包含多任务（问答、代码、数学等），充分验证了方法在长上下文场景下的适用性。消融研究分析了模式共享和稀疏头排除策略的影响，确认了各组件贡献。
* **总结**：SharePrefill 在准确性和效率之间取得了优越平衡，延迟显著降低，同时保持了接近全注意力的准确性，优于或接近现有最先进方法。

## Further Thoughts

论文中关于注意力头相似性和跨输入一致性的观察极具启发性，这种特性不仅限于预填充阶段的加速，或许可以扩展到解码阶段（decoding phase）或多模态系统的优化。模式共享的理念还可以进一步探索，例如是否可以通过跨设备共享模式字典来提升分布式推理效率？此外，注意力头相似性是否与模型架构或训练数据相关，能否通过设计模型结构增强这种相似性，从而进一步提升稀疏注意力的潜力？