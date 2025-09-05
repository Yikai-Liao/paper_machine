---
title: "Adaptive KV-Cache Compression without Manually Setting Budget"
pubDatetime: 2025-09-03T08:38:40+00:00
slug: "2025-09-adaptive-kv-compression"
type: "arxiv"
id: "2509.03136"
score: 0.5921709528036959
author: "grok-3-latest"
authors: ["Chenxia Tang", "Jianchun Liu", "Hongli Xu", "Liusheng Huang"]
tags: ["LLM", "KV Cache", "Compression", "Adaptive Strategy", "Sampling"]
institution: ["未在文中明确提及"]
description: "本文提出GVote，一种自适应KV缓存压缩方法，通过基于高斯分布的未来查询采样动态确定缓存预算，显著提升大型语言模型推理的内存效率和精度平衡。"
---

> **Summary:** 本文提出GVote，一种自适应KV缓存压缩方法，通过基于高斯分布的未来查询采样动态确定缓存预算，显著提升大型语言模型推理的内存效率和精度平衡。 

> **Keywords:** LLM, KV Cache, Compression, Adaptive Strategy, Sampling

**Authors:** Chenxia Tang, Jianchun Liu, Hongli Xu, Liusheng Huang

**Institution(s):** 未在文中明确提及


## Problem Background

大型语言模型（LLMs）在自回归解码中依赖KV缓存来提升效率，但缓存内存占用随序列长度呈二次方增长，成为主要瓶颈；现有固定预算的KV缓存压缩方法无法适应多样化任务需求，导致内存效率和模型精度之间的权衡问题（即‘Procrustes’ bed problem’），例如在复杂任务上精度骤降，或在简单任务上浪费内存，同时还需要昂贵的超参数调优且对分布变化敏感；论文旨在解决这一根本性问题，提出自适应压缩方案以动态调整预算。

## Method

* **核心思想**：提出GVote，一种自适应KV缓存压缩方案，摒弃传统固定预算的‘自上而下’范式，采用‘自下而上’的方式，通过预测未来查询的注意力需求动态确定缓存预算。
* **理论基础**：基于隐藏状态呈现高斯分布的观察，利用统计特性采样合成未来查询（Synthetic Queries），估计未来注意力需求。
* **具体步骤**：
  1. **单步预算估计**：在预填充阶段，基于当前查询的注意力权重，通过核采样（Nucleus Sampling，阈值p_nuc）确定候选集C_0，其大小作为单步预算参考。
  2. **隐藏状态统计计算**：计算隐藏状态的均值和方差，拟合对角高斯分布，忽略初始Token以避免注意力沉积（Attention Sink）影响。
  3. **未来查询采样**：从高斯分布中采样S个隐藏状态，投影到查询空间并应用旋转位置编码（RoPE），生成合成查询Q'，基于注意力权重通过Top-K选择候选集（K值等于C_0大小，以避免噪声影响）。
  4. **投票与聚合**：将所有合成查询选择的候选集取并集，形成最终保留集（Keep-Set），动态确定预算B。
* **设计细节**：方法在每个注意力头和每个请求上独立执行，确保对异构工作负载的自适应性；使用Top-K而非Top-P以减少合成查询噪声导致的无关Token选择。
* **实现与开销**：GVote在GPU上通过向量化操作实现高效并行，核心开销集中在预填充阶段的合成查询生成和注意力计算，不随生成Token数量线性增加；支持现代注意力内核如FlashAttention。

## Experiment

* **有效性**：GVote在多个基准数据集（如GSM8K, RULER, Longbench）上显著降低内存使用（约2倍减少），同时保持或提升精度，例如在Multi-Doc QA数据集上以10%内存使用率达到0.35精度，而基线方法需双倍内存才能接近较低精度。
* **自适应性**：与固定预算方法（如StreamLLM, SnapKV, AdaKV）相比，GVote能针对每个请求动态调整压缩比例，解决不同任务对预算需求的差异问题。
* **泛化性**：GVote在不同模型架构和规模（如Llama3.1-8B, Qwen2.5-7B等）上均表现出色，表明其自适应机制具有良好的通用性。
* **超参数影响**：采样数量S和核阈值p_nuc对性能有影响，推荐S≥8和p_nuc=0.95作为精度与效率的折中，较高的值会提升精度但增加内存和计算开销。
* **实验设置合理性**：实验覆盖多种任务类型和序列长度，数据集和基线选择具有代表性，测试了不同模型规模，结果全面；不足之处在于未深入探讨极长序列（如>128K Token）下的性能和内存开销。

## Further Thoughts

GVote基于隐藏状态高斯分布采样合成未来查询的思路启发了我，是否可以结合上下文语义或任务类型进一步优化合成查询生成，例如通过任务分类器预判查询需求模式；此外，自适应压缩的思想可以扩展到其他资源管理场景，如动态调整模型推理中的计算分配或层级剪枝；投票机制的鲁棒性也让我思考是否能在噪声环境下（如多智能体系统）中应用类似聚合策略来提升决策稳定性。