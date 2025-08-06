---
title: "CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation"
pubDatetime: 2025-08-04T13:26:16+00:00
slug: "2025-08-compresskv-semantic-retrieval"
type: "arxiv"
id: "2508.02401"
score: 0.5652437129537533
author: "grok-3-latest"
authors: ["Xiaolin Lin", "Jingcun Wang", "Olga Kondrateva", "Yiyu Shi", "Bing Li", "Grace Li Zhang"]
tags: ["LLM", "KV Cache", "Compression", "Attention Heads", "Long Context"]
institution: ["Technical University of Darmstadt", "University of Notre Dame", "University of Siegen"]
description: "本文提出CompressKV框架，通过识别语义检索头和层自适应缓存分配，实现高效KV缓存压缩，在极低内存预算下显著提升大型语言模型的长上下文任务性能。"
---

> **Summary:** 本文提出CompressKV框架，通过识别语义检索头和层自适应缓存分配，实现高效KV缓存压缩，在极低内存预算下显著提升大型语言模型的长上下文任务性能。 

> **Keywords:** LLM, KV Cache, Compression, Attention Heads, Long Context

**Authors:** Xiaolin Lin, Jingcun Wang, Olga Kondrateva, Yiyu Shi, Bing Li, Grace Li Zhang

**Institution(s):** Technical University of Darmstadt, University of Notre Dame, University of Siegen


## Problem Background

大型语言模型（LLMs）在长上下文处理中面临键值（KV）缓存体积随输入长度线性增长的问题，导致内存占用高和推理速度慢的瓶颈。
现有KV缓存压缩方法多采用启发式token逐出策略，忽略了分组查询注意力（GQA）架构中注意力头的功能差异，导致重要token被错误剔除，模型性能下降。

## Method

*   **核心思想:** 提出CompressKV框架，通过识别并利用‘语义检索头’（Semantic Retrieval Heads）来精准确定重要token，结合层自适应缓存分配策略，实现高效KV缓存压缩，同时最大化模型性能。
*   **语义检索头识别:** 针对每一层注意力头，计算其在生成正确答案时对整个答案跨度的注意力分数总和，而非仅关注最高分数token，从而捕捉语义依赖和上下文信息，识别出能够检索重要token并关注其语义上下文的注意力头。
*   **重要token选择:** 在每一层中，选择得分最高的语义检索头（实验中为top-4），基于这些头的注意力分布平均值，确定最重要的token，保留其KV缓存对，剔除不重要token，所有注意力头共享同一组选定token索引。
*   **错误感知的层自适应缓存分配:** 离线计算每层在极度压缩情况下的注意力输出误差（通过Frobenius范数衡量压缩缓存与全缓存输出的差异），根据误差比例分配缓存预算，确保高误差层获得更多资源，同时设置最小和最大分配阈值以避免极端情况。
*   **关键优势:** 方法不依赖在线计算注意力统计，避免额外推理开销，且通过离线分析确保对不同模型的泛化能力。

## Experiment

*   **有效性:** 在LongBench基准上，CompressKV在极低缓存预算（如256 token）下保持了全缓存性能的97%-99%，在Needle-in-a-Haystack检索任务中，仅用0.07%的KV存储量即达到90%的基准准确率。
*   **优越性:** 相比基线方法（如StreamingLLM仅保留首尾token，SnapKV基于注意力分数聚类，CAKE引入动态分配），CompressKV在各种缓存预算下均表现更优，尤其在低内存场景下性能提升显著。
*   **实验设置合理性:** 实验覆盖了多种缓存预算（128到2048 token）、两种主流模型（Llama-3.1-8B-Instruct和Mistral-7B-Instruct-v0.3）以及多个任务类型（长上下文理解和检索），消融实验验证了语义检索头选择和层自适应分配的独立贡献。
*   **计算开销:** 方法依赖离线误差分析，不增加在线推理负担，延迟和峰值内存使用与其他逐出方法相当，显著优于全缓存基线。

## Further Thoughts

语义检索头的概念揭示了注意力头在长上下文任务中的功能差异，这一洞察不仅适用于KV缓存压缩，还可能启发模型剪枝、注意力机制优化或可解释性研究；此外，离线误差分析与在线推理分离的设计思路具有普适性，可推广至其他资源受限场景下的优化问题，如边缘设备上的模型部署。