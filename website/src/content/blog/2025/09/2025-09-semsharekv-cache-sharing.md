---
title: "SemShareKV: Efficient KVCache Sharing for Semantically Similar Prompts via Token-Level LSH Matching"
pubDatetime: 2025-09-29T14:16:13+00:00
slug: "2025-09-semsharekv-cache-sharing"
type: "arxiv"
id: "2509.24832"
score: 0.6676261707877027
author: "grok-3-latest"
authors: ["Xinye Zhao", "Spyridon Mastorakis"]
tags: ["LLM", "KV Cache", "Semantic Similarity", "Token Matching", "Memory Optimization"]
institution: ["University of Notre Dame"]
description: "SemShareKV 通过模糊 token 匹配和位置编码实现语义相似提示间的 KVCache 高效共享，显著提升推理速度并减少内存占用。"
---

> **Summary:** SemShareKV 通过模糊 token 匹配和位置编码实现语义相似提示间的 KVCache 高效共享，显著提升推理速度并减少内存占用。 

> **Keywords:** LLM, KV Cache, Semantic Similarity, Token Matching, Memory Optimization

**Authors:** Xinye Zhao, Spyridon Mastorakis

**Institution(s):** University of Notre Dame


## Problem Background

大型语言模型（LLMs）在推理过程中，Key-Value Cache（KVCache）占用了大量内存，尤其在处理长上下文时，由于解码器架构的二次时间复杂度，计算需求显著增加。
现有方法主要关注单提示内的 KVCache 压缩或共享前缀，忽略了语义相似但词汇不同的提示间共享缓存的潜力，而这种场景在多文档摘要和对话代理等任务中非常常见。
论文提出核心问题：是否可以为语义相似的提示复用预计算的 KVCache，从而减少计算冗余并提升效率？

## Method

*   **核心思想：** 通过模糊 token 匹配实现语义相似提示间的 KVCache 共享，减少预填充阶段（prefill phase）的计算开销并压缩内存占用，同时保持输出质量。
*   **具体实现步骤：**
    *   将接收到的提示及其嵌入缓存（E Cache）存储在 CPU 内存中，当新提示（目标提示）到达时，通过局部敏感哈希（Locality-Sensitive Hashing, LSH）距离计算与存储提示的相似性，选择最相似的参考提示及其 KVCache 加载到 GPU。
    *   使用旋转位置编码（Rotary Position Embedding, RoPE）对目标和参考提示的嵌入缓存进行位置信息增强，避免因位置差异导致的语义匹配错误。
    *   通过 LSH 对 token 嵌入进行模糊匹配，将参考提示的 KVCache 按目标提示的 token 顺序重新排列，确保缓存与目标提示对齐。
    *   在 Transformer 层中，采用分层策略：浅层更多重新计算 token 以保证精度，深层减少重新计算以节省资源；同时根据注意力分数动态保留高重要性 token，剔除冗余 token，优化内存使用。
*   **关键创新：** 不依赖精确文本匹配，而是基于语义相似性复用缓存；通过 RoPE 解决位置信息丢失问题；动态调整重新计算和保留策略以平衡性能和效率。

## Experiment

*   **有效性：** 在多个数据集（如 MultiNews, SAMSum, BookSum 等）上，SemShareKV 在预填充阶段实现了高达 6.25 倍的加速（相比完全重新计算和 H2O 基线），比 SnapKV 快 2.23 倍，同时 GPU 内存使用减少了 42%。
*   **性能维持：** 在 ROUGE-L 等指标上，SemShareKV 与基线方法相比性能下降微乎其微，甚至在某些数据集上表现更好，原因是选择性保留语义重要 token 减少了冗余信息。
*   **实验设置合理性：** 数据集覆盖摘要、问答和代码补全等多种任务，通过人工验证和 Llama3 模型改写构建语义相似样本，增强了评估的真实性；消融研究验证了模糊匹配和缓存保留策略的重要性。
*   **局限性：** 对于短提示（少于 700 个 token），由于模糊匹配和 token 重新排列的开销，性能提升有限；超参数需手动调优，匹配阈值也是经验性设置。

## Further Thoughts

基于语义相似性而非精确匹配的 KVCache 共享机制是一个值得关注的创新思路，可以进一步探索是否将语义匹配应用于动态推理调度，或结合其他嵌入技术（如 BERTScore）提升匹配精度；此外，RoPE 的位置编码应用也可能推广到其他缓存优化方法中，解决长上下文推理中的位置信息丢失问题。