---
title: "Bottlenecked Transformers: Periodic KV Cache Abstraction for Generalised Reasoning"
pubDatetime: 2025-05-22T17:33:49+00:00
slug: "2025-05-bottlenecked-transformer-kv"
type: "arxiv"
id: "2505.16950"
score: 0.7954459800659404
author: "grok-3-latest"
authors: ["Adnan Oomerjee", "Zafeirios Fountas", "Zhongwei Yu", "Haitham Bou-Ammar", "Jun Wang"]
tags: ["LLM", "Information Bottleneck", "KV Cache", "Reasoning", "Generalization"]
institution: ["UCL Centre for AI", "Huawei Noah’s Ark", "Hong Kong University of Science and Technology", "University College London"]
description: "本文提出Bottlenecked Transformer架构，通过周期性重写KV缓存以压缩无关信息并增强预测能力，显著提升了Transformer在推理任务中的泛化性能，尤其是在外分布场景下。"
---

> **Summary:** 本文提出Bottlenecked Transformer架构，通过周期性重写KV缓存以压缩无关信息并增强预测能力，显著提升了Transformer在推理任务中的泛化性能，尤其是在外分布场景下。 

> **Keywords:** LLM, Information Bottleneck, KV Cache, Reasoning, Generalization

**Authors:** Adnan Oomerjee, Zafeirios Fountas, Zhongwei Yu, Haitham Bou-Ammar, Jun Wang

**Institution(s):** UCL Centre for AI, Huawei Noah’s Ark, Hong Kong University of Science and Technology, University College London


## Problem Background

大型语言模型（LLMs）基于Transformer架构在信息检索和模式识别任务中表现出色，但其泛化推理能力受限，尤其是在超出训练分布（Out-of-Distribution, OOD）的任务上，往往只能进行模式插值而非真正的抽象推理。
作者通过信息瓶颈理论（Information Bottleneck Theory, IB）分析，发现Transformer的KV缓存作为关键信息瓶颈，倾向于保留输入历史信息而非过滤无关内容，导致模型更偏向记忆而非抽象推理，限制了其在新型问题结构上的泛化能力。

## Method

*   **核心思想**：在Transformer架构中引入一个周期性的信息压缩机制，通过全局重写KV缓存（Key-Value Cache），优化其用于未来预测的能力，而非仅仅重建输入前缀，从而提升泛化推理能力。
*   **具体实现**：
    *   引入一个独立的Cache Processor模块（本身是一个小型Transformer），在生成或训练过程中，每隔固定数量的token（例如每16个token），对整个KV缓存进行全局更新。
    *   更新通过学习到的残差更新（Δ-update）实现，保留KV缓存的完整长度和维度，但重新分配其容量以聚焦于预测性特征，而非输入历史细节。
    *   Cache Processor在每个层级上操作KV缓存切片，采用无因果掩码（non-causal masking）以进行全局计算，并通过可学习的token促进层间信息传递。
    *   训练时采用标准的下一token交叉熵损失，不显式引入互信息正则化，而是依赖随机梯度下降（SGD）的噪声隐式压缩输入信息（即减少I(X;Z)）。
*   **关键优势**：此方法结合了RNN类模型的固定大小隐状态压缩特性与Transformer的无界KV缓存优势，避免了传统缓存剪枝方法（如H2O）在压缩时丢失预测信息的缺陷。

## Experiment

*   **有效性**：Bottlenecked Transformer在三个合成多步推理任务（整数乘法、多项式求值、数独）上表现出显著提升。例如，在整数乘法任务中，4M+2.8M参数的Bottlenecked Transformer分布内准确率达98.23%，远超16M参数Vanilla Transformer的80.17%；在OOD任务上准确率从2.42%提升至19.22%。在多项式求值任务中，OOD准确率从0%提升至13.50%，显示出更好的泛化能力。
*   **实验设置合理性**：任务设计为Markov决策过程（MDP），每次状态转换独立处理，确保模型依赖于每一步推理而非缓存上下文，真实测试推理能力。数据集包含分布内和分布外难度，评估全面，涵盖训练、验证和测试集。
*   **对比分析**：相较于缓存剪枝基线（如H2O），Bottlenecked Transformer避免了剪枝导致的预测信息丢失，性能更优；相较于更大规模的Vanilla Transformer（16M），其参数效率更高（总参数仅约6.8M）。
*   **局限性**：实验主要基于小型Transformer（4M-16M参数），未在更大规模预训练LLM上验证；Cache Processor的计算开销随序列长度增加显著，效率有待优化。

## Further Thoughts

信息瓶颈理论为分析和优化Transformer的泛化能力提供了新视角，未来是否可以探索自适应KV缓存更新机制，根据任务复杂度动态调整更新频率？此外，KV缓存重写类似于神经科学中的记忆巩固，是否可以引入‘离线’处理模块，模拟人类睡眠中的记忆整合，进一步提升模型的抽象推理能力？最后，是否可以将此方法与其他上下文压缩技术（如RAG）结合，以实现长上下文推理的高效泛化？