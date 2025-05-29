---
title: "Exploring the Latent Capacity of LLMs for One-Step Text Generation"
pubDatetime: 2025-05-27T13:39:24+00:00
slug: "2025-05-one-step-generation"
type: "arxiv"
id: "2505.21189"
score: 0.7977113535498414
author: "grok-3-latest"
authors: ["Gleb Mezentsev", "Ivan Oseledets"]
tags: ["LLM", "Non-Autoregressive Generation", "Embedding Space", "Parallel Generation", "Text Reconstruction"]
institution: ["AIRI", "Skoltech"]
description: "本文展示冻结的大型语言模型通过两个可训练 proto-tokens 在一次前向传播中生成数百个 token 的能力，显著提升生成吞吐量并揭示嵌入空间的结构特性，为非自回归生成和高效推理开辟了新方向。"
---

> **Summary:** 本文展示冻结的大型语言模型通过两个可训练 proto-tokens 在一次前向传播中生成数百个 token 的能力，显著提升生成吞吐量并揭示嵌入空间的结构特性，为非自回归生成和高效推理开辟了新方向。 

> **Keywords:** LLM, Non-Autoregressive Generation, Embedding Space, Parallel Generation, Text Reconstruction

**Authors:** Gleb Mezentsev, Ivan Oseledets

**Institution(s):** AIRI, Skoltech


## Problem Background

大型语言模型（LLMs）传统上依赖自回归方式逐个生成 token，效率较低，而近期研究表明 LLMs 可从单个训练嵌入自回归生成长文本；本文探索是否能在非自回归模式下，通过一次前向传播从压缩表示重建多 token 序列，旨在揭示 LLMs 的并行生成潜力并提升生成效率。

## Method

* **核心思想**：通过两个可训练的 'proto-tokens'（原型 token）作为输入，利用冻结的 LLM 在一次前向传播中生成目标文本序列，绕过自回归迭代的低效性。
* **具体实现**：
  * 设计两个可训练嵌入向量 *e* 和 *m*，构造输入序列为 [e, m, m, ..., m]，其中 *e* 出现一次，*m* 重复 N-1 次（N 为目标序列长度）。
  * 使用交叉熵损失优化嵌入向量，使 LLM 基于输入序列预测目标序列，预测过程采用因果注意力掩码，确保第 i 个 token 预测依赖前 i 个输入。
  * 测试多种 proto-tokens 排列方式，发现 [e][m × (N-1)] 效果最佳，表明两个 token 数量和排列对性能至关重要。
  * 探索 token 共享策略，发现 *m* token 可在多文本间共享而不显著影响性能，减少优化参数量，暗示其可能承担结构或机制性角色。
* **关键创新**：不修改 LLM 本身，仅通过少量可训练嵌入实现长序列生成，挑战传统自回归范式，为高效并行生成提供新思路。

## Experiment

* **有效性**：使用两个 proto-tokens，LLMs 可通过一次前向传播准确重建数百个 token（Llama 3.1-8B 最高达 724 token），相比自回归方式（最高 1568 token）稍短，但吞吐量提升显著（平均 279 倍），验证了非自回归生成的效率优势。
* **模型规模影响**：Llama 系列模型重建能力随规模增大而提升（1B 为 362 token，8B 为 724 token），Pythia 模型无明显趋势（160M-1.4B 均在 90-181 token 范围），可能与架构差异相关。
* **文本类型影响**：自然文本（PG-19, AO3 fanfics）和生成文本重建能力相似，随机文本性能显著下降，表明 proto-tokens 编码高层次语言表示而非直接 token 信息。
* **输入排列与共享**：单一 proto-token 几乎无法重建多 token 序列，两个 token 及其排列 [e][m × (N-1)] 至关重要；共享 *m* token 在多文本间表现良好，优化效率提升。
* **实验合理性与局限**：实验覆盖多种模型、数据集和排列方式，设置较为全面，但受限于计算资源，训练迭代次数（5000 次）可能不足以完全收敛，且未深入探讨架构依赖性。

## Further Thoughts

非自回归生成揭示了 LLMs 的并行生成潜力，未来可设计专用编码器将文本映射到 proto-tokens 空间，实现更高效推理；嵌入空间的局部性和连通性为表示学习提供了新思路，可用于上下文压缩或生成控制；proto-tokens 的功能分化（信息编码与结构支持）启发是否能通过设计多功能嵌入进一步优化性能；此外，结合自回归与非自回归方式可能在长度和质量上实现更好平衡。