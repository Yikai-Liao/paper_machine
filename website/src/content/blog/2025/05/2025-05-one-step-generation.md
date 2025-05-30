---
title: "Exploring the Latent Capacity of LLMs for One-Step Text Generation"
pubDatetime: 2025-05-27T13:39:24+00:00
slug: "2025-05-one-step-generation"
type: "arxiv"
id: "2505.21189"
score: 0.7977113535498414
author: "grok-3-latest"
authors: ["Gleb Mezentsev", "Ivan Oseledets"]
tags: ["LLM", "Parallel Generation", "Embedding Space", "Text Reconstruction", "Non-Autoregressive"]
institution: ["AIRI", "Skoltech"]
description: "本文展示了冻结的大型语言模型通过两个可训练 proto-tokens 在单次前向传播中生成数百个准确 token 的能力，显著提升了生成效率并揭示了并行生成的潜力。"
---

> **Summary:** 本文展示了冻结的大型语言模型通过两个可训练 proto-tokens 在单次前向传播中生成数百个准确 token 的能力，显著提升了生成效率并揭示了并行生成的潜力。 

> **Keywords:** LLM, Parallel Generation, Embedding Space, Text Reconstruction, Non-Autoregressive

**Authors:** Gleb Mezentsev, Ivan Oseledets

**Institution(s):** AIRI, Skoltech


## Problem Background

大型语言模型（LLMs）传统上依赖自回归方式逐个生成 token，计算成本高昂，而近期研究表明 LLMs 可从单个训练嵌入自回归生成长文本；本文探索是否能在非自回归方式下，通过单次前向传播从压缩表示中重建多 token 序列，旨在揭示 LLMs 的并行生成潜力并解决自回归生成的高成本问题。

## Method

* **核心思想**：通过两个可训练的 'proto-tokens'（原型 token）作为输入，驱动冻结的 LLM 在单次前向传播中生成目标文本序列，而无需自回归迭代。
* **具体实现**：
  * 引入两个可训练嵌入向量 *e* 和 *m*，其中 *e* 通常位于输入序列开头，*m* 重复填充至目标序列长度 *N-1*，形成输入序列 [*e, m, m, ..., m*]。
  * 通过优化交叉熵损失训练这些嵌入，使 LLM 输出尽可能接近目标序列 [*t1, t2, ..., tN*]，采用标准因果注意力掩码确保预测依赖于前 *i* 个输入 token。
  * 探索 proto-tokens 的排列方式，发现至少需要两个 token 且 *e* 后接多个 *m* 的不对称排列效果最佳。
  * 研究 token 共享策略，即一个 proto-token（如 *m*）可在多个文本间复用，减少优化参数量，同时分析共享对性能的影响。
* **关键特点**：不修改 LLM 模型本身，仅通过输入嵌入优化实现并行生成，显著降低计算成本，同时保持模型冻结状态以确保通用性。

## Experiment

* **生成能力**：实验表明 LLMs 能在单次前向传播中生成数百个准确 token，Llama 系列中较大模型（如 8B）可重建高达 724 个 token，重建能力随模型规模增加而提升，而 Pythia 系列无明显规模效应。
* **方法有效性**：与自回归生成相比，非自回归方法在重建吞吐量上平均提升 279 倍，主要因单次前向传播的低计算成本，尤其适用于快速上下文压缩或设备端推理场景。
* **排列与共享影响**：至少需要两个 proto-tokens，*e* 后接多个 *m* 的排列方式效果最佳；共享 proto-token（如 *m*）时性能略降，但通过多次随机初始化可接近非共享效果。
* **文本类型影响**：自然文本（粉丝小说、PG-19 等）重建效果远优于随机文本，表明方法依赖 LLM 的语义理解而非简单 token 存储。
* **实验设置合理性**：实验覆盖多种模型（Pythia 160M-1.4B, Llama 1B-8B）、数据集（随机文本、自然文本、生成文本）和输入排列方式，设置较为全面，但对模型架构的依赖性（如 Llama 和 Pythia 差异）需进一步研究。

## Further Thoughts

论文揭示了 LLMs 在单次前向传播中生成长序列的潜力，启发我们思考如何利用嵌入空间的局部性和连通性构建专用编码器，将文本直接映射为压缩表示以加速推理；此外，proto-tokens 的功能分化（内容编码与结构支持）提示 LLM 注意力机制可能在并行生成中扮演关键角色，值得进一步探索。