---
title: "Sentinel: Attention Probing of Proxy Models for LLM Context Compression with an Understanding Perspective"
pubDatetime: 2025-05-29T09:24:12+00:00
slug: "2025-05-attention-context-compression"
type: "arxiv"
id: "2505.23277"
score: 0.7078345369809946
author: "grok-3-latest"
authors: ["Yong Zhang", "Yanwen Huang", "Ning Cheng", "Yang Guo", "Yun Zhu", "Yanmeng Wang", "Shaojun Wang", "Jing Xiao"]
tags: ["LLM", "Proxy Model", "Context Compression", "Attention Probing", "Reasoning"]
institution: ["Ping An Technology (Shenzhen) Co., Ltd., China", "University of Electronic Science and Technology of China"]
description: "本文提出 Sentinel 框架，通过探测小型代理模型的注意力信号实现轻量级上下文压缩，在 LongBench 上达到 5 倍压缩率并匹配 7B 模型性能，为检索增强生成提供高效解决方案。"
---

> **Summary:** 本文提出 Sentinel 框架，通过探测小型代理模型的注意力信号实现轻量级上下文压缩，在 LongBench 上达到 5 倍压缩率并匹配 7B 模型性能，为检索增强生成提供高效解决方案。 

> **Keywords:** LLM, Proxy Model, Context Compression, Attention Probing, Reasoning

**Authors:** Yong Zhang, Yanwen Huang, Ning Cheng, Yang Guo, Yun Zhu, Yanmeng Wang, Shaojun Wang, Jing Xiao

**Institution(s):** Ping An Technology (Shenzhen) Co., Ltd., China, University of Electronic Science and Technology of China


## Problem Background

检索增强生成（RAG）通过外部上下文增强大型语言模型（LLM）的能力，但检索到的上下文往往冗长、噪声多或超出输入限制，导致效率和效果问题；现有压缩方法依赖监督训练专用模型，成本高且移植性差，亟需一种轻量级、模型无关的上下文压缩方案。

## Method

* **核心思想**：将上下文压缩重构为基于注意力的理解任务，通过探测小型代理模型（Proxy Model）的解码器注意力信号判断句子与查询的相关性，而无需训练专用压缩模型。
* **具体实现**：
  * 使用现成的 0.5B 规模代理模型（如 Qwen-2.5-0.5B-Instruct）处理查询-上下文对，提取最终解码器 token 的注意力矩阵，捕捉跨层和注意力头的信号。
  * 对每个句子计算注意力特征（通过对句子内 token 的注意力权重进行归一化和平均），形成特征向量。
  * 使用一个轻量级逻辑回归分类器，根据注意力特征预测句子的相关性概率；分类器通过弱监督数据（从 QA 数据集如 SQuAD, HotpotQA 构建）训练，数据经过上下文依赖性过滤和句子顺序打乱以提高鲁棒性。
  * 在推理时，根据分类器输出的相关性分数选择 top-k 句子，压缩上下文后传递给下游 LLM。
* **关键优势**：方法轻量（仅需小型代理模型和简单分类器），模型无关（不依赖特定 LLM 架构），无需生成监督信号或任务特定调优。

## Experiment

* **有效性**：在 LongBench 基准上，Sentinel 使用 0.5B 代理模型实现高达 5 倍上下文压缩，同时在问答任务中与 7B 规模压缩系统（如 CPC, LongLLMLingua）性能相当，例如在 GPT-3.5-Turbo 上的英文任务平均得分为 47.89，接近 CPC 的 49.5。
* **优越性**：相比基线（如 LLMLingua, Raw Attention），Sentinel 在单文档和多文档问答任务中表现突出，甚至在某些任务上超越未压缩的原始上下文输入。
* **鲁棒性**：实验覆盖多语言（英文和中文）、多任务（问答、摘要、代码补全等）和不同压缩比例（0.1 到 0.5）；测试不同代理模型规模（0.5B 到 3B）发现相关性估计稳定，0.5B 模型已足够有效。
* **局限性**：在摘要、少样本推理和代码任务中性能低于原始上下文，可能是句子级压缩破坏了全局结构或格式。
* **实验设置**：设置全面合理，数据和评估模型选择（如 GPT-3.5-Turbo, Qwen-2.5-7B-Instruct）具有代表性，消融实验（如 chunk 大小、压缩比例）进一步验证方法鲁棒性。

## Further Thoughts

论文提出的注意力信号作为上下文理解指标的视角令人启发，或许可以扩展到其他任务（如知识提取或模型解释性）；跨模型规模的注意力相关性稳定性提示在资源受限场景下小型模型可代理大型模型行为；此外，是否可以探索不同架构（如 LLaMA, Mistral）代理模型的信号一致性，或结合多模态输入（如图像+文本）扩展上下文压缩应用？