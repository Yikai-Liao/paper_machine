---
title: "Sentinel: Attention Probing of Proxy Models for LLM Context Compression with an Understanding Perspective"
pubDatetime: 2025-05-29T09:24:12+00:00
slug: "2025-05-sentinel-context-compression"
type: "arxiv"
id: "2505.23277"
score: 0.7078345369809946
author: "grok-3-latest"
authors: ["Yong Zhang", "Yanwen Huang", "Ning Cheng", "Yang Guo", "Yun Zhu", "Yanmeng Wang", "Shaojun Wang", "Jing Xiao"]
tags: ["LLM", "Proxy Model", "Context Compression", "Attention Signals", "RAG"]
institution: ["Ping An Technology (Shenzhen) Co., Ltd., China", "University of Electronic Science and Technology of China"]
description: "Sentinel 提出了一种轻量级上下文压缩框架，通过探针小规模代理模型的注意力信号实现高效压缩，在 LongBench 上达到 5 倍压缩率并匹配 7B 规模系统性能。"
---

> **Summary:** Sentinel 提出了一种轻量级上下文压缩框架，通过探针小规模代理模型的注意力信号实现高效压缩，在 LongBench 上达到 5 倍压缩率并匹配 7B 规模系统性能。 

> **Keywords:** LLM, Proxy Model, Context Compression, Attention Signals, RAG

**Authors:** Yong Zhang, Yanwen Huang, Ning Cheng, Yang Guo, Yun Zhu, Yanmeng Wang, Shaojun Wang, Jing Xiao

**Institution(s):** Ping An Technology (Shenzhen) Co., Ltd., China, University of Electronic Science and Technology of China


## Problem Background

检索增强生成（RAG）通过外部上下文增强大型语言模型（LLM）的能力，但检索到的上下文往往冗长、噪声多或超出输入限制，导致效率和效果问题。
现有压缩方法通常需要专门训练压缩模型，成本高且移植性差，Sentinel 旨在探索一种轻量级、模型无关的压缩方法，利用 LLM 内部的注意力信号判断上下文相关性，解决高效、查询相关上下文压缩的难题。

## Method

*   **核心思想:** 将上下文压缩重构为一个注意力理解任务，利用代理模型的解码器注意力信号来判断句子与查询的相关性，而无需训练专用压缩模型或依赖生成反馈。
*   **具体实现步骤:**
    *   **注意力特征提取:** 使用现成的 0.5B 参数规模代理模型（如 Qwen-2.5-0.5B-Instruct），对输入的查询-上下文对运行推理，提取最终 token 的多层解码器注意力矩阵，捕捉输入句子受到的注意力分布。
    *   **特征构建:** 对每个句子，计算其 token 在各层、各注意力头上的平均注意力权重，并进行归一化处理，生成特征向量，反映句子对最终输出的贡献。
    *   **相关性预测:** 训练一个轻量级逻辑回归分类器，基于注意力特征预测句子的相关性得分，分类器使用弱监督数据（从 QA 数据集构建正负样本）训练，避免手动标注成本。
    *   **推理时压缩:** 在推理阶段，根据分类器得分对句子排序，选择高相关性句子组成压缩后的上下文，满足输入长度限制（如 2000 token）。
*   **关键优势:** 方法不依赖生成能力，仅利用模型内部的注意力信号，计算成本低；通过注意力信号跨规模稳定性的假设，实现小模型代理大模型行为的可能性；设计模型无关，可无缝集成到任何 RAG 流程中。

## Experiment

*   **有效性:** 在 LongBench 基准上，Sentinel 使用 0.5B 代理模型实现了高达 5 倍的输入压缩率，同时在英文和中文任务中与 7B 规模压缩系统（如 CPC、LongLLMLingua）性能相当，例如在 GPT-3.5-Turbo 上的平均得分为 47.89，接近 CPC 的 49.5。
*   **任务表现差异:** 在单文档和多文档问答任务上，Sentinel 甚至超越无压缩的原始提示（输入长度 10,295 token），表明其能有效提取高相关性内容；但在摘要、少样本推理和代码任务上表现稍逊，可能是句子级压缩破坏了上下文结构。
*   **稳定性验证:** 实验验证了注意力相关性估计在不同代理模型规模（0.5B 到 3B）上的稳定性，性能差异小于 2 分，且句子选择重叠率高（在 3000 token 预算下达 0.74-0.78）。
*   **效率提升:** 推理延迟优于基线方法 LLMLingua-2，在分块大小为 1024 和 2048 token 时，速度提升达 1.13× 到 1.20×。
*   **实验设置合理性:** 实验覆盖多种任务类别、压缩比例（0.1 到 0.5）、分块大小（512 到 4096 token）以及多语言场景（英文和中文），设置全面，但在代码和少样本任务上的局限性提示需要进一步优化结构保留策略。

## Further Thoughts

Sentinel 揭示了注意力信号作为上下文理解天然代理的潜力，启发我们是否可以挖掘其他内部信号（如隐藏状态或层间表示）来增强压缩效果；此外，注意力相关性跨模型规模的稳定性是否能促成一种‘通用注意力探针’设计，适用于不同架构的 LLM，甚至在预训练阶段嵌入相关性估计机制；另一个方向是将此方法扩展至多模态 RAG，通过注意力信号压缩图像或音频上下文。