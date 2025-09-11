---
title: "COMPACT: Common-token Optimized Model Pruning Across Channels and Tokens"
pubDatetime: 2025-09-08T16:07:06+00:00
slug: "2025-09-compact-pruning"
type: "arxiv"
id: "2509.06836"
score: 0.8116607153238521
author: "grok-3-latest"
authors: ["Eugene Kwek", "Wenpeng Yin"]
tags: ["LLM", "Model Compression", "Pruning", "Vocabulary Optimization", "Activation Analysis"]
institution: ["Penn State University"]
description: "本文提出 COMPACT 框架，通过无训练的词汇剪枝和基于常见词加权的FFN通道剪枝，实现跨规模大型语言模型的高效压缩，同时保持性能和架构兼容性。"
---

> **Summary:** 本文提出 COMPACT 框架，通过无训练的词汇剪枝和基于常见词加权的FFN通道剪枝，实现跨规模大型语言模型的高效压缩，同时保持性能和架构兼容性。 

> **Keywords:** LLM, Model Compression, Pruning, Vocabulary Optimization, Activation Analysis

**Authors:** Eugene Kwek, Wenpeng Yin

**Institution(s):** Penn State University


## Problem Background

大型语言模型（LLMs）因参数规模庞大而面临高内存占用、推理延迟和能耗成本的挑战，限制了其在边缘设备和延迟敏感场景中的部署。
现有剪枝方法存在局限：深度剪枝移除整个层导致性能骤降，宽度剪枝常破坏标准Transformer架构或需要定制推理代码，且缺乏对参数分布和语言特性的分析，导致剪枝效果在不同规模模型上不稳定。

## Method

*   **核心思想：** 提出 COMPACT 框架，通过联合词汇剪枝和基于常见词加权的FFN中间通道剪枝，实现高效模型压缩，同时保持标准Transformer架构和性能。
*   **词汇剪枝（Vocabulary Pruning）：** 基于自然语言词汇频率遵循Zipf定律，移除词汇表中稀有词对应的嵌入和解嵌入矩阵行，直接减少参数量，尤其对小模型有效（嵌入层参数占比高）。
*   **常见词加权FFN剪枝（Common-Token-Weighted FFN Pruning）：** 使用激活值评估FFN中间通道的重要性，但仅考虑剪枝后仍有效的常见词的激活值（通过Common Act²方法），确保剪枝优化针对常见词分布，而非完整词汇分布。
*   **联合剪枝流程：** 先识别稀有词集合，基于常见词分布计算FFN通道重要性，最后同时移除词汇参数和FFN通道，形成一个整体 pipeline。
*   **特点：** 无需训练，仅需少量校准数据即可完成剪枝；通过调整词汇和FFN剪枝比例，适应不同规模模型（小模型重词汇剪枝，大模型重FFN剪枝）；保持标准架构，兼容现有推理框架。

## Experiment

*   **性能表现：** 在Qwen、LLaMA、Gemma等模型家族（0.5B-70B参数规模）上，COMPACT 在高剪枝比例（高达35%）下仍保持较好的下游任务性能（如MMLU、GSM8K），尤其在小模型上显著优于基线方法（如SliceGPT、ShortGPT），延迟性能崩溃。
*   **规模适应性：** 通过词汇和FFN剪枝的互补策略，COMPACT 对小模型和大模型均表现稳健，解决了现有方法在不同规模上的不稳定性问题。
*   **效率提升：** 在内存使用上显著降低GPU占用（例如LLaMA 3.1-8B减少至0.64x），推理吞吐量提升（1.37x）；剪枝时间短（70B模型仅需2分17秒），接近深度剪枝效率。
*   **实验设置：** 覆盖多种模型架构和规模，测试7个下游任务（分类和生成），评估指标全面（参数量、内存、吞吐量、性能）；未报告困惑度指标（因词汇剪枝导致不公平），显示设计细致性。

## Further Thoughts

COMPACT 基于参数分布和语言特性进行剪枝的思路可扩展至其他模型组件（如注意力头）或多模态任务；词汇剪枝启发任务特定压缩（如医疗领域保留术语）；常见词加权激活值的思想可应用于蒸馏、量化等技术，或结合自适应推理动态调整模型容量。