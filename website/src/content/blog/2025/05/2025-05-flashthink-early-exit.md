---
title: "FlashThink: An Early Exit Method For Efficient Reasoning"
pubDatetime: 2025-05-20T05:28:21+00:00
slug: "2025-05-flashthink-early-exit"
type: "arxiv"
id: "2505.13949"
score: 0.8474416211678248
author: "grok-3-latest"
authors: ["Guochao Jiang", "Guofeng Quan", "Zepeng Ding", "Ziqin Luo", "Dixuan Wang", "Zheng Hu"]
tags: ["LLM", "Early Exit", "Reasoning Efficiency", "Verification Model", "Inference Optimization"]
institution: ["Fudan University"]
description: "本文提出 FlashThink 方法，通过验证模型实现推理过程的提前退出，在保持模型准确性的同时显著减少推理内容长度，提升了大型语言模型的推理效率。"
---

> **Summary:** 本文提出 FlashThink 方法，通过验证模型实现推理过程的提前退出，在保持模型准确性的同时显著减少推理内容长度，提升了大型语言模型的推理效率。 

> **Keywords:** LLM, Early Exit, Reasoning Efficiency, Verification Model, Inference Optimization

**Authors:** Guochao Jiang, Guofeng Quan, Zepeng Ding, Ziqin Luo, Dixuan Wang, Zheng Hu

**Institution(s):** Fudan University


## Problem Background

大型语言模型（LLMs）在推理任务中表现出色，但经常生成冗长的推理内容，即使面对简单问题也会‘过度思考’，导致计算开销和推理时间显著增加，限制了实际应用中的效率。
作者通过观察发现，模型在生成推理内容的中途往往已经具备得出正确答案的能力，无需完成全部推理过程，因此提出通过提前退出推理阶段来提升效率。

## Method

*   **核心思想：** FlashThink 方法旨在通过一个外部验证模型（Verification Model）判断推理内容是否已足够得出正确答案，从而提前退出推理阶段，减少不必要的生成内容，同时不修改原始推理模型的参数。
*   **具体实现：**
    *   使用预定义的分隔符（Delimiter Tokens）将推理内容分割成多个小块（Chunks），便于逐段评估。
    *   在生成每个推理小块后，调用验证模型评估当前累计的推理内容是否足以解决问题，输出一个布尔值决定是否继续推理。
    *   如果验证模型判断可以提前退出，则直接生成最终答案；否则继续生成下一个推理小块，直至满足退出条件或达到最大生成长度。
*   **改进策略：** 提出 FT[2] 方法，通过构建正负样本（基于推理内容是否支持正确答案）对验证模型进行微调，使其适应特定推理模型和输入数据分布，进一步提升提前退出的准确性和效率。
*   **优势：** 该方法不改变原始模型的训练方式或参数，仅在推理阶段动态调整生成过程，计算开销主要来自验证模型的调用，整体成本可控。

## Experiment

*   **有效性：** 在四个基准数据集（GSM8K, MATH, GPQA Diamond, DROP）上，FlashThink 方法显著减少了推理内容长度，例如 DeepSeek-R1 和 QwQ-32B 的推理内容长度分别减少了 77.04% 和 77.47%，同时模型准确性得以保持甚至略有提升（DeepSeek-R1 平均得分从 87.00 提升到 87.15）。
*   **全面性与合理性：** 实验覆盖了数学和知识推理任务，数据集未下采样，确保结果代表性；测试了多个推理模型（如 DeepSeek-R1, QwQ-32B）及验证模型（如 Qwen2.5 系列），验证了方法的适应性；FT[2] 微调进一步提升了效果，例如 QwQ-32B 在 GPQA Diamond 上的准确性从 58.08 提升到 62.73，效率从 65.32% 提升到 75.64%。
*   **差异性：** 方法效果受模型特性和数据集复杂度的影响，例如 R1-Distill-Qwen-32B 在 GPQA Diamond 上的效率提升仅为 32.13%，表明优化效果存在上下文依赖性，需要针对性调整。
*   **开销：** 主要额外开销来自验证模型的调用，但相比原始推理模型的生成成本，整体计算负担较轻。

## Further Thoughts

FlashThink 的‘提前退出’机制提供了一个通用思路，即通过外部辅助模型动态调整生成过程，可以在不修改主模型参数的情况下提升效率，这种方法可能推广到其他生成任务（如文本摘要、对话生成）中；此外，验证模型的微调策略（FT[2]）启发我们可以通过定制辅助模型来优化主模型在特定任务上的表现，而无需重新训练整个大模型。