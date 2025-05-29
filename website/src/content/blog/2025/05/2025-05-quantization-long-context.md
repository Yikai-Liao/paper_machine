---
title: "Does quantization affect models' performance on long-context tasks?"
pubDatetime: 2025-05-26T17:54:30+00:00
slug: "2025-05-quantization-long-context"
type: "arxiv"
id: "2505.20276"
score: 0.8264704553088347
author: "grok-3-latest"
authors: ["Anmol Mekala", "Anirudh Atmakuru", "Yixiao Song", "Marzena Karpinska", "Mohit Iyyer"]
tags: ["LLM", "Quantization", "Long Context", "Performance Evaluation", "Multilingual"]
institution: ["UMass Amherst", "Microsoft", "University of Maryland, College Park"]
description: "本文通过系统评估揭示了量化对大型语言模型在长上下文任务中性能的影响，8位量化稳健而4位量化导致显著损失，强调了模型、任务和语言的异质性对量化效果的关键作用。"
---

> **Summary:** 本文通过系统评估揭示了量化对大型语言模型在长上下文任务中性能的影响，8位量化稳健而4位量化导致显著损失，强调了模型、任务和语言的异质性对量化效果的关键作用。 

> **Keywords:** LLM, Quantization, Long Context, Performance Evaluation, Multilingual

**Authors:** Anmol Mekala, Anirudh Atmakuru, Yixiao Song, Marzena Karpinska, Mohit Iyyer

**Institution(s):** UMass Amherst, Microsoft, University of Maryland, College Park


## Problem Background

大型语言模型（LLMs）在支持超过128K token的上下文窗口时，面临高内存需求和推理延迟的挑战，而量化（Quantization）作为降低成本的方法可能影响性能。
现有研究主要聚焦于短上下文任务，缺乏对长上下文任务（输入≥64K token）和长输出任务中量化效果的系统性评估，特别是在多语言和不同模型架构下的表现差异。

## Method

*   **核心思想:** 系统评估量化对大型语言模型在长上下文和长输出任务中性能的影响，探索不同量化方法、模型架构、任务类型和语言的影响差异。
*   **具体实现:** 
    *   选择了五种量化方法：FP8（8位浮点）、GPTQ-int8（8位整数）、AWQ-int4（4位整数）、GPTQ-int4（4位整数）和BNB-nf4（4位浮点），并以BF16作为全精度基准。
    *   评估了五种模型：Llama-3.1 8B 和 70B，以及Qwen-2.5 7B、32B 和 72B，涵盖不同规模和架构。
    *   任务设计包括长输入任务（RULER 和 ONE RULER 用于检索，NOCHA 用于长上下文推理）和长输出任务（FACTSCORE 用于事实性传记生成，CS4 用于受限故事生成），共覆盖9.7K个测试样本。
    *   实验变量包括上下文长度（8K、64K、128K）、语言（英语及26种其他语言）和任务难度，全面分析量化效果。
    *   使用广义线性混合效应模型（GLMM）进行统计分析，验证量化方法与基准的性能差异显著性。
*   **关键点:** 不修改模型结构，仅通过后训练量化（Post-Training Quantization）降低精度，关注推理阶段的性能表现，并通过多维度对比揭示量化影响的异质性。

## Experiment

*   **有效性:** 8位量化（FP8 和 GPTQ-int8）表现稳健，平均准确率下降仅0.2%-0.8%，与BF16基准在统计上无显著差异；4位量化（AWQ-int4、GPTQ-int4、BNB-nf4）导致更大损失，平均下降1.8%-6.9%，在长上下文任务中尤为严重（如Llama-3.1 70B在ONE RULER上的BNB-nf4量化下降59%）。
*   **上下文长度影响:** 随着输入上下文长度增加（从8K到128K），4位量化的性能下降更明显，特别是在检索任务中，平均下降高达23%。
*   **语言差异:** 非英语语言在量化后的性能下降更严重，最高达英语的5倍，尤其在低资源语言中表现突出。
*   **模型与任务异质性:** 量化效果因模型和任务而异，例如Qwen-2.5 72B在BNB-nf4下表现稳健，而Llama-3.1 70B下降32%；长输入任务比长输出任务对量化更敏感。
*   **实验设置合理性:** 实验覆盖了多种模型、量化方法和任务类型，测试样本量大（9.7K），统计分析严谨，充分揭示了量化影响的复杂性，但输出长度限制在650 token以内，未能完全反映超长输出的性能表现。

## Further Thoughts

量化效果的高度异质性启发我们，未来可以针对特定模型架构、任务类型或语言设计定制化量化策略，例如为低资源语言优化量化算法，或探索量化与上下文长度的动态适配机制，以最小化性能损失。