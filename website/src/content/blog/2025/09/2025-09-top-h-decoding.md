---
title: "Top-H Decoding: Adapting the Creativity and Coherence with Bounded Entropy in Text Generation"
pubDatetime: 2025-09-02T17:02:29+00:00
slug: "2025-09-top-h-decoding"
type: "arxiv"
id: "2509.02510"
score: 0.79182563224338
author: "grok-3-latest"
authors: ["Erfan Baghaei Potraghloo", "Seyedarmin Azizi", "Souvik Kundu", "Massoud Pedram"]
tags: ["LLM", "Sampling", "Creativity", "Coherence", "Entropy"]
institution: ["University of Southern California", "Intel Labs"]
description: "本文提出 top-H 解码方法，通过熵约束动态调整 token 采样池，显著提升了大型语言模型在高温度下的文本生成质量，平衡了创造性和连贯性。"
---

> **Summary:** 本文提出 top-H 解码方法，通过熵约束动态调整 token 采样池，显著提升了大型语言模型在高温度下的文本生成质量，平衡了创造性和连贯性。 

> **Keywords:** LLM, Sampling, Creativity, Coherence, Entropy

**Authors:** Erfan Baghaei Potraghloo, Seyedarmin Azizi, Souvik Kundu, Massoud Pedram

**Institution(s):** University of Southern California, Intel Labs


## Problem Background

大型语言模型（LLMs）在开放式文本生成任务中面临一个核心挑战：如何在创造性（Creativity）和连贯性（Coherence）之间取得平衡。
现有的采样策略（如温度缩放、top-k、top-p 和 min-p 采样）在高温度设置下往往导致生成的文本失去连贯性，尤其是在需要更多创造力的场景中。
这些方法通常依赖于启发式规则，缺乏理论基础来系统地分析和控制概率分布的整体特性，从而难以有效适应模型的置信度变化。

## Method

*   **核心思想:** 提出 top-H 解码策略，通过引入熵约束（Entropy Constraint）来动态调整 token 采样池，确保生成的文本在创造性和连贯性之间达到更好的平衡。
*   **理论基础:** 作者将问题形式化为熵约束的最小发散问题（Entropy-Constrained Minimum Divergence, ECMD），即在采样子集的熵不超过原始分布熵的一定比例（由参数 α 控制）的前提下，最小化采样分布与模型原始预测分布之间的 Jensen-Shannon 发散（JSD）。
*   **问题转化:** 证明 ECMD 等价于熵约束的质量最大化问题（Entropy-Constrained Mass Maximization, ECMM），并进一步证明 ECMM 是 NP 难问题。
*   **实现方式:** 由于 ECMM 的计算复杂性，作者设计了一种贪婪算法（top-H 解码），具体步骤为：
    1. 将 token 按概率降序排列。
    2. 逐个将 token 加入采样池，并计算当前采样池归一化分布的熵。
    3. 当熵超过阈值 α·H(p)（其中 H(p) 为原始分布熵）时，停止添加并返回当前采样池。
*   **动态调整:** top-H 在自回归生成过程中，根据每一步的概率分布熵动态调整采样池大小，当模型不确定性高时允许更多 token 进入采样池以提升创造性，当模型置信度高时限制采样池以保证连贯性。
*   **参数控制:** 参数 α（默认设为 0.4）控制熵阈值的严格程度，影响创造性和连贯性的权衡。

## Experiment

*   **有效性:** 实验结果表明，top-H 在多个任务和数据集上显著优于现有方法，尤其是在高温度设置下。例如，在 GSM8K 数据集上，top-H 相较于 min-p 采样准确率提升高达 25.63%（LLaMA3.1-8B 模型，T=2.0）；在创意写作任务（Alpaca-Eval）中，胜率提升高达 17.11%。
*   **稳定性:** top-H 对温度变化表现出更强的鲁棒性，例如在 Alpaca-Eval 上，top-p 采样在 T=1 到 T=2 时胜率下降 34.06%，而 top-H 仅下降 3.78%。
*   **全面性:** 实验覆盖了创意写作（Alpaca-Eval, MT-Bench）、推理问答（GSM8K, GPQA）等多种任务，使用的模型包括 LLaMA3.1-8B、Qwen2.5-3B 和 Phi-3-Mini 等，评估方法包括准确率、胜率、LLM-as-judge 评分和人工评估，结果一致表明 top-H 的优势。
*   **计算开销:** top-H 的计算开销极低，与 top-p 和 min-p 相比，平均每 token 生成时间增加不到 1%，显示出良好的实用性。
*   **局限性探讨:** 虽然实验设置较为全面，但论文未深入讨论 top-H 在极低温度或特定领域任务（如代码生成）中的表现，参数 α 的调优依赖于开发集，可能存在泛化性问题。

## Further Thoughts

top-H 基于熵约束的动态采样策略启发我们，是否可以探索其他不确定性度量（如困惑度 Perplexity 或互信息）来进一步优化采样过程？此外，熵约束的理念是否可以扩展到多模态生成任务中，例如在图像生成或文本-图像对齐任务中控制生成内容的多样性和一致性？另一个值得思考的方向是，top-H 的低计算开销特性是否可以使其作为通用插件，集成到更广泛的生成框架中，甚至应用于在线推理场景以提升用户体验？