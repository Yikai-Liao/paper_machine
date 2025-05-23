---
title: "Do Language Models Use Their Depth Efficiently?"
pubDatetime: 2025-05-20T04:00:56+00:00
slug: "2025-05-llm-depth-efficiency"
type: "arxiv"
id: "2505.13898"
score: 0.8179251043817588
author: "grok-3-latest"
authors: ["Róbert Csordás", "Christopher D. Manning", "Christopher Potts"]
tags: ["LLM", "Transformer Depth", "Residual Stream", "Layer Efficiency", "Compositional Computation"]
institution: ["Stanford University"]
description: "本文通过多维度分析揭示大型语言模型未有效利用深度，后半部分层主要细化概率分布而非执行复杂组合计算，深层模型仅拉伸浅层计算而非创新。"
---

> **Summary:** 本文通过多维度分析揭示大型语言模型未有效利用深度，后半部分层主要细化概率分布而非执行复杂组合计算，深层模型仅拉伸浅层计算而非创新。 

> **Keywords:** LLM, Transformer Depth, Residual Stream, Layer Efficiency, Compositional Computation

**Authors:** Róbert Csordás, Christopher D. Manning, Christopher Potts

**Institution(s):** Stanford University


## Problem Background

大型语言模型（LLMs）的性能随着深度（Transformer层数）增加而提升，但收益递减。
作者质疑这些模型是否有效利用了深度：是否通过增加层数执行更复杂的组合性计算（构建高级特征），还是仅仅将相同类型的计算分散到更多层中？
这一问题关系到模型架构设计和资源利用效率，可能揭示当前Transformer架构在规模扩展上的局限性。

## Method

*   **核心思想:** 通过多维度分析，探究大型语言模型是否利用深度进行复杂计算，或仅将计算分散到更多层。
*   **具体方法:**
    *   **残差流分析:** 测量各层及子层（注意力层和MLP层）输出对残差流的贡献（使用L2范数和余弦相似度），以评估各层对整体计算的影响力。
    *   **层跳跃实验:** 通过跳跃特定层，观察对后续层计算和未来token预测的影响，判断层间依赖性和计算作用（是否构建子结果或仅细化概率分布）。
    *   **多跳任务分析:** 针对多跳推理和复杂数学问题，使用因果干预（如残差擦除）和集成梯度，检查模型是否根据任务复杂度调整计算深度。
    *   **线性映射实验:** 训练线性映射，从浅层模型的残差流预测深层模型的残差流，分析深层模型是否执行新型计算，或仅拉伸浅层模型的计算。
    *   **MoEUT对比:** 探索性地分析Mixture-of-Experts Universal Transformers（MoEUT），对比其深度利用效率与标准Transformer的差异。
*   **关键点:** 方法结合定量指标和因果干预，聚焦Llama 3.1和Qwen 3系列模型，主要在数学任务上验证深度利用效率。

## Experiment

*   **有效性:** 在Llama 3.1 70B等模型上，残差流分析显示后半部分层贡献显著低于前半部分（图2a）；层跳跃实验表明后半部分层对未来token预测影响较小（图3b），主要用于当前token概率分布细化（图4）。
*   **任务复杂度无关性:** 在MATH和MQuAKE数据集上，深度分数未随任务难度或跳数增加而变化（图7），表明模型未动态调整计算深度。
*   **深浅模型对比:** 线性映射实验显示Qwen 2.5 14B与1.5B模型层对应呈对角线趋势（图8），深层模型未展现新型计算，仅拉伸浅层计算。
*   **合理性与局限性:** 实验覆盖多个模型和数学任务（GSM8K, MATH, MQuAKE），设置合理且对复杂计算敏感；但未广泛涉及其他任务领域，线性映射实验仅限于单一模型对，结论普适性需进一步验证。

## Further Thoughts

模型未根据任务复杂度调整计算深度的现象，提示是否可以通过引入自适应计算机制（如PonderNet）或显式递归结构来提升深度效率；MoEUT在深度利用上的优越性启发未来架构设计可探索混合专家和共享层机制；此外，残差宽度作为潜在瓶颈，是否可以通过动态路由或维度调整来优化深度计算的组合性？