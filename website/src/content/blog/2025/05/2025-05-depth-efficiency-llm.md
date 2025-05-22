---
title: "Do Language Models Use Their Depth Efficiently?"
pubDatetime: 2025-05-20T04:00:56+00:00
slug: "2025-05-depth-efficiency-llm"
type: "arxiv"
id: "2505.13898"
score: 0.8179251043817588
author: "grok-3-latest"
authors: ["Róbert Csordás", "Christopher D. Manning", "Christopher Potts"]
tags: ["LLM", "Transformer Architecture", "Depth Efficiency", "Residual Stream", "Compositional Computation"]
institution: ["Stanford University"]
description: "本文通过多角度分析揭示大型语言模型未有效利用深度，后半部分层主要用于概率分布微调而非复杂计算，提示需改进架构以提升深度效率。"
---

> **Summary:** 本文通过多角度分析揭示大型语言模型未有效利用深度，后半部分层主要用于概率分布微调而非复杂计算，提示需改进架构以提升深度效率。 

> **Keywords:** LLM, Transformer Architecture, Depth Efficiency, Residual Stream, Compositional Computation

**Authors:** Róbert Csordás, Christopher D. Manning, Christopher Potts

**Institution(s):** Stanford University


## Problem Background

大型语言模型（LLMs）的性能随着模型深度（即Transformer层数）的增加而提升，但收益逐渐递减。
作者提出核心问题：这些模型是否通过增加深度来执行更复杂的计算（例如通过层间组合生成更高阶特征），还是仅仅将相同的计算分散到更多层中？
这一问题关系到模型架构设计的效率和资源利用的有效性，可能解释规模扩展带来的性能提升为何逐渐减弱。

## Method

*   **残差流分析：** 通过计算每一层及子层（注意力层和MLP层）对残差流的贡献（使用L2范数和余弦相似性），评估各层对模型整体计算的影响，揭示层间贡献的分布模式。
*   **层跳跃干预：** 通过跳过特定层，观察对后续层计算和输出预测的影响，区分对当前token和未来token预测的作用，以判断层间依赖性和计算的组合性。
*   **多跳任务分析：** 针对复杂任务（如多跳推理和数学问题），使用因果干预和集成梯度方法，检查模型是否在更复杂的任务上利用更多层来组合子结果。
*   **线性映射实验：** 训练线性映射，从浅层模型的残差流预测深层模型的残差流，分析深层模型是否执行了浅层模型无法完成的计算，判断深度是否带来新型计算。
*   **MoEUT对比：** 探索其他架构（如Mixture-of-Experts Universal Transformers, MoEUT）是否能更有效地利用深度，通过对比实验评估不同架构在深度利用上的差异。

## Experiment

*   **残差流贡献：** 实验显示模型前半部分的层对残差流的贡献显著高于后半部分，后半部分的层主要用于微调输出概率分布，而非构建新特征。
*   **层跳跃影响：** 跳过后半部分的层对未来token预测的影响很小，表明这些层未参与可复用的子计算，更多专注于当前token的概率调整。
*   **多跳任务结果：** 在复杂任务中，未发现模型利用更多深度进行组合计算的证据，计算深度与任务复杂性无关。
*   **线性映射结果：** 深层模型的层与浅层模型的相对位置层对应最好，表明深层模型只是将相同计算‘拉伸’到更多层，而非执行全新计算。
*   **MoEUT对比：** MoEUT模型在某些情况下（如不建模问题部分时）似乎更有效地利用了深度。
*   **实验设置评价：** 实验覆盖多个模型（Llama 3.1 和 Qwen 3 系列）和任务类型（数学和多跳推理），分析方法多样（如Logitlens、因果干预），设置较为全面合理；但线性映射实验仅基于一对模型（Qwen 2.5 1.5B 和 14B），可能缺乏普适性。

## Further Thoughts

论文揭示了Transformer架构在深度利用上的低效性，提示可以探索新的模型设计，如引入自适应计算时间机制或显式中间步骤，以增强层间组合性计算能力；此外，MoEUT的初步结果表明共享参数或专家混合机制可能是一个值得深入研究的方向，以提高深度利用效率。