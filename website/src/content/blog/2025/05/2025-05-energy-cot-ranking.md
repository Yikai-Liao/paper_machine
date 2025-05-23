---
title: "Learning to Rank Chain-of-Thought: An Energy-Based Approach with Outcome Supervision"
pubDatetime: 2025-05-21T01:06:29+00:00
slug: "2025-05-energy-cot-ranking"
type: "arxiv"
id: "2505.14999"
score: 0.5635275054707785
author: "grok-3-latest"
authors: ["Eric H. Jiang", "Haozheng Luo", "Shengyuan Pang", "Xiaomin Li", "Zhenting Qi", "Hengli Li", "Cheng-Fu Yang", "Zongyu Lin", "Xinfeng Li", "Hao Xu", "Kai-Wei Chang", "Ying Nian Wu"]
tags: ["LLM", "Energy-Based Model", "Chain-of-Thought", "Ranking", "Reasoning"]
institution: ["University of California, Los Angeles", "Northwestern University", "Zhejiang University", "Harvard University", "Peking University", "Nanyang Technological University"]
description: "本文提出 Energy Outcome Reward Model (EORM)，通过基于能量模型的后验验证和重新排序，仅依赖结果标签训练，显著提升了大型语言模型在数学推理任务中的准确率，同时降低了计算和标注成本。"
---

> **Summary:** 本文提出 Energy Outcome Reward Model (EORM)，通过基于能量模型的后验验证和重新排序，仅依赖结果标签训练，显著提升了大型语言模型在数学推理任务中的准确率，同时降低了计算和标注成本。 

> **Keywords:** LLM, Energy-Based Model, Chain-of-Thought, Ranking, Reasoning

**Authors:** Eric H. Jiang, Haozheng Luo, Shengyuan Pang, Xiaomin Li, Zhenting Qi, Hengli Li, Cheng-Fu Yang, Zongyu Lin, Xinfeng Li, Hao Xu, Kai-Wei Chang, Ying Nian Wu

**Institution(s):** University of California, Los Angeles, Northwestern University, Zhejiang University, Harvard University, Peking University, Nanyang Technological University


## Problem Background

大型语言模型（LLMs）在数学推理任务中面临多步逻辑一致性的挑战，尽管链式思维（Chain-of-Thought, CoT）提示方法能够生成中间推理步骤，但无法保证正确性；现有方法如自一致性通过大量采样提高准确率，但计算成本高昂，因此需要一种高效的验证机制来提升 CoT 输出的可靠性。

## Method

*   **核心思想:** 提出 Energy Outcome Reward Model (EORM)，一种基于能量模型（Energy-Based Models, EBMs）的轻量级后验验证器，通过为 CoT 解决方案分配标量能量分数（energy score）来重新排序候选答案，能量分数低的被认为是更高质量的推理路径。
*   **具体实现:** 
    *   使用 Transformer 编码器处理输入的 CoT 序列，提取特征表示，并通过一个多层感知机（MLP）头部将特征映射为一个标量能量值。
    *   训练时采用成对的 Bradley-Terry 损失函数，仅依赖最终结果的正确性标签（outcome labels），而无需逐步骤标注，确保能量分数低的解决方案对应正确的推理路径。
    *   推理时，从一组候选解决方案中选择能量最低的作为最终输出。
*   **优势:** 不需要修改原始语言模型，仅作为后验工具即可应用，避免了复杂强化学习或过程监督的训练负担，降低了数据标注和计算成本。

## Experiment

*   **有效性:** 在多个数学推理基准数据集（如 GSM8k 和 MATH）上，EORM 显著提高了最终答案的准确率，例如在 Llama 3 8B 模型上，GSM8k 准确率达到 90.7%，MATH 达到 63.7%，优于多种基线方法。
*   **泛化能力:** 在分布外（out-of-distribution, OOD）数据集（如 AIME 2024, AGIE Gaokao Math）上，EORM 表现出较强的泛化能力，表明其不仅在训练数据上有效，还能适应未见过的问题类型。
*   **实验设置合理性:** 实验通过生成大量 CoT 候选（例如每个问题 256 个候选）并使用 EORM 后验排序，模拟了实际应用场景；同时展示了随着候选数量增加准确率逐步提高的趋势，验证了方法的有效性。
*   **计算开销:** 虽然生成大量候选增加了一定计算成本，但 EORM 本身的排序过程是轻量级的，与自一致性等方法相比，在准确率和计算开销之间取得了更好的平衡。

## Further Thoughts

EORM 将能量模型（EBMs）应用于 CoT 验证的创新思路启发了我，是否可以将类似框架扩展到其他多步推理任务（如代码生成或法律推理）中；此外，EORM 仅依赖结果标签训练的特性提示我们可以在对话系统等领域设计轻量级验证器；其后验验证的模块化设计也让我思考如何开发其他通用后验工具来增强现有模型性能，而无需重新训练。