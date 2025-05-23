---
title: "When Can Large Reasoning Models Save Thinking? Mechanistic Analysis of Behavioral Divergence in Reasoning"
pubDatetime: 2025-05-21T08:55:35+00:00
slug: "2025-05-reasoning-save-thinking"
type: "arxiv"
id: "2505.15276"
score: 0.8246063666612478
author: "grok-3-latest"
authors: ["Rongzhi Zhu", "Yi Liu", "Zequn Sun", "Yiwei Wang", "Wei Hu"]
tags: ["LLM", "Reasoning", "RLHF", "Test Time Scaling"]
institution: ["Nanjing University", "University of California, Merced"]
description: "本文通过机制分析揭示了 RL 训练的大型推理模型在‘节省思考’指令下的行为分歧原因，识别三种思考模式（NT, ET, IT），并从内部状态和性能表现上为改进模型效率和可靠性提供了见解。"
---

> **Summary:** 本文通过机制分析揭示了 RL 训练的大型推理模型在‘节省思考’指令下的行为分歧原因，识别三种思考模式（NT, ET, IT），并从内部状态和性能表现上为改进模型效率和可靠性提供了见解。 

> **Keywords:** LLM, Reasoning, RLHF, Test Time Scaling

**Authors:** Rongzhi Zhu, Yi Liu, Zequn Sun, Yiwei Wang, Wei Hu

**Institution(s):** Nanjing University, University of California, Merced


## Problem Background

大型推理模型（Large Reasoning Models, LRMs）在复杂推理任务上表现出色，但常因过度思考（overthinking）导致计算资源浪费和准确率下降。
论文研究 RL 训练的 LRMs 在被提示‘节省思考’（save thinking）时为何表现出行为不一致性（有时跳过思考，有时重新推理），旨在揭示其内部机制并解决推理效率与准确性之间的平衡问题。

## Method

*   **行为模式分类**：将模型在‘节省思考’指令下的输出分为三种模式：无思考（No Thinking, NT，直接回答）、显式思考（Explicit Thinking, ET，重新推理并标记结束）和隐式思考（Implicit Thinking, IT，重新推理但不标记结束），通过手动标注输出行为进行分类。
*   **内部机制分析**：从以下三个维度深入剖析模型行为分歧的原因：
    1. **思考终止置信度**：分析模型在预测思考结束标记（如 </think>）时的 softmax 概率分布，计算 Top1 概率、熵和概率差值（DF），以评估模型对终止思考的信心，揭示其是否倾向于直接回答。
    2. **注意力激活模式**：通过主成分分析（PCA）可视化和 Davies-Bouldin 指数（DB Index）量化模型在不同层级的注意力激活差异，探索 NT、ET 和 IT 模式在内部注意力分配上的区别，尤其关注早期层级如何影响后续行为。
    3. **输入关注焦点**：分析模型生成第一个输出 token 时对输入不同部分的注意力分布（如用户指令和预填充思考内容），通过计算 Top1 注意力分数和差值（DF），揭示模型行为背后的关注重点及其与推理决策的关系。
*   **性能对比分析**：在不同行为模式下评估模型的准确率和输出长度，并与无预填充思考的基线对比，以量化‘节省思考’指令对推理效率和质量的影响。

## Experiment

*   **数据集与模型**：实验基于 GSM8K（小学数学问题）和 MATH500（高中竞赛数学问题）两个数据集，使用 RL 训练的开源模型 QwQ-32B 进行测试，覆盖不同任务难度。
*   **行为分布**：在 GSM8K 上，71.9% 的问题模型选择 NT 模式直接回答，而在更难的 MATH500 上仅 23.6% 选择 NT，表明任务难度显著影响行为模式选择。
*   **内部状态差异**：NT 模式下模型对思考终止标记的预测置信度更高（Top1 概率 78.04，熵 1.06），而 ET 和 IT 模式置信度较低（Top1 概率约 70，熵约 1.25）；注意力分析显示 NT 和 ET 模式在早期层级（第 5 层）即表现出显著差异（DB Index 急剧下降），NT 更关注预填充思考内容，ET 和 IT 更关注用户指令。
*   **性能结果**：NT 模式大幅缩短输出长度（节省超 99% token），但准确率显著下降（GSM8K 从 94.09% 降至 37.76%，MATH500 从 99.15% 降至 52.54%）；ET 模式在减少输出长度（GSM8K 减少 32.6%，MATH500 减少 13.8%）的同时保持甚至略提升准确率（GSM8K 96.35% vs 95.61%，MATH500 97.63% vs 95.78%）；IT 模式样本较少但趋势类似 ET。
*   **实验设置评价**：实验设计合理，覆盖不同难度数据集，手动标注确保分类准确，内部状态分析指标多样（置信度、注意力分布等），但局限在于仅聚焦数学推理任务，泛化性有待验证。

## Further Thoughts

论文揭示了模型早期注意力模式对推理行为的决定性影响，启发我们可以通过设计特定提示或训练策略调整早期注意力分配，以提升指令遵循性；此外，置信度与准确性的正相关性提示可以在推理时动态评估置信度，作为是否跳过思考的依据，可能为自适应推理策略提供新思路；最后，任务难度对行为模式的影响表明模型可能具备一定的难度感知能力，未来可探索如何利用这一特性优化推理控制机制。