---
title: "When Can Large Reasoning Models Save Thinking? Mechanistic Analysis of Behavioral Divergence in Reasoning"
pubDatetime: 2025-05-21T08:55:35+00:00
slug: "2025-05-reasoning-save-thinking"
type: "arxiv"
id: "2505.15276"
score: 0.8246063666612478
author: "grok-3-latest"
authors: ["Rongzhi Zhu", "Yi Liu", "Zequn Sun", "Yiwei Wang", "Wei Hu"]
tags: ["LLM", "Reasoning", "Attention Mechanism", "Reinforcement Learning", "Behavioral Analysis"]
institution: ["State Key Laboratory for Novel Software Technology, Nanjing University, China", "University of California, Merced, USA", "National Institute of Healthcare Data Science, Nanjing University, China"]
description: "本文通过机制分析揭示了RL训练的大型推理模型在节省思考指令下的三种行为模式（NT, ET, IT）及其内部状态差异，为提升模型效率和可靠性提供了重要见解。"
---

> **Summary:** 本文通过机制分析揭示了RL训练的大型推理模型在节省思考指令下的三种行为模式（NT, ET, IT）及其内部状态差异，为提升模型效率和可靠性提供了重要见解。 

> **Keywords:** LLM, Reasoning, Attention Mechanism, Reinforcement Learning, Behavioral Analysis

**Authors:** Rongzhi Zhu, Yi Liu, Zequn Sun, Yiwei Wang, Wei Hu

**Institution(s):** State Key Laboratory for Novel Software Technology, Nanjing University, China, University of California, Merced, USA, National Institute of Healthcare Data Science, Nanjing University, China


## Problem Background

大型推理模型（Large Reasoning Models, LRMs）在复杂任务上表现出色，但常因过度思考（overthinking）导致计算效率低下，尤其是在简单任务上生成冗长的推理链。
本文聚焦于通过强化学习（RL）训练的LRMs在被提示‘节省思考’（save thinking）时的行为不一致性问题：有时模型直接跳过思考生成答案，有时却重新进行显式或隐式思考。
这种行为分歧可能影响模型的效率和准确率，作者旨在揭示其背后的内部机制，并探索如何在不牺牲性能的前提下提高效率。

## Method

*   **核心思想:** 通过分析RL训练的LRMs在节省思考指令下的行为模式，揭示导致行为分歧的内部机制，为改进模型效率和可靠性提供依据。
*   **行为模式分类:** 将模型在节省思考指令下的行为分为三种模式：无思考（No Thinking, NT，跳过思考直接生成答案）、显式思考（Explicit Thinking, ET，重新进行思考并标记结束）和隐式思考（Implicit Thinking, IT，重新思考但不标记结束）。
*   **分析视角:** 从以下三个角度深入研究模型内部状态：
    *   **预测置信度（Confidence in Thinking Termination）**：测量模型在预测思考结束标记（如</think>）时的置信度（通过Top1概率、熵和概率差值DF等指标），以判断模型是否倾向于终止思考。
    *   **注意力机制（Attention from Thinking to Generation）**：通过层级注意力激活向量和PCA可视化，分析模型从输入处理到答案生成时的注意力分布差异，揭示不同模式下的内部表征特性，并使用Davies-Bouldin Index量化模式间分离度。
    *   **输入部分的注意力焦点（Attentional Focus on Input Sections）**：分析模型对输入提示不同部分的注意力分配（如用户问题部分和预填充思考部分），探索模型行为分歧与注意力重点的关系。
*   **实验实施:** 使用RL训练的QwQ-32B模型，在GSM8K和MATH500数据集上测试，结合手动标注和定量分析，评估不同模式下的性能表现（准确率和输出长度）。

## Experiment

*   **有效性:** 实验结果表明，不同思考模式对模型性能有显著影响。NT模式下输出长度大幅减少（节省超过99%的token），但准确率显著下降（GSM8K从94.09%降至37.76%，MATH500从99.15%降至52.54%）；ET模式下输出长度适度减少（GSM8K减少约32.6%，MATH500减少约13.8%），准确率却维持甚至略有提升（GSM8K达96.35%，MATH500达97.63%）；IT模式样本较少，但趋势类似ET，准确率保持较高。
*   **合理性:** 实验设置较为全面，选取了两个难度不同的数学推理数据集（GSM8K和MATH500），并与无预填充思考的基线对比，数据标注和手动验证也增加了结果可信度。此外，内部状态分析（如置信度和注意力分布）与行为模式和性能表现形成了逻辑闭环。
*   **局限性:** 数据集局限于数学推理领域，可能限制结论的泛化性；IT模式样本较少，分析不够充分；未探索不同模型或训练方法的对比。

## Further Thoughts

论文揭示了RL训练的LRMs在指令遵循和推理需求之间的内在矛盾，NT模式下模型更关注预填充内容，而ET和IT模式下更关注任务本身，这种注意力分配差异可能与训练目标有关。未来是否可以通过设计自适应提示或调整RL奖励函数（如引入长度惩罚或指令遵循奖励）来平衡效率和准确率？此外，注意力机制在早期层（第5层）的分歧提示我们，早期层可能对行为模式有关键影响，是否可以通过针对性微调早期层或设计注意力引导机制来改善模型对节省思考指令的遵循性？