---
title: "When to Continue Thinking: Adaptive Thinking Mode Switching for Efficient Reasoning"
pubDatetime: 2025-05-21T11:41:39+00:00
slug: "2025-05-adaptive-reasoning-efficiency"
type: "arxiv"
id: "2505.15400"
score: 0.8211307104211535
author: "grok-3-latest"
authors: ["Xiaoyun Zhang", "Jingqing Ruan", "Xing Ma", "Yawen Zhu", "Haodong Zhao", "Hao Li", "Jiansong Chen", "Ke Zeng", "Xunliang Cai"]
tags: ["LLM", "Reasoning", "Efficiency", "Adaptive Allocation", "Safety Alignment"]
institution: ["Meituan"]
description: "本文提出自适应自恢复推理框架（ASRR），通过利用模型的内部自恢复机制和动态长度惩罚，根据任务难度动态分配推理资源，实现高效推理并提升安全对齐性能。"
---

> **Summary:** 本文提出自适应自恢复推理框架（ASRR），通过利用模型的内部自恢复机制和动态长度惩罚，根据任务难度动态分配推理资源，实现高效推理并提升安全对齐性能。 

> **Keywords:** LLM, Reasoning, Efficiency, Adaptive Allocation, Safety Alignment

**Authors:** Xiaoyun Zhang, Jingqing Ruan, Xing Ma, Yawen Zhu, Haodong Zhao, Hao Li, Jiansong Chen, Ke Zeng, Xunliang Cai

**Institution(s):** Meituan


## Problem Background

大型推理模型（LRMs）在复杂任务上通过长推理链表现出色，但常因‘过度思考’（Overthinking）在简单任务上生成冗长推理，导致计算资源浪费；此外，冗长推理还可能引发安全对齐问题，如生成不安全输出。
核心挑战在于如何让模型根据任务难度动态调整推理长度，在简单任务上减少不必要推理，在复杂任务上保留足够推理深度以保证准确性。

## Method

*   **核心思想:** 提出自适应自恢复推理框架（ASRR），利用模型内在的‘内部自恢复机制’（即在无显式推理模式下仍能通过隐式推理补充必要步骤），根据任务难度动态分配推理资源。
*   **具体实现:** 
    *   **显式推理抑制与隐式自恢复:** 在训练和推理时，通过在输入提示中添加‘No-Thinking’前缀（如‘Okay, I have finished thinking.’），抑制简单任务上的冗长推理，同时允许模型在困难任务上通过隐式推理恢复必要步骤。这种设计避免了简单任务的计算浪费，同时保证复杂任务的准确性。
    *   **动态长度惩罚（Dynamic Length Penalty, DLP）:** 引入基于准确率阈值的奖励调节机制。在强化学习（RL）训练中，将训练数据分组并计算组级准确率（Group-wise Accuracy），仅当准确率达到预设阈值时激活长度惩罚，且惩罚强度随准确率提升而动态增加（通过公式调节惩罚系数）。此外，使用‘过长比例’（Overlong Ratio）计算每个样本的长度惩罚，确保模型先关注正确性，再优化效率，避免‘短而错’或‘准而冗长’的问题。
*   **关键优势:** 不依赖额外的控制结构或复杂提示工程，通过强化学习让模型自适应地感知任务难度并调整推理深度，同时提升安全对齐性能。

## Experiment

*   **效率提升:** 在 DeepSeek-R1-Distill-Qwen-1.5B 和 7B 模型上，ASRR 相比基线 GRPO 分别减少了 32.5% 和 25.7% 的生成长度，准确率（pass@1）仅下降 1.2% 和 0.6%，表明方法在大幅降低计算开销的同时保持了性能。
*   **难度感知能力:** 在困难任务（如 AIME）上，‘继续思考’（Continue-Thinking）比例高达 80.6%（1.5B）和 81.5%（7B），准确率显著提升；在简单任务（如 GSM8K）上，继续思考比例低至 2.6% 和 0.3%，有效避免不必要计算，验证了自适应分配能力。
*   **安全对齐改进:** 在安全基准（如 BeaverTails, HarmfulQA）上，无害率显著提升，例如 1.5B 模型在 HarmfulQA 上从 61.7% 提升至 83.4%（+21.7%），表明减少不必要推理有助于降低不安全输出风险。
*   **实验设置合理性:** 实验覆盖多个数学推理基准（MATH500, AIME2024 等）和安全基准，模型规模包括 1.5B 和 7B，基准选择全面；与其他长度控制方法（如 S1, ThinkPrune）相比，ASRR 在相同 token 预算下实现更高准确率，但未在更大规模模型或不同架构上广泛验证，泛化性有待进一步探索。

## Further Thoughts

论文揭示的‘内部自恢复机制’启发了我思考是否可以通过预训练阶段设计特定数据分布，进一步增强模型对任务难度的隐式感知能力，从而减少推理阶段的计算需求；此外，动态长度惩罚的准确率阈值调节机制让我联想到是否可以引入实时反馈机制，根据用户交互或任务上下文动态调整阈值，以提升效率和个性化体验。