---
title: "When to Continue Thinking: Adaptive Thinking Mode Switching for Efficient Reasoning"
pubDatetime: 2025-05-21T11:41:39+00:00
slug: "2025-05-adaptive-reasoning-efficiency"
type: "arxiv"
id: "2505.15400"
score: 0.8211307104211535
author: "grok-3-latest"
authors: ["Xiaoyun Zhang", "Jingqing Ruan", "Xing Ma", "Yawen Zhu", "Haodong Zhao", "Hao Li", "Jiansong Chen", "Ke Zeng", "Xunliang Cai"]
tags: ["LLM", "Reasoning", "Efficiency", "Adaptive Learning", "Reinforcement Learning"]
institution: ["Meituan"]
description: "本文提出自适应自我恢复推理框架（ASRR），利用‘内部自我恢复机制’，使大型推理模型根据问题难度动态分配推理资源，显著降低计算开销并保持性能，同时提升安全性。"
---

> **Summary:** 本文提出自适应自我恢复推理框架（ASRR），利用‘内部自我恢复机制’，使大型推理模型根据问题难度动态分配推理资源，显著降低计算开销并保持性能，同时提升安全性。 

> **Keywords:** LLM, Reasoning, Efficiency, Adaptive Learning, Reinforcement Learning

**Authors:** Xiaoyun Zhang, Jingqing Ruan, Xing Ma, Yawen Zhu, Haodong Zhao, Hao Li, Jiansong Chen, Ke Zeng, Xunliang Cai

**Institution(s):** Meituan


## Problem Background

大型推理模型（LRMs）通过长推理链在复杂任务上表现出色，但经常在简单任务上产生冗余推理，导致显著的计算开销。
作者发现模型在‘无推理模式’下也能通过‘内部自我恢复机制’隐式补充推理步骤达到较高准确率，但当前模型在难度感知和推理资源分配上存在局限：对困难问题推理不足导致准确率下降，对简单问题过度推理造成资源浪费。
因此，论文试图解决如何让 LRMs 根据问题难度动态调整推理长度，实现高效推理并保持性能。

## Method

*   **核心思想:** 提出自适应自我恢复推理框架（ASRR），利用模型的‘内部自我恢复机制’，通过抑制不必要推理并允许隐式恢复，根据问题难度动态分配推理资源。
*   **具体实现:** 
    *   **显式推理抑制与隐式自我恢复:** 在‘无推理模式’下，通过在输入提示中添加特殊前缀（如‘Okay, I have finished thinking.’）抑制显式推理，鼓励模型对简单问题直接生成答案，同时允许在困难问题上通过隐式推理路径恢复必要的推理步骤。
    *   **动态长度惩罚（DLP）:** 引入基于准确率阈值的奖励调节机制，将训练数据分组并计算组内平均准确率，仅当准确率达到预设阈值时激活长度惩罚，且惩罚强度随准确率提升逐步增加，避免过早优化长度导致错误输出，同时在高准确率时有效抑制过度推理。
*   **训练方式:** 通过强化学习（RL）训练模型，使其学习根据任务难度自适应调整推理深度。
*   **关键点:** 方法不依赖额外的控制结构或复杂提示工程，而是通过模型自身的难度感知能力实现高效推理。

## Experiment

*   **有效性:** 在多个数学推理基准（如 MATH500, AIME2024, AMC2023, Olympiad Bench, GSM8K）上，与基线 GRPO 相比，ASRR 在 1.5B 模型上平均减少 32.5% 推理长度，在 7B 模型上减少 25.7%，而准确率（pass@1）仅下降 1.2% 和 0.6%，表明效率提升显著且性能损失极小。
*   **难度感知能力:** ASRR 使模型能根据问题难度灵活调整推理深度，例如在高难度 AIME 任务上 Continue-Thinking 比率高达 80.6%（1.5B）和 81.5%（7B），而在低难度 GSM8K 上仅为 2.6% 和 0.3%。
*   **安全性提升:** 在安全对齐基准（如 BeaverTails, HarmfulQA）上，ASRR 显著提高无害率，例如 1.5B 模型在 HarmfulQA 上从 61.7% 提升至 83.4%（+21.7%）。
*   **实验设置合理性:** 实验覆盖多种难度和类型任务，基准数据集选择全面，模型规模从 1.5B 到 7B 具有代表性，并与多种长度控制方法（如 S1, L1, ThinkPrune, DPO）对比，验证了方法的优越性；不足之处在于未在更大规模模型或更多架构上测试，可能影响普适性结论。

## Further Thoughts

论文揭示的‘内部自我恢复机制’非常具有启发性，表明模型即使在显式推理被抑制的情况下也能通过隐式推理路径补充必要步骤，这提示我们未来可以通过设计更精细的提示或训练策略进一步挖掘模型的隐式推理能力，例如针对不同任务类型定制推理抑制与恢复策略；此外，动态长度惩罚的准确率阈值调节机制也启发我们探索自适应阈值调整算法，根据任务特性或实时性能反馈动态优化阈值，而非预设固定值。