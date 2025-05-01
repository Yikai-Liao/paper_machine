---
title: "Reinforced MLLM: A Survey on RL-Based Reasoning in Multimodal Large Language Models"
pubDatetime: 2025-05-01T15:52:24Z
slug: "2025-04-rl-mllm-reasoning"
type: "arxiv"
id: "2504.21277"
score: 0.8866336442940691
author: "grok-3-latest"
authors: ["Guanghao Zhou", "Panjia Qiu", "Cen Chen", "Jie Wang", "Zheming Yang", "Jian Xu", "Minghui Qiu"]
tags: ["LLM", "MLLM", "Reinforcement Learning", "Reasoning", "Cross-Modal Alignment"]
institution: ["East China Normal University", "ByteDance"]
description: "本文系统综述了强化学习（RL）在多模态大语言模型（MLLMs）推理中的应用，分析了算法设计、奖励机制和应用场景，揭示了提升跨模态推理能力和泛化性的有效路径。"
---

> **Summary:** 本文系统综述了强化学习（RL）在多模态大语言模型（MLLMs）推理中的应用，分析了算法设计、奖励机制和应用场景，揭示了提升跨模态推理能力和泛化性的有效路径。 

> **Keywords:** LLM, MLLM, Reinforcement Learning, Reasoning, Cross-Modal Alignment
> **Recommendation Score:** 0.8866336442940691

**Authors:** Guanghao Zhou, Panjia Qiu, Cen Chen, Jie Wang, Zheming Yang, Jian Xu, Minghui Qiu
**Institution(s):** East China Normal University, ByteDance

## Problem Background

多模态大语言模型（MLLMs）在扩展大语言模型（LLMs）能力以处理视觉、音频和视频等多模态数据方面取得了显著进展，但其在跨模态推理中的表现仍面临挑战，尤其是在复杂任务中整合多模态信息和实现自适应推理的能力不足。
传统的监督微调（SFT）方法存在标注成本高和灾难性遗忘等问题，而强化学习（RL）通过优化推理路径和奖励机制被认为是提升MLLMs推理能力和泛化性的有效途径。

## Method

*   **核心范式:** 论文综述了RL在MLLMs推理中的两大主要范式：
    *   **基于价值的方法（Value-Based Methods）**：如Proximal Policy Optimization (PPO)，通过逐步奖励分配和价值函数估计，精确优化推理过程中的每一步决策，适合复杂推理任务，但训练成本较高且在长链推理中可能面临稳定性问题。优化策略包括价值预训练和解耦优势估计（Decoupled-GAE）以减少偏差和方差。
    *   **无价值方法（Value-Free Methods）**：如Group Relative Policy Optimization (GRPO)，通过轨迹级奖励简化计算，依赖组内相对奖励评估生成质量，适合长链推理任务，但可能面临熵崩溃和奖励噪声问题。改进包括动态采样、剪切策略调整（如DAPO）和去除归一化偏差（如Dr.GRPO）。
*   **奖励机制设计:** 分为结果导向奖励机制（Outcome Reward Mechanism, ORM）和过程导向奖励机制（Process Reward Mechanism, PRM）：
    *   ORM关注最终输出的正确性，简单易实现，但存在时间信用分配问题和稀疏奖励导致的低样本效率。
    *   PRM强调中间推理步骤的质量，通过逻辑一致性或信息完整性提供细粒度监督，提升模型可解释性，但设计复杂且缺乏标准化评估标准。
*   **训练效率优化:** 包括课程学习（Curriculum Learning），通过从易到难的任务顺序提升收敛速度；数据高效学习，如优先采样（Prioritized Sampling）和高质量样本选择，减少计算开销；以及通过KL正则化等方法缓解灾难性遗忘问题。
*   **跨模态整合:** 针对多模态任务，提出任务导向、跨模态交互和课程式奖励策略，以增强视觉、语言和时间信息的对齐和推理能力。

## Experiment

*   **有效性:** RL-based MLLMs在多个基准数据集（如MathVista、MMMU-Val）上表现出显著优势，例如Vision-R1-7B在MathVista上的得分达到74.9%，接近闭源模型GPT-4o的77.3%，尤其在泛化能力（OOD数据）和复杂推理任务中优于非RL方法。
*   **优越性:** 相较于传统SFT方法，RL方法通过优化推理路径和奖励机制，显著提升了模型在多模态推理任务中的自适应性和准确性，尤其在数学、科学和图表推理任务中表现突出。
*   **局限性与合理性:** 实验设置覆盖了数学、科学、图表等多领域基准，但对动态环境或实时交互任务的评估不足，可能限制了对模型在真实场景中表现的全面理解；此外，部分任务（如HallBench）提升幅度有限，可能与奖励稀疏性或跨模态协调不足有关。
*   **开销:** RL方法训练成本较高，尤其是在基于价值的方法中需要同时训练策略和价值模型，而无价值方法通过简化计算降低了部分开销，但仍需大量高质量样本支持。

## Further Thoughts

论文中跨模态奖励设计和泛化能力的讨论启发了我思考是否可以引入外部知识库或预训练模态特定模型（如目标检测模型）生成中间奖励信号，以增强奖励密度和跨模态对齐效果；此外，探索自适应奖励机制，根据任务难度或模态特性动态调整奖励权重，可能进一步避免模型陷入局部最优或过拟合特定模态数据。