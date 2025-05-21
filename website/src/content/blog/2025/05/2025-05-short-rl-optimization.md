---
title: "Efficient RL Training for Reasoning Models via Length-Aware Optimization"
pubDatetime: 2025-05-18T07:46:43+00:00
slug: "2025-05-short-rl-optimization"
type: "arxiv"
id: "2505.12284"
score: 0.8512681540500151
author: "grok-3-latest"
authors: ["Danlong Yuan", "Tian Xie", "Shaohan Huang", "Zhuocheng Gong", "Huishuai Zhang", "Chong Luo", "Furu Wei", "Dongyan Zhao"]
tags: ["LLM", "Reasoning", "Reinforcement Learning", "Response Length", "Reward Design"]
institution: ["Peking University", "University of Science and Technology of China", "Microsoft Research"]
description: "本文提出 Short-RL 方法，通过三种创新奖励设计直接在强化学习训练中优化推理模型的响应长度，显著减少计算成本并保持或提升性能。"
---

> **Summary:** 本文提出 Short-RL 方法，通过三种创新奖励设计直接在强化学习训练中优化推理模型的响应长度，显著减少计算成本并保持或提升性能。 

> **Keywords:** LLM, Reasoning, Reinforcement Learning, Response Length, Reward Design

**Authors:** Danlong Yuan, Tian Xie, Shaohan Huang, Zhuocheng Gong, Huishuai Zhang, Chong Luo, Furu Wei, Dongyan Zhao

**Institution(s):** Peking University, University of Science and Technology of China, Microsoft Research


## Problem Background

大型推理模型（Large Reasoning Models, LRMs）在推理任务中表现出色，但其过长的推理路径导致训练和推理过程中计算成本高、内存占用大，效率低下，且过长路径并不总是带来性能提升；现有方法多依赖额外训练阶段或数据，无法直接应用于在线策略强化学习（on-policy RL）框架，因此需要在 RL 训练中直接优化推理长度，同时保持模型性能。

## Method

* **核心思想**：通过设计创新的奖励函数，直接在强化学习（RL）训练过程中优化大型推理模型的推理路径长度，避免额外训练阶段，同时维持或提升模型性能。
* **具体实现**：提出 Short-RL 方法，包含三种奖励设计：
  * **Correctness-Conditioned Length Reward**：仅对正确回答的样本计算长度奖励，基于正确样本的最短和最长长度（ℓ_min 和 ℓ_max）定义奖励，避免对错误回答的过度惩罚，保护模型在训练早期的探索行为。
  * **Nucleus Length Reward**：引入长度容忍度参数（τ_ℓ），对处于合理长度范围内的正确回答免除惩罚（奖励设为 0.5，与最短正确回答一致），对超出范围的回答按线性函数计算惩罚，允许模型在一定范围内灵活生成输出。
  * **Accuracy-Aware Length Reward**：引入准确率阈值（τ_acc），仅在批次准确率达到历史最高准确率减去阈值的条件时应用长度奖励，形成稀疏奖励机制，避免早期训练因长度惩罚干扰学习动态。
* **整合方式**：三种设计被统一到一个奖励函数中，通过条件性（仅正确回答）、适应性（长度容忍度）和稀疏性（准确率条件）平衡推理长度与性能。
* **关键特点**：方法直接嵌入在线策略 RL 训练，无需修改模型架构或增加额外训练阶段，通过超参数（如 τ_ℓ, τ_acc, α）调节奖励强度和应用时机。

## Experiment

* **有效性**：在逻辑推理任务（Logic-RL）中，Short-RL 实现平均推理长度减少 40%（按步长计算），同时准确率提升 14%（从 79% 到 93%）；在数学推理任务（DeepScaleR, Open-Reasoner-Zero, SimpleRL-Reason）中，推理长度分别减少 33%、11% 和 21%，性能保持或略有提升。
* **对比优越性**：相比基线方法（Standard RL, Kimi, CosFn, Efficient），Short-RL 在长度控制和准确率上均表现优越；Kimi 方法在早期训练时导致长度急剧缩短和性能崩溃（准确率接近 0），而 Short-RL 通过条件性奖励设计有效避免这一问题。
* **实验设置合理性**：实验覆盖逻辑推理和数学推理两大领域，涉及多个数据集（AIME, AMC, MATH500 等）和任务类型，评估了域内和域外泛化能力；消融研究验证了三种奖励设计的互补性，并探讨了超参数（τ_ℓ, τ_acc）的影响；但未充分测试超参数在更大范围内的鲁棒性，可能存在一定局限性。
* **计算成本**：方法未显著增加计算开销，仅通过奖励函数调整实现优化，具有较高的实用性。

## Further Thoughts

Short-RL 的条件性奖励设计（尤其是准确率感知和长度容忍度）启发了一种动态平衡探索与效率的思路，不仅适用于推理长度优化，还可能推广到其他 RL 任务中，如控制生成内容的风格或多样性；此外，是否可以通过更复杂的条件（如任务难度、模型收敛状态）进一步精细化奖励设计，或结合多目标优化在长度、准确率和多样性之间找到更优解，值得深入探索。