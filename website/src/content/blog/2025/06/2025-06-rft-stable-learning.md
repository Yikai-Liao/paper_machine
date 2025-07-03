---
title: "Reinforcement Fine-Tuning Enables MLLMs Learning Novel Tasks Stably"
pubDatetime: 2025-06-30T04:15:01+00:00
slug: "2025-06-rft-stable-learning"
type: "arxiv"
id: "2506.23508"
score: 0.782317323356401
author: "grok-3-latest"
authors: ["Zhihao Zhang", "Qiaole Dong", "Qi Zhang", "Jun Zhao", "Enyu Zhou", "Zhiheng Xi", "Senjie Jin", "Xiaoran Fan", "Yuhao Zhou", "Yanwei Fu", "Tao Ji", "Tao Gui", "Xuanjing Huang"]
tags: ["LLM", "Multimodal Learning", "Fine-Tuning", "Reinforcement Learning", "Catastrophic Forgetting"]
institution: ["Fudan University", "Shanghai Artificial Intelligence Laboratory"]
description: "本文通过拼图任务揭示强化微调（RFT）在多模态大语言模型中实现稳定新任务学习的能力，并从数据分布和学习动力学视角解释了监督微调（SFT）与 RFT 在遗忘行为上的差异。"
---

> **Summary:** 本文通过拼图任务揭示强化微调（RFT）在多模态大语言模型中实现稳定新任务学习的能力，并从数据分布和学习动力学视角解释了监督微调（SFT）与 RFT 在遗忘行为上的差异。 

> **Keywords:** LLM, Multimodal Learning, Fine-Tuning, Reinforcement Learning, Catastrophic Forgetting

**Authors:** Zhihao Zhang, Qiaole Dong, Qi Zhang, Jun Zhao, Enyu Zhou, Zhiheng Xi, Senjie Jin, Xiaoran Fan, Yuhao Zhou, Yanwei Fu, Tao Ji, Tao Gui, Xuanjing Huang

**Institution(s):** Fudan University, Shanghai Artificial Intelligence Laboratory


## Problem Background

多模态大语言模型（MLLMs）在后训练阶段常通过监督微调（SFT）和强化微调（RFT）适应下游任务，但这些方法对模型预训练知识的影响尚未明确。
本文通过引入拼图任务（Jigsaw Puzzles）作为预训练数据中未包含的新任务，研究 SFT 和 RFT 在学习新任务时对已有知识的遗忘行为（Catastrophic Forgetting），旨在解决为何 SFT 导致严重遗忘而 RFT 能稳定学习的问题。

## Method

*   **核心思想:** 对比监督微调（SFT）和强化微调（RFT）在学习新任务时的行为，探索数据分布对遗忘的影响。
*   **监督微调（SFT）:** 基于静态的人工标注数据，通过教师强制（Teacher Forcing）方式快速学习新任务，直接优化模型对标注答案的对数似然（Log-Likelihood），强调快速收敛。
*   **强化微调（RFT）:** 使用 Group Relative Policy Optimization (GRPO) 算法，通过奖励驱动的采样生成数据（Rollouts），并根据规则化奖励（包括命中奖励、准确率奖励和格式奖励）自适应调整生成数据的似然权重，强调探索和稳定性。
*   **混合方法:** 利用 RFT 生成的正确 Rollouts 作为 SFT 的训练数据，验证数据分布而非算法本身对遗忘行为的影响。
*   **理论分析:** 基于学习动力学（Learning Dynamics）理论，分析 SFT 和 RFT 数据在模型输出空间中的概率分布差异，揭示 RFT 数据更贴近模型高概率区域，从而减少对已有知识的干扰。

## Experiment

*   **有效性:** SFT 在拼图任务上快速达到高准确率（1 个 epoch 后为 62%），但在已有任务上性能下降明显（如 RefCOCO val 下降 12.6%），表现出灾难性遗忘；RFT（GRPO）学习较慢（10 个 epoch 后为 66%），但在已有任务上性能稳定（RefCOCO val 仅下降 0.39%）。
*   **创新性验证:** 使用 RFT 生成的 Rollouts 进行 SFT 训练后，遗忘显著减少（RefCOCO val 仅下降 0.04%），新任务准确率仍为 66%，证明数据分布对遗忘的关键影响。
*   **实验设置:** 基于 Qwen2.5-VL-3B 模型，在 COCO 2014 数据集上构建拼图任务，并通过多个基准（如 RefCOCO, DocVQA, MME）评估新任务和已有任务表现，设置全面但局限于单一模型和任务（3x3 拼图），泛化性待验证。
*   **分析深度:** 通过困惑度（Perplexity）分析数据分布差异，RFT 数据更贴近模型高概率区域，解释了其稳定性。

## Further Thoughts

数据分布对遗忘行为的影响是一个关键启发，未来可以探索通过结合 RFT 的探索能力和 SFT 的高效性，设计更贴近模型输出分布的数据生成策略，以优化后训练过程；此外，学习动力学理论为理解知识干扰提供了新视角，或许可以基于此开发更智能的持续学习算法，减少新旧任务间的冲突。