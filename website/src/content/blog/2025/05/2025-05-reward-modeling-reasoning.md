---
title: "RM-R1: Reward Modeling as Reasoning"
pubDatetime: 2025-05-05T06:11:12+00:00
slug: "2025-05-reward-modeling-reasoning"
type: "arxiv"
id: "2505.02387"
score: 0.8303192989445728
author: "grok-3-latest"
authors: ["Xiusi Chen", "Gaotang Li", "Ziqi Wang", "Bowen Jin", "Cheng Qian", "Yu Wang", "Hongru Wang", "Yu Zhang", "Denghui Zhang", "Tong Zhang", "Hanghang Tong", "Heng Ji"]
tags: ["LLM", "Reward Modeling", "Reasoning", "Distillation", "Reinforcement Learning"]
institution: ["University of Illinois Urbana-Champaign", "University of California, San Diego", "Texas A&M University", "Stevens Institute of Technology"]
description: "本文提出将奖励建模作为推理任务的范式，通过推理链蒸馏和强化学习训练 RM-R1 模型，显著提升了奖励模型的解释性和性能，超越了更大规模的开源和闭源模型。"
---

> **Summary:** 本文提出将奖励建模作为推理任务的范式，通过推理链蒸馏和强化学习训练 RM-R1 模型，显著提升了奖励模型的解释性和性能，超越了更大规模的开源和闭源模型。 

> **Keywords:** LLM, Reward Modeling, Reasoning, Distillation, Reinforcement Learning

**Authors:** Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru Wang, Yu Zhang, Denghui Zhang, Tong Zhang, Hanghang Tong, Heng Ji

**Institution(s):** University of Illinois Urbana-Champaign, University of California, San Diego, Texas A&M University, Stevens Institute of Technology


## Problem Background

奖励模型（Reward Models, RMs）在大型语言模型（LLMs）的对齐中至关重要，特别是在通过人类反馈的强化学习（RLHF）中，用于提供准确的奖励信号以指导模型优化。
然而，现有奖励模型存在显著缺陷：标量奖励模型（Scalar RMs）仅输出数值分数，缺乏解释性；生成式奖励模型（Generative RMs）虽有推理痕迹，但推理过程往往肤浅，难以应对复杂的偏好判断任务。
论文提出一个核心问题：能否将奖励建模转化为推理任务，通过引入深思熟虑的推理过程来提升模型的解释性和性能？

## Method

*   **核心思想：** 将奖励建模任务重构为推理任务，通过生成详细的推理链或评估标准（Rubrics）来增强奖励模型的解释性和判断准确性。
*   **具体实现：**
    *   **推理奖励模型（Reasoning Reward Models, REAS RMS）**：提出一种新型奖励模型类别，强调在评分或判断前进行逻辑推理，生成可解释的推理过程。
    *   **两阶段训练流程：**
        1. **推理链蒸馏（Distillation of Reasoning Traces）**：利用强大的教师模型（如 Claude-3.7-Sonnet 和 OpenAI-O3）合成高质量推理链，用于初始训练，确保模型具备基本的推理能力，避免从零开始学习。
        2. **强化学习（Reinforcement Learning, RL）**：采用可验证奖励的强化学习（RLVR），通过 Group Relative Policy Optimization (GRPO) 算法进一步优化模型，使用基于正确性的奖励函数，同时引入 KL 正则化防止过拟合。
    *   **任务分类与定制化推理：** 模型首先将输入任务分类为‘聊天（Chat）’或‘推理（Reasoning）’类型，并根据类型采取不同推理策略：聊天任务生成评估标准（Rubrics）及依据，推理任务则先自行解决问题再评估候选答案。
    *   **Chain-of-Rubrics (CoR) 框架：** 为聊天任务设计结构化提示，引导模型生成评估标准、标准依据及具体评估内容，提升判断的逻辑性和透明度。
*   **关键创新：** 通过推理过程增强奖励模型的解释性，同时利用蒸馏和 RL 的协同作用提升模型在多样化任务上的泛化能力。

## Experiment

*   **有效性：** RM-R1 系列模型（7B 到 32B 参数规模）在多个基准数据集（RewardBench, RM-Bench, RMB）上表现出色，RM-R1-Qwen-Instruct-32B 在 RewardBench 上整体准确率达 92.9%，超越更大规模的开源模型（如 Llama3.1-405B）和闭源模型（如 GPT-4o）；在 RM-Bench 上，RM-R1-DeepSeek-Distilled-Qwen-32B 提升了 12.8% 的准确率。
*   **推理训练的优势：** 相比非推理方法（如纯监督微调 SFT），推理训练（Distillation + RL）显著提升性能，尤其是在复杂推理任务上，验证了推理过程对奖励建模的重要性。
*   **跨领域泛化：** 基于 Qwen-2.5-Instruct 的模型在聊天、安全和推理任务上表现均衡，而基于 DeepSeek-Distilled 的模型在推理任务上更强，表明训练数据和预训练背景对领域性能有影响。
*   **扩展性分析：** 实验表明模型规模和推理时计算预算（token 数量）均与性能呈正相关，推理奖励模型从规模扩展中获益更多。
*   **实验设置合理性：** 基准数据集覆盖聊天、安全、推理等多种任务类型和难度，数据量充足；对比基线包括标量 RM、生成式 RM 和其他推理增强 RM，覆盖面广；消融实验验证了训练流程中任务分类、蒸馏、RL 等组件的必要性。

## Further Thoughts

将奖励建模转化为推理任务的范式非常新颖，启发我们思考是否可以将其他监督任务（如分类、回归）也重构为生成式推理任务，以提升解释性和性能；此外，任务分类驱动的定制化推理策略提示我们可以在模型设计中引入任务感知机制，根据输入特性动态调整处理流程；蒸馏与 RL 协同训练的思路也可能适用于其他复杂推理任务，如代码生成或数学求解，值得进一步探索。