---
title: "RM-R1: Reward Modeling as Reasoning"
pubDatetime: 2025-05-05T06:11:12+00:00
slug: "2025-05-reasoning-reward-modeling"
type: "arxiv"
id: "2505.02387"
score: 0.8290677678528908
author: "grok-3-latest"
authors: ["Xiusi Chen", "Gaotang Li", "Ziqi Wang", "Bowen Jin", "Cheng Qian", "Yu Wang", "Hongru Wang", "Yu Zhang", "Denghui Zhang", "Tong Zhang", "Hanghang Tong", "Heng Ji"]
tags: ["LLM", "Reward Modeling", "Reasoning", "Distillation", "RLHF"]
institution: ["University of Illinois Urbana-Champaign", "University of California, San Diego", "Texas A&M University", "Stevens Institute of Technology"]
description: "本文提出 RM-R1 模型家族，通过将奖励建模转化为推理任务，利用蒸馏和强化学习显著提升了奖励模型的解释性和性能，超越了更大规模的开源和商业模型。"
---

> **Summary:** 本文提出 RM-R1 模型家族，通过将奖励建模转化为推理任务，利用蒸馏和强化学习显著提升了奖励模型的解释性和性能，超越了更大规模的开源和商业模型。 

> **Keywords:** LLM, Reward Modeling, Reasoning, Distillation, RLHF

**Authors:** Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru Wang, Yu Zhang, Denghui Zhang, Tong Zhang, Hanghang Tong, Heng Ji

**Institution(s):** University of Illinois Urbana-Champaign, University of California, San Diego, Texas A&M University, Stevens Institute of Technology


## Problem Background

奖励模型（Reward Model, RM）在通过人类反馈的强化学习（RLHF）中对大型语言模型（LLM）的对齐至关重要，但传统标量奖励模型（Scalar RM）输出不透明，缺乏解释性，而生成式奖励模型（Generative RM）虽能生成文本判断，但推理过程往往浅显，难以应对复杂的偏好任务。
论文的出发点在于探索是否能将奖励建模转化为推理任务，通过引入深层推理能力提升通用领域奖励模型的解释性和性能，以解决评估多样化和复杂性挑战。

## Method

*   **核心思想:** 将奖励建模作为推理任务，通过生成结构化的推理轨迹（Reasoning Traces）增强模型对人类偏好的理解和判断能力，提出推理奖励模型（Reasoning Reward Models, REAS RMS）的新范式。
*   **具体实现:** 开发了 RM-R1 模型家族，采用两阶段训练流程：
    *   **蒸馏阶段（Distillation）:** 从强模型（如 Claude-3.7-Sonnet 和 OpenAI-O3）合成高质量推理轨迹，对基础指令模型（如 Qwen-2.5-Instruct）进行监督微调（Supervised Fine-Tuning, SFT），以赋予初步推理能力。
    *   **强化学习阶段（RL with Verifiable Rewards, RLVR）:** 采用群组相对策略优化（Group Relative Policy Optimization, GRPO）进行强化学习，进一步优化模型的推理和判断能力，奖励函数基于正确性（Correctness）设计为二元奖励（+1 或 -1）。
*   **任务分类与定制化推理:** 引入‘Chain-of-Rubrics’（CoR）提示框架，将任务分为‘聊天’（Chat）和‘推理’（Reasoning）两类，针对‘聊天’任务生成评估标准（Rubrics）和理由，针对‘推理’任务则先自行解决问题再评估候选答案，确保推理过程的针对性和逻辑性。
*   **关键创新:** 不依赖简单的数值分数，而是利用语言模型的生成能力输出详细的推理过程和判断，显著提升解释性，同时通过任务分类和两阶段训练优化模型在不同领域的适应性。

## Experiment

*   **有效性:** RM-R1 模型在多个基准数据集（RewardBench, RM-Bench, RMB）上取得了最先进或接近最先进的性能，例如 RM-R1-Qwen-Instruct-32B 在 RewardBench 上整体准确率达 92.9%，超越 GPT-4o（86.7%）和 Llama3.1-405B（84.1%），最高提升达 13.8%；在 RM-Bench 上，RM-R1-DeepSeek-Distilled-Qwen-32B 在数学和代码领域准确率分别达 91.8% 和 74.1%，显著优于先前最高水平。
*   **实验设置合理性:** 实验覆盖了不同规模模型（7B 到 32B）、不同训练策略（Instruct 和 DeepSeek-Distilled 基础模型）以及多个领域（聊天、安全、数学、代码），并通过消融实验验证了蒸馏、任务分类和强化学习各组件的重要性。
*   **数据效率:** RM-R1 展现出高数据效率，仅用 8.7K 样本进行蒸馏即可达到竞争性性能，而 DeepSeek-Distilled 模型使用 800K 样本，显示训练策略的高效性。
*   **局限性:** 实验未深入探讨模型在极端边缘案例或多模态任务中的表现，可能存在适用性限制。

## Further Thoughts

将奖励建模转化为推理任务的范式转变令人启发，这种思路可推广至其他需要解释性的 AI 系统，如自动评估或决策支持工具；任务分类与定制化推理策略提示我们是否能进一步细化分类维度（如按难度或文化背景）以应对更复杂偏好；蒸馏与强化学习结合的训练框架也可能适用于其他生成式任务，如代码生成或多轮对话；此外，增加推理时计算预算显著提升性能的发现，启发我们在实际应用中动态调整资源以优化模型表现。