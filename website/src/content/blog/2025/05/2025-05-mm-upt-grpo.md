---
title: "Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO"
pubDatetime: 2025-05-28T15:11:16+00:00
slug: "2025-05-mm-upt-grpo"
type: "arxiv"
id: "2505.22453"
score: 0.7822917568837624
author: "grok-3-latest"
authors: ["Lai Wei", "Yuting Li", "Chen Wang", "Yue Wang", "Linghe Kong", "Weiran Huang", "Lichao Sun"]
tags: ["LLM", "Multi-Modal Learning", "Unsupervised Learning", "Reinforcement Learning", "Reasoning", "Synthetic Data"]
institution: ["Shanghai Jiao Tong University", "Shanghai Innovation Institute", "Zhongguancun Academy", "Lehigh University"]
description: "本文提出 MM-UPT 框架，通过 GRPO 算法和多数投票自奖励机制，实现多模态大语言模型在无监督条件下的推理能力自改进，显著提升性能并探索合成数据潜力。"
---

> **Summary:** 本文提出 MM-UPT 框架，通过 GRPO 算法和多数投票自奖励机制，实现多模态大语言模型在无监督条件下的推理能力自改进，显著提升性能并探索合成数据潜力。 

> **Keywords:** LLM, Multi-Modal Learning, Unsupervised Learning, Reinforcement Learning, Reasoning, Synthetic Data

**Authors:** Lai Wei, Yuting Li, Chen Wang, Yue Wang, Linghe Kong, Weiran Huang, Lichao Sun

**Institution(s):** Shanghai Jiao Tong University, Shanghai Innovation Institute, Zhongguancun Academy, Lehigh University


## Problem Background

多模态大语言模型（MLLMs）在后训练阶段的改进通常依赖于监督微调（SFT）或强化学习（RL），但这些方法需要大量昂贵的人工标注数据，随着任务复杂性增加，数据获取变得不可持续。
本文旨在解决这一关键问题：探索在完全无外部监督的情况下，利用无标签数据实现 MLLMs 推理能力的持续自改进。

## Method

*   **核心思想:** 提出 MM-UPT（Multi-Modal Unsupervised Post-Training）框架，通过在线强化学习算法 GRPO 实现无监督后训练，利用自奖励机制替代传统监督信号。
*   **具体实现:** 
    *   基于 GRPO 算法，模型针对无标签多模态输入（如图像和问题）生成多个响应。
    *   通过多数投票（Majority Voting）从多个响应中选出最一致的答案作为伪标签（Pseudo-Label），并据此计算奖励信号，奖励与多数答案一致的响应，惩罚不一致的响应。
    *   使用组归一化奖励（Group-Normalized Rewards）估计优势（Advantage），避免单独价值模型的计算开销，并通过 KL 散度约束限制模型偏离参考策略。
    *   此外，探索合成数据生成策略，包括上下文合成（In-Context Synthesizing，利用图像、问题和答案生成新问题）和直接合成（Direct Synthesizing，仅基于图像生成问题），以扩展无标签数据来源。
*   **关键特点:** 不依赖外部标注或奖励模型，训练过程简单且可在线迭代，支持模型持续自改进。

## Experiment

*   **有效性:** MM-UPT 在 Qwen2.5-VL-7B 模型上显著提升多模态推理能力，例如在 MathVista 数据集上从 66.3% 提升至 72.9%，在 We-Math 上从 62.9% 提升至 68.7%，平均提升约 3-7%。
*   **对比优势:** 相较于其他无监督基线（如 SRLM, LMSI, Genixer, STIC），MM-UPT 表现更优，甚至接近监督方法（如 GRPO, SFT）的性能。
*   **实验设置合理性:** 实验覆盖两种无标签数据场景：人类创建问题（无答案）和模型自生成问题，测试了多个基准数据集（如 MathVision, MathVista），并在不同规模模型（如 Qwen2.5-VL-3B, ThinkLite-VL-7B）上验证了方法的普适性。
*   **合成数据潜力:** 使用合成数据训练的模型性能接近甚至超过人类问题训练结果，例如直接合成策略在 GeoQA 数据集上平均提升 6.5%。
*   **局限性:** 在模型对数据集缺乏先验知识时（如 ThinkLite-11K），多数投票可能导致性能下降，平均准确率从 49.47% 降至 44.11%，表明方法适用性依赖于模型初始能力。

## Further Thoughts

多数投票作为自奖励机制的简洁性启发了我，可以将其推广至其他无监督学习任务，如文本生成或对话系统，通过聚合多响应提升一致性；此外，合成数据的成功应用提示是否可以通过更复杂的生成策略（如结合多模型协作或对抗生成）进一步提升数据质量；最后，多数投票的局限性让我思考是否能设计动态奖励机制，例如基于语义聚类或置信度调整奖励，以适应模型能力不足的场景。