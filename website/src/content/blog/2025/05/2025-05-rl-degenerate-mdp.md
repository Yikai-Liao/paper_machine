---
title: "RL in Name Only? Analyzing the Structural Assumptions in RL post-training for LLMs"
pubDatetime: 2025-05-19T19:57:15+00:00
slug: "2025-05-rl-degenerate-mdp"
type: "arxiv"
id: "2505.13697"
score: 0.7405691053306718
author: "grok-3-latest"
authors: ["Soumya Rani Samineni", "Durgesh Kalwar", "Karthik Valmeekam", "Kaya Stechly", "Subbarao Kambhampati"]
tags: ["LLM", "Reinforcement Learning", "Supervised Fine-Tuning", "MDP Modeling", "Reasoning"]
institution: ["Arizona State University (SCAI)"]
description: "本文通过理论和实验揭示了当前强化学习（RL）后训练框架在特定 MDP 假设下的局限性，证明其与简单监督微调（SFT）方法等价，并指出响应长度增加是结构偏见的副作用，而非推理能力提升。"
---

> **Summary:** 本文通过理论和实验揭示了当前强化学习（RL）后训练框架在特定 MDP 假设下的局限性，证明其与简单监督微调（SFT）方法等价，并指出响应长度增加是结构偏见的副作用，而非推理能力提升。 

> **Keywords:** LLM, Reinforcement Learning, Supervised Fine-Tuning, MDP Modeling, Reasoning

**Authors:** Soumya Rani Samineni, Durgesh Kalwar, Karthik Valmeekam, Kaya Stechly, Subbarao Kambhampati

**Institution(s):** Arizona State University (SCAI)


## Problem Background

近年来，大型语言模型（LLMs）在后训练阶段广泛采用强化学习（RL）方法（如 GRPO）以提升推理能力，尤其在 DeepSeek R1 发布后受到关注。然而，作者质疑这种 RL 框架的实际作用，指出其依赖的马尔可夫决策过程（MDP）建模存在结构性假设（如状态为动作序列、奖励仅在终止状态分配且均匀分布），可能导致 RL 应用变得‘名不副实’，未能真正带来新的推理能力，同时响应长度增加可能只是训练偏见的副作用，而非推理提升的结果。

## Method

* **理论分析**：作者从当前 LLM-MDP 建模的两个核心结构假设入手，即状态被定义为生成的 token 序列（动作历史），奖励仅在终止状态由外部验证器分配并均匀分布到每个 token 上。通过推导 GRPO 的目标函数，证明在这些假设下，GRPO 的更新过程可以简化为一种加权监督学习形式，即过滤迭代监督微调（Filtered Iterative Supervised Fine-Tuning, F-ISFT）。具体方法是将 GRPO 目标函数分解为正向（正确响应）和负向（错误响应）两部分，展示其与 SFT 的等价性。
* **实证验证**：设计实验对比 GRPO 与多种 F-ISFT 变体（仅正样本、仅负样本、正负样本结合）的性能，使用 Qwen-2.5 模型（0.5B 和 1.5B 参数规模）在 GSM8K 和 Countdown 数据集上测试。此外，分析 GRPO 中均匀优势分配和长度缩放对响应长度的影响，验证长度增加是否为结构偏见导致。
* **核心思想**：通过理论推导和实验对比，揭示 RL 在当前 MDP 建模下的冗余性，强调结构假设对训练结果的深远影响。

## Experiment

* **性能对比**：在 GSM8K 数据集上，GRPO 和 F-ISFT（正负样本结合）性能相当，Qwen-2.5-0.5B 准确率从 0.6% 提升至 65%，1.5B 从 22.7% 提升至 85%，F-ISFT 的训练动态和测试准确率与 GRPO 几乎一致；在 Countdown 数据集上，方法间差异稍大，但 F-ISFT（正样本）在部分情况下优于 GRPO，表明 RL 的性能提升并不显著优于 SFT。
* **响应长度分析**：实验显示 GRPO 训练中响应长度先减少后增加，作者归因于均匀优势分配和长度缩放的结构偏见，而非推理能力提升。
* **实验设置合理性**：数据集选择覆盖数学推理和逻辑游戏，任务难度适中；模型规模从小到中，兼顾计算成本；超参数一致，确保对比公平。但实验未涉及更大规模模型或更多样化任务，可能限制结论泛化性。

## Further Thoughts

论文启发我们重新思考 MDP 建模在 LLM 后训练中的设计，探索更复杂的建模方式（如引入中间状态奖励或非均匀信用分配）以充分发挥 RL 潜力；同时，简单 SFT 方法在特定场景下可能替代复杂 RL 框架，提示我们在资源有限时优先考虑高效方法；此外，响应长度增加被误解为推理提升的现象提醒我们，需谨慎设计训练目标和评估指标，避免被表面现象误导，深入分析模型行为的根本驱动因素。