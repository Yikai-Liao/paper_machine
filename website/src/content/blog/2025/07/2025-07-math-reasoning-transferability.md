---
title: "Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning"
pubDatetime: 2025-07-01T05:23:05+00:00
slug: "2025-07-math-reasoning-transferability"
type: "arxiv"
id: "2507.00432"
score: 0.9000959633075564
author: "grok-3-latest"
authors: ["Maggie Huan", "Yuetai Li", "Tuney Zheng", "Xiaoyu Xu", "Seungone Kim", "Minxin Du", "Radha Poovendran", "Graham Neubig", "Xiang Yue"]
tags: ["LLM", "Reasoning", "Transferability", "Reinforcement Learning", "Supervised Fine-Tuning"]
institution: ["Carnegie Mellon University", "University of Pennsylvania", "University of Washington", "M-A-P", "The Hong Kong Polytechnic University"]
description: "本文通过大规模评估和控制实验揭示了强化学习（RL）微调在提升大型语言模型数学推理能力的同时，能更好地转移到其他推理和非推理任务，相较于监督微调（SFT）具有显著优势，并从潜在空间和输出分布角度解释了这种差异。"
---

> **Summary:** 本文通过大规模评估和控制实验揭示了强化学习（RL）微调在提升大型语言模型数学推理能力的同时，能更好地转移到其他推理和非推理任务，相较于监督微调（SFT）具有显著优势，并从潜在空间和输出分布角度解释了这种差异。 

> **Keywords:** LLM, Reasoning, Transferability, Reinforcement Learning, Supervised Fine-Tuning

**Authors:** Maggie Huan, Yuetai Li, Tuney Zheng, Xiaoyu Xu, Seungone Kim, Minxin Du, Radha Poovendran, Graham Neubig, Xiang Yue

**Institution(s):** Carnegie Mellon University, University of Pennsylvania, University of Washington, M-A-P, The Hong Kong Polytechnic University


## Problem Background

大型语言模型（LLMs）在数学推理任务上的表现显著提升，但这种能力的提升是否能转移到其他推理任务（如科学问答、编码、逻辑推理）和非推理任务（如对话问答、指令跟随）尚不明确。
论文旨在探究数学推理能力的可转移性（Transferability），解决模型在特定领域优化后可能出现的泛化能力下降问题，特别是在现实世界中需要广泛语言和常识能力的应用场景中。

## Method

*   **大规模模型评估：** 评估超过 20 个开源推理模型在数学推理、其他推理和非推理任务上的表现，提出‘转移性指数’（Transferability Index, TI）来量化从数学推理到其他领域的性能转移情况。TI 通过计算模型在目标任务组上的相对性能增益，并与数学任务的增益进行标准化比较，得出正值表示正向转移，负值表示性能下降。
*   **控制实验：** 使用 Qwen3-14B 模型作为基础，在一个包含 47K 高质量数学问题的数据集上分别进行监督微调（Supervised Fine-Tuning, SFT）和强化学习（Reinforcement Learning, RL）微调。SFT 通过拒绝采样（Rejection Sampling）从 Qwen3-32B 模型生成思维链（Chain-of-Thought, CoT）作为训练目标，旨在让模型模仿教师模型的推理过程；RL 采用 GRPO 框架，以答案正确性作为奖励信号，通过策略优化鼓励模型探索有效的推理路径。两种方法在相同数据和模型基础上进行对比，确保变量控制。
*   **诊断分析：** 运用主成分分析（PCA）量化微调前后模型潜在空间（Latent Space）的表征漂移，计算各层隐藏状态在主要成分上的投影变化，并通过欧几里得距离衡量整体漂移程度；同时使用 KL 散度（KL Divergence）测量模型输出概率分布的变化，并通过词元排名变化（Token Rank Shift）分析微调后词元分布的具体调整情况。这些分析旨在从内部表征和外部输出的角度解释转移性差异的原因。

## Experiment

*   **有效性：** 实验结果表明，RL 微调的模型在数学推理任务（如 AIME24、MATH500）上显著提升性能，同时在其他推理任务（如 GPQA、LiveCodeBench）和非推理任务（如 CoQA、IFEval）上展现正向转移（TI > 0），例如 UniReason-Qwen3-14B (RL) 在非推理任务平均性能提升 24.0%。相比之下，SFT 微调的模型虽然在数学任务上有所提升，但在非推理任务上常出现负向转移（TI < 0），如 UniReason-Qwen3-14B-_think_ (SFT) 在非推理任务上性能下降 41.2%。
*   **表征稳定性：** PCA 分析显示 RL 模型的潜在空间漂移较小（例如在数学任务上漂移值为 8.5），表明其内部表征更稳定；SFT 模型漂移较大（例如在非推理任务上漂移值高达 113.7），可能导致灾难性遗忘。KL 散度和词元排名分析进一步表明 RL 模型对任务相关词元的调整更具选择性（平均词元排名变化仅为 0.98），而 SFT 模型调整了许多无关词元（平均变化高达 10.6）。
*   **实验设置合理性：** 实验覆盖了多种任务类别（数学推理、其他推理、非推理）、模型规模（7B 到 32B）和模型家族（Qwen、Llama），数据集结合了低难度和高质量样本，确保代表性。评估使用了多个基准测试和准确率指标，增强了结果的可靠性。

## Further Thoughts

RL 微调通过奖励机制优化推理路径而非静态模仿，可能更接近人类学习中的探索与反馈模式，未来可探索其在其他任务（如情感分析、生成任务）中的应用；潜在空间分析（PCA 和 KL 散度）为理解模型泛化机制提供了新视角，可扩展到跨领域任务评估；转移性指数（TI）作为量化指标，未来可用于指导多任务模型设计和训练策略优化。