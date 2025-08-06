---
title: "Decomposing the Entropy-Performance Exchange: The Missing Keys to Unlocking Effective Reinforcement Learning"
pubDatetime: 2025-08-04T10:08:10+00:00
slug: "2025-08-entropy-performance-rlvr"
type: "arxiv"
id: "2508.02260"
score: 0.5959345685026466
author: "grok-3-latest"
authors: ["Jia Deng", "Jie Chen", "Zhipeng Chen", "Wayne Xin Zhao", "Ji-Rong Wen"]
tags: ["LLM", "Reinforcement Learning", "Entropy Dynamics", "Reward Shaping", "Reasoning"]
institution: ["Gaoling School of Artificial Intelligence, Renmin University of China"]
description: "本文通过细粒度分析 RLVR 中的熵-性能交换机制，提出基于困惑度和位置的动态奖励塑造方法，显著提升了大型语言模型在数学推理任务上的性能。"
---

> **Summary:** 本文通过细粒度分析 RLVR 中的熵-性能交换机制，提出基于困惑度和位置的动态奖励塑造方法，显著提升了大型语言模型在数学推理任务上的性能。 

> **Keywords:** LLM, Reinforcement Learning, Entropy Dynamics, Reward Shaping, Reasoning

**Authors:** Jia Deng, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Ji-Rong Wen

**Institution(s):** Gaoling School of Artificial Intelligence, Renmin University of China


## Problem Background

强化学习与可验证奖励（RLVR）是增强大型语言模型（LLMs）推理能力的重要方法，但熵（entropy）与性能（performance）之间的交换关系缺乏细粒度理解。
现有研究往往将训练过程视为整体，忽略了熵动态在不同训练阶段、样本和 token 层面的具体影响，导致优化策略不够精准。
论文旨在通过系统性分析熵-性能交换机制，揭示其在不同粒度上的作用规律，从而解决如何更有效地利用熵动态优化 RLVR 训练的问题。

## Method

*   **分析框架:** 将 RLVR 训练过程分为上升阶段（rising stage）和平台阶段（plateau stage），并在阶段级（stage-level）、样本级（instance-level）和 token 级（token-level）三个粒度上系统分析熵-性能交换机制。
*   **基础算法:** 采用 GRPO（一种针对推理任务设计的强化学习变体）作为核心训练框架，通过对策略分布的熵动态进行量化分析，揭示性能提升的驱动因素。
*   **细粒度指标:** 引入 token 级别的熵（Entropy）、梯度（Gradient）和性能影响（Performance Impact）指标，用于量化 token 在训练中的不确定性和对推理准确性的贡献，帮助识别高学习潜力的 token。
*   **奖励塑造方法:** 基于分析结果，提出两种动态调整 token 级别优势（advantage）的奖励塑造策略：
    *   **基于困惑度（PPL-based Advantage Shaping）:** 计算每个样本的标准化对数困惑度（log-PPL），对低困惑度样本的 token 赋予更高优势权重，聚焦于语义更连贯、模型更确定的推理路径。
    *   **基于位置（Position-based Advantage Shaping）:** 对序列后部的 token 施加位置奖励（positional bonus），根据 token 在序列中的相对位置动态调整优势，强调最终决策过程中的关键 token。
*   **实现细节:** 两种方法通过调整 token 优势值来影响策略更新方向，确保模型学习集中在高学习潜力的样本和 token 上，同时避免过快收敛导致探索不足。

## Experiment

*   **数据集与基准:** 使用 STILL-3 数据集（包含 90K 高质量数学问题）进行 RL 训练，并在多个数学推理基准（AIME 2024/2025, AMC 2023, MATH500, MINERVA）及两个域外基准（GPQA, HumanEval）上评估性能，评估指标包括 Acc@N、Maj@N 和 Pass@N（N=8）。
*   **模型与基线:** 实验基于 Qwen2.5-7B 和 Qwen2.5-Math-7B 模型，以 GRPO 作为基线算法，比较提出的两种方法（GRPO+PPL 和 GRPO+POSITION）的效果。
*   **性能提升:** 结果显示 GRPO+PPL 和 GRPO+POSITION 在数学推理任务上显著优于基线，平均性能提升分别为 1.51%（Qwen2.5-7B）和 2.31%（Qwen2.5-Math-7B）；生成的推理过程更长，包含更多形式推理 token。
*   **熵动态分析:** 上升阶段通过减少负样本熵快速提升性能，平台阶段则聚焦于低困惑度样本中的高熵 token 优化，验证了两种方法的理论依据。
*   **合理性与局限:** 实验设置全面，涵盖多个基准和模型，指标设计合理，能有效反映推理能力提升；但在域外任务上的泛化性提升有限，表明方法可能对特定领域（如数学推理）更为有效。
*   **计算开销:** 动态奖励塑造增加了少量计算负担（如计算困惑度和位置奖励），但整体仍在可接受范围内。

## Further Thoughts

熵动态的分阶段分析为 RLVR 训练提供了新视角，启发我们可以在其他强化学习任务中探索多阶段优化策略，针对不同阶段设计差异化奖励机制。
低困惑度样本和序列后部 token 的高学习潜力提示，未来的 RL 训练可以进一步细化奖励设计，针对特定数据特征或 token 位置进行个性化优化。
动态奖励塑造策略表明静态奖励可能限制模型潜力，结合训练进度和数据特征的自适应奖励机制可能成为提升模型性能的关键方向。