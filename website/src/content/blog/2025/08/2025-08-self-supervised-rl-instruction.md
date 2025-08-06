---
title: "Beyond the Trade-off: Self-Supervised Reinforcement Learning for Reasoning Models' Instruction Following"
pubDatetime: 2025-08-04T07:48:59+00:00
slug: "2025-08-self-supervised-rl-instruction"
type: "arxiv"
id: "2508.02150"
score: 0.736281861348137
author: "grok-3-latest"
authors: ["Qingyu Ren", "Qianyu He", "Bowei Zhang", "Jie Zeng", "Jiaqing Liang", "Yanghua Xiao", "Weikang Zhou", "Zeye Sun", "Fei Yu"]
tags: ["LLM", "Reasoning", "Instruction Following", "Reinforcement Learning", "Self-Supervised Learning"]
institution: ["Shanghai Key Laboratory of Data Science, College of Computer Science and Artificial Intelligence, Fudan University", "School of Data Science, Fudan University", "Ant Group"]
description: "本文提出一种自监督强化学习框架，利用推理模型内部信号显著提升指令跟随能力，同时维持推理性能，避免了对外部强模型的依赖。"
---

> **Summary:** 本文提出一种自监督强化学习框架，利用推理模型内部信号显著提升指令跟随能力，同时维持推理性能，避免了对外部强模型的依赖。 

> **Keywords:** LLM, Reasoning, Instruction Following, Reinforcement Learning, Self-Supervised Learning

**Authors:** Qingyu Ren, Qianyu He, Bowei Zhang, Jie Zeng, Jiaqing Liang, Yanghua Xiao, Weikang Zhou, Zeye Sun, Fei Yu

**Institution(s):** Shanghai Key Laboratory of Data Science, College of Computer Science and Artificial Intelligence, Fudan University, School of Data Science, Fudan University, Ant Group


## Problem Background

推理模型在复杂问题求解中表现出色，但存在推理能力与指令跟随能力之间的权衡问题，尤其是在多约束指令场景下表现不足，限制了其在现实应用中的潜力；现有方法依赖外部强模型来提升指令跟随能力，导致方法瓶颈（如学生模型受限于教师模型）和实际限制（如高成本和访问受限）。

## Method

* **核心思想**：提出一种自监督强化学习（Self-Supervised RL）框架，利用推理模型自身的内部信号提升指令跟随能力，避免对外部强模型的依赖，同时维持推理性能。
* **数据集构建与课程分解**：合成包含硬约束（如格式要求）和软约束（如语义风格）的多约束指令数据集，并通过逐步增加约束数量的课程分解（Incremental Constraint Curriculum），将复杂指令拆分为简单子任务，提供更密集的训练信号，解决奖励稀疏问题。
* **奖励建模**：针对硬约束采用程序化验证提供二元奖励（满足为1，不满足为0）；针对软约束，利用自监督数据（基于课程分解中响应差异构建正负样本）训练约束级别的二元分类奖励模型，估计响应满足约束的概率；最终将各约束的奖励聚合为样本级奖励，用于指导训练。
* **强化学习优化**：采用 GRPO 算法基于复合奖励信号优化策略模型，同时整合数学和科学领域的推理数据，确保模型通用能力的维持；训练过程高效，奖励模型推理速度优于传统方法（如 LLM-as-a-judge）。

## Experiment

* **有效性**：在多个指令跟随基准（如 IFEval, CFBench, ComplexBench）上，使用该方法的模型显著提升了硬约束和软约束下的表现，例如 R1-0528-Qwen3-8B-IF 在 IFEval 上的平均得分从 80.7 提升至 87.6，在 ComplexBench 上从 66.1 提升至 71.1。
* **推理能力维持**：在通用能力基准（如 GPQA, AIME, MMLU-Pro）上，模型推理性能基本不变，甚至在部分任务（如 FOLIO）上有所提升，例如 R1-Distill-Qwen-7B-IF 的 FOLIO 得分从 69.5 提升至 71.9。
* **泛化性**：在域外任务上的测试表明方法具有较好的泛化能力，模型在未见约束类型上仍能提升指令跟随能力。
* **消融验证**：消融实验证明奖励建模（结合规则奖励和概率奖励）和课程分解对性能提升至关重要，去掉任一组件均导致性能下降。
* **实验设置合理性**：实验覆盖不同规模和架构模型（1.5B 到 8B 参数），在多领域任务上全面评估，但未在更大规模模型（如 32B）上测试，存在一定局限性。

## Further Thoughts

自监督奖励建模的思路可扩展至其他任务（如对话或代码生成），通过内部信号减少外部依赖；课程学习策略在多任务学习（如多步推理）中具有潜力，通过动态调整难度可能进一步提升效率；硬约束与软约束的分离处理启发任务分解优化，或许可应用于多模态模型，通过分解图像和文本约束提升指令跟随能力。