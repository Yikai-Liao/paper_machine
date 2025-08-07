---
title: "Beyond the Trade-off: Self-Supervised Reinforcement Learning for Reasoning Models' Instruction Following"
pubDatetime: 2025-08-04T07:48:59+00:00
slug: "2025-08-self-supervised-rl-if"
type: "arxiv"
id: "2508.02150"
score: 0.736281861348137
author: "grok-3-latest"
authors: ["Qingyu Ren", "Qianyu He", "Bowei Zhang", "Jie Zeng", "Jiaqing Liang", "Yanghua Xiao", "Weikang Zhou", "Zeye Sun", "Fei Yu"]
tags: ["LLM", "Reinforcement Learning", "Instruction Following", "Curriculum Learning", "Reward Modeling"]
institution: ["Shanghai Key Laboratory of Data Science, College of Computer Science and Artificial Intelligence, Fudan University", "School of Data Science, Fudan University", "Ant Group"]
description: "本文提出一种自监督强化学习框架，利用推理模型内部信号显著提升其指令遵循能力，同时维持推理性能，摆脱了对外部强模型的依赖。"
---

> **Summary:** 本文提出一种自监督强化学习框架，利用推理模型内部信号显著提升其指令遵循能力，同时维持推理性能，摆脱了对外部强模型的依赖。 

> **Keywords:** LLM, Reinforcement Learning, Instruction Following, Curriculum Learning, Reward Modeling

**Authors:** Qingyu Ren, Qianyu He, Bowei Zhang, Jie Zeng, Jiaqing Liang, Yanghua Xiao, Weikang Zhou, Zeye Sun, Fei Yu

**Institution(s):** Shanghai Key Laboratory of Data Science, College of Computer Science and Artificial Intelligence, Fudan University, School of Data Science, Fudan University, Ant Group


## Problem Background

推理模型在复杂问题解决中表现出色，但存在推理能力与指令遵循能力之间的显著权衡问题：推理模型在多约束指令遵循任务上表现不佳，而现有方法依赖外部强模型来提升指令遵循能力，导致方法学瓶颈（如学生模型受限于教师模型）和实际限制（如高成本和访问受限）；论文旨在通过自监督强化学习，利用模型内部信号提升指令遵循能力，摆脱对外部模型的依赖。

## Method

* **核心思想**：提出一种自监督强化学习（RL）框架，利用推理模型自身的内部信号提升其指令遵循能力，同时维持推理性能，避免依赖外部强模型。
* **数据集构建**：合成多约束指令数据集，涵盖硬约束（如格式要求）和软约束（如语义风格），并整合数学和科学领域的推理数据以维持模型通用能力；设计渐进约束课程（Incremental Constraint Curriculum），将复杂多约束指令分解为逐步增加约束数量的子指令，提供更密集的训练信号。
* **奖励建模**：针对硬约束和软约束分别设计奖励机制；硬约束通过程序化验证（Programmatic Verification）直接判断是否满足，输出二元奖励（0或1）；软约束通过自监督数据训练一个约束级二分类奖励模型（Constraint-Wise Binary Classification），利用模型自身生成的响应构建正负样本，预测响应满足约束的概率，提供细粒度奖励信号。
* **RL训练**：采用GRPO算法优化策略模型（Policy Model），结合硬约束和软约束的复合奖励信号进行训练；通过分析奖励动态和响应长度变化，确保模型在指令遵循和推理任务上的平衡。
* **关键创新**：自监督性（无需外部标注或强模型）、高效性（通过课程学习和约束分解解决稀疏奖励问题）以及对硬软约束的差异化处理。

## Experiment

* **有效性**：实验结果表明，方法显著提升了模型在指令遵循基准（如IFEval, CFBench, ComplexBench）上的表现，例如R1-0528-Qwen3-8B-IF在IFEval平均得分从80.7提升至87.6，在ComplexBench从66.7提升至71.1。
* **推理能力维持**：在推理任务基准（如GPQA, AIME, MMLU-Pro）上，模型性能基本不变，甚至在部分任务（如FOLIO, MMLU-Pro）上有所提升，证明方法未牺牲推理能力。
* **泛化性**：在域外任务上的测试显示模型对未见约束的指令仍具较好遵循能力，表明方法具有泛化性。
* **消融研究**：去掉规则奖励、概率奖励或渐进约束课程均导致性能下降，验证了各组件的必要性；奖励模型与人类标注一致性较高（Kendall Tau系数61.2），推理速度优于LLM-as-a-Judge方法。
* **实验设置合理性**：实验覆盖不同规模和架构模型（1.5B至8B参数），包含多种基准测试，并通过训练动态分析和消融研究深入验证方法有效性；不足之处在于未在更大规模模型（如32B）上测试，但现有结果显示出可扩展性潜力。

## Further Thoughts

自监督奖励建模的思路可推广至其他领域，如无监督偏好优化，通过模型自身信号减少对标注数据的依赖；渐进约束课程设计为解决稀疏奖励问题提供了新视角，可否应用于多目标优化或多步推理任务；硬软约束分离建模启发我们在其他NLP任务中针对性优化不同需求维度；冷启动阶段引入推理数据对模型行为的影响是否意味着早期数据选择可长期塑造模型能力，值得在创造性或安全性培养中进一步探索。