---
title: "Infi-MMR: Curriculum-based Unlocking Multimodal Reasoning via Phased Reinforcement Learning in Multimodal Small Language Models"
pubDatetime: 2025-05-29T04:51:56+00:00
slug: "2025-05-infi-mmr-reasoning"
type: "arxiv"
id: "2505.23091"
score: 0.5104187933746268
author: "grok-3-latest"
authors: ["Zeyu Liu", "Yuhang Liu", "Guanghao Zhu", "Congkai Xie", "Zhen Li", "Jianbo Yuan", "Xinyao Wang", "Qing Li", "Shing-Chi Cheung", "Shengyu Zhang", "Fei Wu", "Hongxia Yang"]
tags: ["Multimodal Learning", "Reinforcement Learning", "Curriculum Learning", "Reasoning", "Small Language Models"]
institution: ["The Hong Kong Polytechnic University", "Zhejiang University", "University of Electronic Science and Technology of China", "Reallm Labs", "The Hong Kong University of Science and Technology", "Independent"]
description: "本文提出 Infi-MMR 框架，通过课程式强化学习的三阶段训练，系统性地提升多模态小型语言模型的推理能力，在多个多模态推理基准上取得最先进性能。"
---

> **Summary:** 本文提出 Infi-MMR 框架，通过课程式强化学习的三阶段训练，系统性地提升多模态小型语言模型的推理能力，在多个多模态推理基准上取得最先进性能。 

> **Keywords:** Multimodal Learning, Reinforcement Learning, Curriculum Learning, Reasoning, Small Language Models

**Authors:** Zeyu Liu, Yuhang Liu, Guanghao Zhu, Congkai Xie, Zhen Li, Jianbo Yuan, Xinyao Wang, Qing Li, Shing-Chi Cheung, Shengyu Zhang, Fei Wu, Hongxia Yang

**Institution(s):** The Hong Kong Polytechnic University, Zhejiang University, University of Electronic Science and Technology of China, Reallm Labs, The Hong Kong University of Science and Technology, Independent


## Problem Background

多模态小型语言模型（MSLMs）由于参数规模较小，在整合视觉信息与逻辑推理时面临显著挑战，包括高质量多模态推理数据的稀缺、视觉处理对基础推理能力的削弱，以及直接应用强化学习可能导致复杂但不准确的推理步骤等问题。
论文旨在通过系统性的训练框架，逐步解锁 MSLMs 的多模态推理潜力，解决上述关键问题。

## Method

* **核心思想：** 提出 Infi-MMR 框架，基于课程学习（Curriculum Learning）和规则驱动的强化学习（Reinforcement Learning），通过三个阶段逐步提升 MSLMs 的多模态推理能力。
* **阶段 1 - 基础推理激活（Foundational Reasoning Activation）：** 使用纯文本的高质量推理数据集（如 DeepScaleR，包含 39,000 个可验证的数学问题-答案对），通过强化学习增强模型的逻辑推理能力，避免直接引入多模态数据导致的干扰，为后续阶段奠定坚实基础。
* **阶段 2 - 跨模态推理适应（Cross-Modal Reasoning Adaptation）：** 引入带图像描述的多模态数据（caption-augmented multimodal data），利用工具（如 Omnicaptioner）生成图像描述，作为文本与视觉之间的桥梁，通过强化学习逐步将推理能力迁移到多模态场景，减少跨模态融合的复杂性影响。
* **阶段 3 - 多模态推理增强（Multimodal Reasoning Enhancement）：** 使用无描述的多模态数据（caption-free multimodal data，如 ViRL39k 数据集），强制模型直接从原始视觉输入中推理，消除对文本描述的依赖，减少语言偏见，进一步提升跨模态推理能力。
* **强化学习算法：** 采用 Group Relative Policy Optimization (GRPO) 算法，通过生成多个候选输出并计算组内相对优势，优化策略更新，减少对 Critic 模型的依赖，降低计算成本。
* **奖励函数设计：** 奖励函数结合输出格式正确性（R_format）和答案准确性（R_acc），通过加权方式引导模型先学习结构化输出，再优化推理准确性，确保训练稳定性。

## Experiment

* **有效性：** 基于 Qwen2.5-VL-3B-Instruct 训练的 Infi-MMR-3B 模型在多个多模态推理基准上取得最先进性能，例如 MathVerse testmini 准确率为 43.68%，MathVision test 为 27.04%，OlympiadBench 为 21.33%，显著优于基线模型和其他开源/专有模型。
* **逐阶段提升：** 从基础推理激活（FRA）到跨模态推理适应（CMRA）再到最终模型，各阶段性能逐步提升，验证了课程式训练的有效性，尤其是在多模态数学推理任务上的表现突出。
* **实验设置合理性：** 实验覆盖文本推理（MATH500）和多模态推理（MathVerse, MathVision 等）多个基准，采用数据去重和多模态嵌入相似性过滤的去污染措施，确保评估公平；但未深入探讨图像描述质量对结果的影响，且训练数据规模（39k 样本）可能限制泛化能力。
* **消融研究：** 初始阶段使用文本数据训练比直接用多模态数据更有效，避免了推理不稳定和输出冗长问题；带描述数据在迁移推理能力时优于无描述数据，但最终无描述数据训练对消除语言偏见至关重要。

## Further Thoughts

课程式训练从单一模态到多模态的逐步迁移策略非常具有启发性，是否可以推广到其他跨模态任务（如文本到语音或静态图像到视频推理）？此外，图像描述作为桥梁的作用提示我们，是否可以通过更高质量的描述生成模型进一步提升迁移效果？另一个想法是，在强化学习中引入动态调整的奖励机制，根据任务难度或模型当前能力调整奖励权重，以更精细地引导推理能力发展。