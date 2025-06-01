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
description: "本文提出 Infi-MMR 框架，通过三阶段课程学习和强化学习系统性提升多模态小型语言模型的推理能力，在多个基准测试中取得最先进性能。"
---

> **Summary:** 本文提出 Infi-MMR 框架，通过三阶段课程学习和强化学习系统性提升多模态小型语言模型的推理能力，在多个基准测试中取得最先进性能。 

> **Keywords:** Multimodal Learning, Reinforcement Learning, Curriculum Learning, Reasoning, Small Language Models

**Authors:** Zeyu Liu, Yuhang Liu, Guanghao Zhu, Congkai Xie, Zhen Li, Jianbo Yuan, Xinyao Wang, Qing Li, Shing-Chi Cheung, Shengyu Zhang, Fei Wu, Hongxia Yang

**Institution(s):** The Hong Kong Polytechnic University, Zhejiang University, University of Electronic Science and Technology of China, Reallm Labs, The Hong Kong University of Science and Technology, Independent


## Problem Background

大型语言模型（LLMs）在推理能力上取得了显著进步，但将这种能力扩展到多模态大型语言模型（MLLMs），尤其是参数较小的多模态小型语言模型（MSLMs），面临三大挑战：
* 高质量多模态推理数据的稀缺，现有数据集多集中于简单任务，缺乏复杂推理问题和可验证答案；
* 视觉与文本数据融合导致基础推理能力下降，跨模态融合的复杂性干扰结构化推理；
* 直接应用强化学习可能生成冗长且不准确的推理步骤，影响模型可靠性。

## Method

* **框架概述**：提出 **Infi-MMR**，一个基于课程学习的渐进式规则驱动强化学习（RL）框架，通过三个阶段系统性提升 MSLMs 的多模态推理能力。
* **阶段一 - 基础推理激活（Foundational Reasoning Activation, FRA）**：使用高质量纯文本推理数据集（如数学问题集 DeepScaleR），通过强化学习激活模型的基础逻辑推理能力，避免多模态数据引入的干扰，奠定坚实推理基础。
* **阶段二 - 跨模态推理适应（Cross-Modal Reasoning Adaptation, CMRA）**：引入带有图像描述（caption）的多模态数据（如 ViRL39k 数据集），利用图像描述作为文本与视觉推理的桥梁，通过 RL 逐步将文本推理能力转移到多模态场景；图像描述由 Omnicaptioner 框架生成，确保视觉信息的结构化表达。
* **阶段三 - 多模态推理增强（Multimodal Reasoning Enhancement, MRE）**：使用无描述的多模态数据，消除对文本描述的依赖，强制模型直接从原始视觉输入中推理，减少语言偏见，提升纯跨模态推理能力。
* **强化学习算法**：采用 Group Relative Policy Optimization (GRPO) 算法，通过生成多个候选输出并基于规则奖励函数优化策略，减少对批评模型的依赖，降低计算成本。
* **奖励函数设计**：奖励函数结合格式正确性（检查推理过程是否符合结构要求）和答案准确性（针对数学、字符串和多选题分别定制评估方式），确保推理过程既规范又精准。

## Experiment

* **模型与性能**：基于 Qwen2.5-VL-3B-Instruct 训练的 **Infi-MMR-3B** 在多模态数学推理任务上取得最先进性能，例如 MathVerse testmini 达 43.68%，MathVision test 达 27.04%，OlympiadBench 达 21.33%，MathVista testmini 达 67.2%，显著优于基线模型及部分更大参数模型。
* **阶段性提升**：从 FRA 到 CMRA 再到 MRE 阶段，模型在多模态任务上的性能逐步提升，验证了课程学习策略的有效性。
* **实验设置合理性**：实验涵盖文本（MATH500）和多模态基准数据集，数据去污处理避免泄露，确保评估公平；消融研究表明初始文本训练优于直接多模态训练，caption-augmented 数据在推理深度上优于无描述数据。
* **局限性**：未深入探讨图像描述质量对结果的影响，可能存在潜在变量未被控制。

## Further Thoughts

课程学习策略从文本到多模态的渐进式训练可作为跨领域能力迁移的通用方法，例如从语言到代码生成；图像描述作为桥梁的思路启发在其他多模态任务中引入中间表示（如语音转录）以降低模态融合难度；规则驱动强化学习的设计表明可通过规则化方式减少对人工标注的依赖，对资源受限场景具有借鉴意义。