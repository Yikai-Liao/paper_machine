---
title: "MENLO: From Preferences to Proficiency -- Evaluating and Modeling Native-like Quality Across 47 Languages"
pubDatetime: 2025-09-30T17:48:58+00:00
slug: "2025-09-menlo-multilingual-evaluation"
type: "arxiv"
id: "2509.26601"
score: 0.6291675184874622
author: "grok-3-latest"
authors: ["Chenxi Whitehouse", "Sebastian Ruder", "Tony Zhiyang Lin", "Oksana Kurylo", "Haruka Takagi", "Janice Lam", "Nicolò Busetto", "Denise Diaz"]
tags: ["LLM", "Multilingual Evaluation", "Native-Like Quality", "Pairwise Evaluation", "Reinforcement Learning"]
institution: ["Meta Superintelligence Labs"]
description: "本文提出 MENLO 框架，通过成对评估和强化学习显著提升了大语言模型在 47 种语言中的类母语响应质量，并将评判模型作为奖励模型优化生成能力。"
---

> **Summary:** 本文提出 MENLO 框架，通过成对评估和强化学习显著提升了大语言模型在 47 种语言中的类母语响应质量，并将评判模型作为奖励模型优化生成能力。 

> **Keywords:** LLM, Multilingual Evaluation, Native-Like Quality, Pairwise Evaluation, Reinforcement Learning

**Authors:** Chenxi Whitehouse, Sebastian Ruder, Tony Zhiyang Lin, Oksana Kurylo, Haruka Takagi, Janice Lam, Nicolò Busetto, Denise Diaz

**Institution(s):** Meta Superintelligence Labs


## Problem Background

大语言模型（LLM）在多语言环境下的响应质量难以达到类母语水平，尤其是在真实对话场景中，现有的评估方法（如标准化测试或任务导向基准）难以扩展到多语言、多文化的复杂情境，且缺乏对‘类母语质量’的系统化定义和评估框架。
论文旨在解决如何系统化评估和提升 LLM 在多语言环境下的类母语响应质量这一关键问题。

## Method

*   **核心框架：MENLO（Multilingual Evaluation of Native-Like Output）**：基于受众设计（Audience Design）理论，将类母语质量分解为四个维度：流畅性（Fluency）、语调（Tone）、本地化语调（Localized Tone）和本地化事实性（Localized Factuality），以此评估和优化 LLM 响应。
*   **数据集构建**：创建包含 6,423 个提示-响应偏好对的 MENLO 数据集，覆盖 47 种语言变体，通过人类标注实现高一致性（Krippendorff’s α = 0.84）；提示设计基于本地化上下文，响应由先进 LLM（如 GPT-4o、Llama4-Maverick）生成。
*   **评估策略**：对比单点评估（Pointwise Evaluation，单独评分单个响应）和成对评估（Pairwise Evaluation，同时评分两个响应），并测试标注准则（Rubrics）的作用，探索零样本（Zero-shot）和少样本（Few-shot）设置下的评判效果。
*   **评判模型训练**：通过监督微调（Supervised Fine-Tuning, SFT）和强化学习（Reinforcement Learning, RL）训练 LLM 作为评判模型（Judge），如 Qwen3-4B 和 Llama4-Scout；RL 采用复合奖励机制，包括点式奖励（匹配真实评分）、平滑奖励（接近真实评分）、偏好奖励（相对评分方向一致）及格式惩罚，确保评判准确性和推理能力。
*   **生成优化**：将 RL 训练的评判模型作为生成式奖励模型（Reward Model），通过 RL 后训练（Post-Training）提升策略模型（Policy Model，如 Qwen3-4B）的多语言响应质量，采用 GRPO 算法并采样多轮生成以计算奖励。
*   **关键创新**：不直接依赖绝对评分，而是通过成对比较和结构化准则减少主观性；同时将评判与生成优化统一在一个框架内，实现可扩展的多语言评估与改进。

## Experiment

*   **评估效果**：成对评估在零样本设置下显著优于单点评估（Macro-F1 提升高达 12.4%，偏好准确率提升高达 18.0%）；使用标注准则进一步提升评判性能，尤其在单点评估中（Macro-F1 平均提升 4.3%）。
*   **训练效果**：RL 训练的评判模型（如 Llama4-Scout）在多任务设置下表现最佳，性能接近人类标注者（Krippendorff’s α 接近人类水平）；相比 SFT，RL 在 Qwen3-4B 上提升 Macro-F1 约 4.0%，偏好准确率提升 2.9%。
*   **生成优化效果**：将 RL 评判模型作为奖励模型后训练 Qwen3-4B，显著提升响应质量（自动评判得分提升 0.8-1.16，人类评判提升 0.36，胜率达 55.7%-77.9%）；但自动评判高估改进幅度（比人类评判高约 0.5 分）。
*   **实验设置合理性**：实验覆盖 47 种语言变体，数据量大（81,014 条标注），对比充分（零样本 vs. 微调、单点 vs. 成对、不同模型架构），结果在大多数维度（如流畅性、语调）上显著；但本地化事实性维度表现较差，表明方法在处理复杂本地知识时有局限。
*   **开销与局限**：训练和评判过程计算成本较高，尤其 RL 训练需多 GPU 支持；某些语言和维度（如 Localized Factuality）改进有限，需进一步探索外部知识整合。

## Further Thoughts

成对评估（Pairwise Evaluation）作为一种更有效的评判方式，可能适用于其他主观性强的任务（如情感分析、风格迁移），其通过相对比较减少评分偏差的思路值得借鉴；基于受众设计的提示构建方法，为多语言模型的文化适应性提供了新思路，未来可结合上下文自适应生成进一步优化；RL 评判模型作为奖励模型直接提升生成质量的策略，展示了评估与优化的统一潜力，启发我们探索是否能结合外部知识库或检索增强生成（RAG）来改进本地化事实性评估。