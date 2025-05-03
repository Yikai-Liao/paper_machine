---
title: "DeepCritic: Deliberate Critique with Large Language Models"
pubDatetime: 2025-05-01T17:03:17+00:00
slug: "2025-05-deepcritic-math-critique"
type: "arxiv"
id: "2505.00662"
score: 0.721612304706838
author: "grok-3-latest"
authors: ["Wenkai Yang", "Jingwen Chen", "Yankai Lin", "Ji-Rong Wen"]
tags: ["LLM", "Critique Model", "Reasoning", "Supervised Fine-Tuning", "Reinforcement Learning"]
institution: ["Renmin University of China", "Beijing Jiaotong University"]
description: "本文提出 DeepCritic 框架，通过两阶段训练（监督微调和强化学习）显著提升大型语言模型在数学推理任务中的批判能力，为自动化监督和模型自我改进提供了有效路径。"
---

> **Summary:** 本文提出 DeepCritic 框架，通过两阶段训练（监督微调和强化学习）显著提升大型语言模型在数学推理任务中的批判能力，为自动化监督和模型自我改进提供了有效路径。 

> **Keywords:** LLM, Critique Model, Reasoning, Supervised Fine-Tuning, Reinforcement Learning

**Authors:** Wenkai Yang, Jingwen Chen, Yankai Lin, Ji-Rong Wen

**Institution(s):** Renmin University of China, Beijing Jiaotong University


## Problem Background

随着大型语言模型（LLMs）能力的快速提升，提供准确且可扩展的监督成为迫切需求。
现有 LLM 批判模型在复杂任务（如数学推理）中生成的批判过于浅显，缺乏深度分析和批判性思维，导致判断准确性低，无法为生成模型提供有效反馈，限制了自动化监督的潜力。

## Method

*   **核心思想:** 提出 DeepCritic 框架，通过两阶段训练提升 LLM 的批判能力，使其能够在数学推理任务中生成深思熟虑、细致的批判。
*   **第一阶段 - 监督微调（SFT）:**
    *   使用 Qwen2.5-72B-Instruct 模型生成 4.5K 条长篇批判数据作为种子数据集。
    *   对每个推理步骤，首先生成初步批判（Initial Critique），分析步骤的正确性。
    *   随后生成深入批判（In-Depth Critique），从不同视角验证步骤或反思初步批判的缺陷。
    *   最后将初步和深入批判合并为一个完整的深思熟虑批判（Deliberate Critique），包含迭代评估、多视角验证和元批判（Meta-Critiquing）。
    *   在此数据上对目标模型（基于 Qwen2.5-7B-Instruct）进行监督微调，学习批判的结构和格式。
*   **第二阶段 - 强化学习（RL）:**
    *   在 SFT 模型基础上，通过强化学习进一步激励批判能力。
    *   RL 数据来源分为两种：一是人类标注数据集 PRM800K，二是通过蒙特卡洛采样（Monte Carlo Sampling）自动生成标注数据。
    *   奖励机制基于最终判断的准确性，鼓励模型生成更准确的批判。
*   **关键点:** 强调深思熟虑的过程，确保批判不仅是表面复述，而是包含深度分析和反思，同时探索自动化数据生成以减少对人类标注的依赖。

## Experiment

*   **有效性:** DeepCritic-7B-RL-PRM800K 模型在 6 个错误识别基准测试集中的 5 个上显著优于所有基线（包括 GPT-4o 和 DeepSeek-R1-Distill 模型），平均 F1 分数达到 67.1，相比基模型 Qwen2.5-7B-Instruct（34.1）提升约 33 个百分点。
*   **阶段性提升:** SFT 阶段将 F1 分数从 34.1 提升至 54.1，RL 阶段进一步提升至 63.5（自动数据）或 67.1（PRM800K 数据），显示出两阶段训练的累积效果。
*   **测试时扩展性:** 通过多数投票（Maj@8），平均 F1 分数从 67.1 提升至 70.5，表明模型具有良好的测试时计算扩展性。
*   **生成器辅助效果:** DeepCritic 作为验证器或反馈提供者，能显著提升生成模型（如 Qwen2.5-72B-Instruct）的性能，尤其在基于批判的精炼任务中，错误到正确的转化率（w→c）高于基线，体现弱模型监督强模型的潜力。
*   **实验设置合理性:** 实验覆盖多个数据集（MR-GSM8K, PRM800K, ProcessBench），包含不同难度和来源的问题，对比多种基线模型（PRMs 和 LLM 批判模型），采样参数和训练设置披露详细，整体设计全面且具可重复性。

## Further Thoughts

深思熟虑批判（Deliberate Critique）的理念通过多视角验证和元批判机制，模拟人类深度思考过程，是否可扩展至其他领域（如代码生成、文本分析）以提升模型自监督能力？
自动化 RL 数据生成（通过蒙特卡洛采样）减少了对人类标注的依赖，是否可以通过更复杂的自动化方法实现完全自我监督？
弱模型监督强模型的潜力提示，是否可以设计专注于特定任务的小型批判模型，以低成本监督更大规模生成模型？